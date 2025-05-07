import os
import torch
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
import torch.nn.functional as F
import csv
import numpy as np
from scipy import stats
import sys
import wandb
import warnings
import json

np.set_printoptions(threshold=sys.maxsize)

class RawDataLoggingCallback(Callback):
    def __init__(self, handle_nans='drop'):
        """
        Initialize the callback with NaN handling options.
        
        Args:
            handle_nans (str): Strategy for handling NaNs. Options:
                - 'drop': Remove NaN values before calculations (default)
                - 'zero': Replace NaN values with zeros
                - 'mean': Replace NaN values with mean of non-NaN values
        """
        self.validation_filename = "validation.csv"
        self.test_filename = "test.csv"
        self.predict_filename = "predict.csv"

        self.validation_plot_path_prefix = "validation"  # We'll add "_regionName.png"
        self.test_plot_path_prefix = "test"              # We'll add "_regionName.png"

        self.actual_epoch = -1  # Start at -1 to indicate pre-training steps
        self.handle_nans = handle_nans
        
        # Store region names when they're first encountered
        self.region_names = None

    def aggregate_dicts(self, dicts):
        aggregated_dict = {}
        for d in dicts:
            for key, value in d.items():
                if key in aggregated_dict:
                    aggregated_dict[key].extend(value)
                else:
                    # copy to avoid modifying the original list
                    aggregated_dict[key] = value.copy()
        return aggregated_dict
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.actual_epoch += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Store region names if available and not already stored
        _, _, metadata = batch
        if self.region_names is None and metadata is not None and 'regions' in metadata:
            region_names = [x[0] for x in metadata['regions']]
            self.region_names = region_names
            
        # Log metadata only if single-GPU / single-process
        log_metadata = (trainer.world_size == 1)
        self.gather_and_log(
            trainer,
            pl_module,
            outputs,
            batch,
            self.validation_filename,
            log_metadata=log_metadata,
            log_sequence=False
        )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Store region names if available and not already stored
        _, _, metadata = batch
        if self.region_names is None and metadata is not None and 'regions' in metadata:
            region_names = [x[0] for x in metadata['regions']]
            self.region_names = region_names
            
        # Log metadata only if single-GPU / single-process
        log_metadata = (trainer.world_size == 1)
        self.gather_and_log(
            trainer,
            pl_module,
            outputs,
            batch,
            self.test_filename,
            log_metadata=log_metadata,
            log_sequence=False
        )

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Store region names if available and not already stored
        _, _, metadata = batch
        if self.region_names is None and metadata is not None and 'regions' in metadata:
            region_names = [x[0] for x in metadata['regions']]
            self.region_names = region_names
            
        # Log metadata only if single-GPU / single-process
        log_metadata = (trainer.world_size == 1)
        self.gather_and_log(
            trainer,
            pl_module,
            outputs,
            batch,
            self.predict_filename,
            log_metadata=log_metadata,
            log_sequence=False
        )

    def gather_and_log(self, trainer, pl_module, outputs, batch, filename, log_metadata=False, log_sequence=False):
        input_, target, metadata = batch
        # In multi-output regression, `outputs["prediction"]` might have shape (batch_size, k)
        prediction = outputs["prediction"]

        # Gather data from all processes
        inputs = pl_module.all_gather(input_)
        targets = pl_module.all_gather(target)
        predictions = pl_module.all_gather(prediction)

        current_epoch = self.actual_epoch

        # Only rank 0 writes to CSV
        if trainer.global_rank == 0:
            self.log_data(current_epoch, inputs, targets, predictions, metadata,
                          filename,
                          log_metadata=log_metadata,
                          log_sequence=log_sequence)

    def on_validation_epoch_end(self, trainer, pl_module):
        # We'll produce multiple scatterplots, one per output dimension
        self.compute_scatterplot(
            file_prefix=self.validation_plot_path_prefix,
            category="validation",
            filename=self.validation_filename
        )
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.compute_scatterplot(
            file_prefix=self.test_plot_path_prefix,
            category="test",
            filename=self.test_filename
        )

    @rank_zero_only
    def compute_scatterplot(self, file_prefix, category, filename):
        """
        For multi-output regression, we expect columns in the CSV like:
          epoch, 
          targets_Region1, targets_Region2, ..., targets_RegionK,
          predictions_Region1, predictions_Region2, ..., predictions_RegionK,
          gene_id (optional metadata for averaging)
        We will:
          1) Read CSV
          2) Filter to most recent epoch
          3) If gene_id column exists, average rows with the same gene_id
          4) For each dimension i with region name:
             - scatter plot predictions_region vs. targets_region
             - compute correlation + R^2
             - wandb.log() + save figure
        """
        if not os.path.exists(filename):
            print(f"[WARNING] CSV file {filename} does not exist, cannot plot.")
            return

        data = pd.read_csv(filename)
        if "epoch" not in data.columns:
            print(f"[WARNING] No epoch column in {filename}, cannot plot.")
            return

        # Filter to the most recent epoch
        most_recent_epoch = data["epoch"].max()
        recent_data = data[data["epoch"] == most_recent_epoch]

        # Figure out how many outputs we have by scanning the columns
        # Now we look for all columns that match the pattern "targets_*"
        target_cols = [col for col in recent_data.columns if col.startswith("targets_")]
        pred_cols = [col for col in recent_data.columns if col.startswith("predictions_")]

        # Sort them to ensure targets and predictions align
        target_cols.sort()
        pred_cols.sort()
        
        # Get region names - first try the instance variable, then try from the CSV
        region_names = self.region_names
        
        # If region_names is None, try to get from CSV as fallback
        if region_names is None and 'regions' in recent_data.columns:
            # If regions is a single value (serialized list), use the first row
            regions_data = recent_data['regions'].iloc[0]
            try:
                # Try to parse as JSON if it's a serialized list
                region_names = json.loads(regions_data)
            except (json.JSONDecodeError, TypeError):
                # If not valid JSON and it's a string, it might be a custom format
                if isinstance(regions_data, str):
                    # Try to handle common string formats like "[a, b, c]" or "a, b, c"
                    regions_str = regions_data.strip('[]').split(',')
                    region_names = [r.strip() for r in regions_str]
        
        # Check if gene_id column exists and aggregate data if it does
        if 'gene_id' in recent_data.columns:
            print(f"Found gene_id column in {filename}, averaging rows with the same gene_id")
            
            # Select only the columns we need for analysis
            analysis_cols = ['gene_id'] + target_cols + pred_cols
            data_for_analysis = recent_data[analysis_cols]
            
            # Group by gene_id and calculate the mean for each target and prediction column
            aggregated_data = data_for_analysis.groupby('gene_id').mean().reset_index()
            
            # Log how many unique genes we have after aggregation
            original_count = len(recent_data)
            aggregated_count = len(aggregated_data)
            print(f"Aggregated {original_count} rows into {aggregated_count} unique gene_ids")
            
            # Use the aggregated data for plotting
            plot_data = aggregated_data
        else:
            # No gene_id column, use the data as is
            print(f"No gene_id column found in {filename}, using all rows as independent observations")
            plot_data = recent_data
        
        # We'll loop over each dimension
        for i, (tcol, pcol) in enumerate(zip(target_cols, pred_cols)):
            # Create scatter plot for dimension i
            tvals = plot_data[tcol].values
            pvals = plot_data[pcol].values

            # Check for NaNs and handle them according to strategy
            nan_mask = np.isnan(tvals) | np.isnan(pvals)
            nan_count = np.sum(nan_mask)
            
            if nan_count > 0:
                total_count = len(tvals)
                nan_percentage = (nan_count / total_count) * 100
                print(f"[WARNING] Found {nan_count} NaN values ({nan_percentage:.2f}%) in dimension {i}")
                
                # Handle NaNs according to strategy
                if self.handle_nans == 'drop':
                    valid_mask = ~nan_mask
                    tvals = tvals[valid_mask]
                    pvals = pvals[valid_mask]
                    print(f"Dropped {nan_count} NaN values for dimension {i}")
                elif self.handle_nans == 'zero':
                    tvals = np.nan_to_num(tvals, nan=0.0)
                    pvals = np.nan_to_num(pvals, nan=0.0)
                    print(f"Replaced {nan_count} NaN values with zeros for dimension {i}")
                elif self.handle_nans == 'mean':
                    t_mean = np.nanmean(tvals) if not np.isnan(np.nanmean(tvals)) else 0.0
                    p_mean = np.nanmean(pvals) if not np.isnan(np.nanmean(pvals)) else 0.0
                    tvals = np.nan_to_num(tvals, nan=t_mean)
                    pvals = np.nan_to_num(pvals, nan=p_mean)
                    print(f"Replaced {nan_count} NaN values with means for dimension {i}")

            # Compute correlation + r2 (safely handling empty arrays or arrays with single value)
            if len(tvals) <= 1 or len(np.unique(tvals)) <= 1 or len(np.unique(pvals)) <= 1:
                # Not enough data for meaningful statistics
                corr = np.nan
                r_squared = np.nan
                r2_sklearn = np.nan
                slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan
            else:
                try:
                    # np.corrcoef returns the correlation matrix [[1, corr], [corr, 1]]
                    corr = np.corrcoef(tvals, pvals)[0, 1]
                    r_squared = corr ** 2
                    
                    # Calculate sklearn R2 score with NaN handling
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        r2_sklearn = r2_score(tvals, pvals)
                    
                    # Linear regression using scipy
                    slope, intercept, r_value, p_value, std_err = stats.linregress(pvals, tvals)
                except Exception as e:
                    print(f"[WARNING] Error computing statistics for dimension {i}: {str(e)}")
                    corr = np.nan
                    r_squared = np.nan
                    r2_sklearn = np.nan
                    slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                spearman_corr, spearman_p = stats.spearmanr(tvals, pvals)

            # Scatter plot
            plt.figure(figsize=(8, 6))
            plt.scatter(pvals, tvals, alpha=0.5, label="Data")
            
            # Use region name for labels if available
            region_label = f"Region {i}"
            if region_names is not None and i < len(region_names):
                region_label = region_names[i]
                
            # Extract the region name directly from the column name if it's in the new format
            # e.g., "targets_RegionName" or "predictions_RegionName"
            if '_' in tcol:
                region_from_col = tcol.split('_', 1)[1]
                if region_from_col != str(i):  # If it's not just a number
                    region_label = region_from_col
                
            plt.xlabel(f"Predictions ({region_label})")
            plt.ylabel(f"Targets ({region_label})")
            
            # Add gene_id to title if we're using aggregated data
            title_prefix = f"{category.capitalize()} Scatter (Epoch {most_recent_epoch}"
            if 'gene_id' in plot_data.columns:
                title_prefix += ", Gene-averaged"
            title = f"{title_prefix}, {region_label})"
            plt.title(title)

            # Only plot regression line if we have valid statistics
            if not np.isnan(slope) and not np.isnan(intercept):
                line = slope * pvals + intercept
                plt.plot(pvals, line, color='r', label=f'y={slope:.2f}x+{intercept:.2f}')

            # Annotate R^2
            stats_text = f"Pearson R = {corr:.3f}\nR^2 = {r_squared:.3f}\nR2_sklearn = {r2_sklearn:.3f}\nSpearman R = {spearman_corr:.3f}"
            if nan_count > 0:
                stats_text += f"\nNaNs: {nan_count}/{len(tvals) + nan_count} ({nan_percentage:.1f}%)"
            
            # Add count of data points (especially important when averaging by gene_id)
            stats_text += f"\nData points: {len(tvals)}"
            
            plt.annotate(stats_text,
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         ha='left', va='top', fontsize=10)
            
            plt.legend()

            # Save the plot with region name
            safe_region_label = region_label.replace(' ', '_').replace('/', '_').replace('\\', '_')
            plot_type = "gene_averaged" if 'gene_id' in plot_data.columns else "raw"
            out_path = f"{file_prefix}_{safe_region_label}_{plot_type}.png"
            plt.savefig(out_path)
            
            # Log to W&B (only valid metrics)
            metrics_dict = {}
            metric_prefix = f"{category}_gene_avg" if 'gene_id' in plot_data.columns else category
            
            if not np.isnan(r2_sklearn):
                metrics_dict[f"{metric_prefix}_R^2_{region_label}"] = r2_sklearn
            if not np.isnan(corr):
                metrics_dict[f"{metric_prefix}_corr_{region_label}"] = corr
            
            # Always log the image
            metrics_dict[f"{metric_prefix}_scatter_{region_label}"] = wandb.Image(plt)
            
            # Add NaN percentage if applicable
            if nan_count > 0:
                metrics_dict[f"{metric_prefix}_nan_percentage_{region_label}"] = nan_percentage
                
            wandb.log(metrics_dict)
            plt.close()

    @rank_zero_only
    def log_data(self, current_epoch, inputs, targets, predictions, metadata, filename,
                 log_metadata=False, log_sequence=False):

        """
        For multi-output regression, we assume:
          - targets shape: (batch_size, k)
          - predictions shape: (batch_size, k)
        
        We'll create columns with region names:
           targets_RegionName1, targets_RegionName2, ...
           predictions_RegionName1, predictions_RegionName2, ...
        """
        # Convert from torch -> numpy if needed and flatten batch dimension
        flattened_inputs = self.flatten_tensor(inputs)  # shape depends on your model
        flattened_targets = self.flatten_multi_output(targets)  # shape (n_samples, k)
        flattened_predictions = self.flatten_multi_output(predictions)  # shape (n_samples, k)

        # Count NaNs in targets and predictions
        nan_count_targets = np.isnan(flattened_targets).sum()
        nan_count_preds = np.isnan(flattened_predictions).sum()
        
        if nan_count_targets > 0 or nan_count_preds > 0:
            total_elements = flattened_targets.size + flattened_predictions.size
            nan_percentage = ((nan_count_targets + nan_count_preds) / total_elements) * 100
            print(f"[WARNING] Found {nan_count_targets} NaNs in targets and {nan_count_preds} NaNs in predictions ({nan_percentage:.2f}%)")
        
        # Start building dict for DataFrame
        # We do not store 'inputs' by default unless you want to see them
        data = {
            "epoch": [current_epoch] * len(flattened_targets),
        }

        # Extract region names from metadata if available
        region_names = None
        if metadata is not None and 'regions' in metadata:
            region_names = [x[0] for x in metadata['regions']]
        
        # Add columns for each target dimension with region names if available
        num_outputs = flattened_targets.shape[1]
        for j in range(num_outputs):
            # Use region name if available, otherwise use index number
            if region_names is not None and j < len(region_names):
                region_label = region_names[j]
                # Replace spaces and special characters in column names
                safe_label = region_label.replace(' ', '_').replace('/', '_').replace('\\', '_')
                data[f"targets_{safe_label}"] = flattened_targets[:, j]
                data[f"predictions_{safe_label}"] = flattened_predictions[:, j]
            else:
                data[f"targets_{j}"] = flattened_targets[:, j]
                data[f"predictions_{j}"] = flattened_predictions[:, j]

        # Optionally, store the input sequences if log_sequence=True
        if log_sequence:
            # For example, flatten_inputs might be shape (n_samples, seq_len) or (n_samples,)
            # We'll just store them in a single column or multiple columns.
            # Adjust as you see fit.  Here we do a single column:
            data["inputs"] = list(flattened_inputs)  # each entry is a python list

        # If you want to include metadata columns:
        if log_metadata and metadata is not None:
            # Special handling for 'regions' key
            if 'regions' in metadata:
                regions = metadata['regions']
                # Convert to JSON string if it's a list or other complex type
                if isinstance(regions, (list, tuple, dict)):
                    data['regions'] = [json.dumps(regions)] * len(flattened_targets)
                else:
                    data['regions'] = [regions] * len(flattened_targets)
            
            # Handle other metadata
            for key, value in metadata.items():
                if key == 'regions':
                    continue  # Already handled above
                    
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                    data[key] = value.tolist()
                else:
                    # If it's just a scalar or string or list, put it directly
                    data[key] = value

        # Create DataFrame and log NaN statistics
        df = pd.DataFrame(data)
        
        # Write to CSV
        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

    def flatten_tensor(self, tensor):
        """
        Flattens along the batch dimension (and possibly sequence),
        returning something you can store in a list or DataFrame.
        This is just an example: adapt to your shape if needed.
        """
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            tensor = tensor.detach().cpu().numpy()

        # Suppose 'tensor' could be shape (batch, sequence_len) or just (batch,)
        # We can reshape to (batch, -1)
        # Then each row will be one sample's data (possibly a sequence).
        if tensor.ndim > 1:
            tensor = tensor.reshape(tensor.shape[0], -1)
        return tensor  # shape (batch_size, something)

    def flatten_multi_output(self, tensor):
        """
        For multi-output regression, we assume shape is (batch_size, k)
        or possibly (batch_size, 1, k). We flatten to 2D: (n_samples, k).
        """
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            tensor = tensor.detach().cpu().numpy()

        # Squeeze out extra dimensions if needed.  For example,
        # if we have (batch_size, 1, k), we want (batch_size, k).
        tensor = np.squeeze(tensor)
        # Make sure final shape is (n_samples, k).
        if tensor.ndim == 1:
            # Means there's only 1 output dimension: reshape to (n_samples, 1)
            tensor = np.expand_dims(tensor, axis=-1)

        return tensor  # shape (batch_size, k)

    def check_for_inf_nan(self, tensor, name="tensor"):
        """Utility method to check for and report inf/nan values in tensors."""
        if isinstance(tensor, torch.Tensor):
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            if has_nan or has_inf:
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                total = tensor.numel()
                print(f"[WARNING] {name} contains {nan_count} NaNs and {inf_count} infs out of {total} elements")
                return True
        elif isinstance(tensor, np.ndarray):
            has_nan = np.isnan(tensor).any()
            has_inf = np.isinf(tensor).any()
            if has_nan or has_inf:
                nan_count = np.isnan(tensor).sum()
                inf_count = np.isinf(tensor).sum()
                total = tensor.size
                print(f"[WARNING] {name} contains {nan_count} NaNs and {inf_count} infs out of {total} elements")
                return True
        return False