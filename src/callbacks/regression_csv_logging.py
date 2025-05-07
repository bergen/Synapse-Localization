import os
import torch
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
import torch.nn.functional as F
import csv
import numpy as np
from scipy import stats
import sys
import wandb
np.set_printoptions(threshold=sys.maxsize)

class RawDataLoggingCallback(Callback):
    def __init__(self):
        self.validation_filename = "validation.csv"
        self.test_filename = "test.csv"
        self.predict_filename = "predict.csv"

        self.validation_plot_path = "validation.png"
        self.test_plot_path = "test.png"

        self.actual_epoch = -1  # Start at -1 to indicate pre-training steps

    def aggregate_dicts(self, dicts):
        aggregated_dict = {}
        for d in dicts:
            for key, value in d.items():
                if key in aggregated_dict:
                    aggregated_dict[key].extend(value)
                else:
                    aggregated_dict[key] = value.copy() # to avoid modifying the original list
        return aggregated_dict
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.actual_epoch += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        if trainer.world_size==1:
            log_metadata=True
        else:
            log_metadata=False
        self.gather_and_log(trainer, pl_module, outputs, batch, self.validation_filename, log_metadata=log_metadata)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        if trainer.world_size==1:
            log_metadata=True
        else:
            log_metadata=False
        self.gather_and_log(trainer, pl_module, outputs, batch, self.test_filename, log_metadata=log_metadata)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        if trainer.world_size==1:
            log_metadata=True
        else:
            log_metadata=False
        self.gather_and_log(trainer, pl_module, outputs, batch, self.predict_filename, log_metadata=log_metadata, log_sequence=False)


    def gather_and_log(self, trainer, pl_module, outputs, batch, filename, log_metadata=False, log_sequence=False):
        input, target, metadata = batch
        prediction = outputs["prediction"]

        # Gather data from all processes
        inputs = pl_module.all_gather(input)
        targets = pl_module.all_gather(target)
        predictions = pl_module.all_gather(prediction)
        # IMPORTANT: we only log metadata when we are running on a single gpu
        
        current_epoch = self.actual_epoch

        # Rank 0 writes to CSV
        if trainer.global_rank == 0:
            self.log_data(current_epoch, inputs, targets, predictions, metadata, filename, log_metadata=log_metadata, log_sequence=log_sequence)


    def on_validation_epoch_end(self, trainer, pl_module):
        self.compute_scatterplot(self.validation_plot_path, "validation", self.validation_filename)
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.compute_scatterplot(self.test_plot_path, "test", self.test_filename)


    @rank_zero_only
    def compute_scatterplot(self, plot_path, category, filename):
        data = pd.read_csv(filename)
        data['predictions'] = data['predictions'].apply(lambda x: float(x.strip('[]')))

        # Get the most recent epoch
        most_recent_epoch = data['epoch'].max()
        
        # Filter data for the most recent epoch
        recent_data = data[data['epoch'] == most_recent_epoch]

        correlation = recent_data['targets'].corr(recent_data['predictions'])
        
        # Compute r^2
        r_squared = correlation ** 2
        
        # Create scatterplot
        plt.figure(figsize=(10, 6))
        plt.scatter(recent_data['predictions'], recent_data['targets'], alpha=0.5)
        plt.xlabel("Predictions")
        plt.ylabel("Log Counts")
        plt.title(f'Scatterplot of Predictions vs Log Counts (Epoch {most_recent_epoch})')
        
        # Add regression line
        x = recent_data['predictions']
        y = recent_data['targets']
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line = slope * x + intercept
        plt.plot(x, line, color='r', label=f'y={slope:.2f}x+{intercept:.2f}')
        
        # Add r^2 information to plot
        plt.annotate(f'R^2: {r_squared:.2f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    ha='left', va='top')
        
        plt.legend()
        # Save the plot as an image file
        wandb.log({category + "_R^2": wandb.Image(plt)})
        plt.savefig(plot_path)

    @rank_zero_only
    def log_data(self, current_epoch, inputs, targets, predictions, metadata, filename, log_metadata=False, log_sequence=False):

        # Flatten the inputs and outputs along the batch dimension
        flattened_inputs = self.flatten_tensor(inputs)
        flattened_targets = self.flatten_degenerate(targets)
        flattened_predictions = self.flatten_tensor(predictions)

        # Create a DataFrame and write to CSV
        data = {
            'epoch': [current_epoch] * len(flattened_inputs),
            'targets': flattened_targets,
            'predictions': flattened_predictions,
        }

        if log_sequence:
            data['inputs'] = flattened_inputs
        
        if log_metadata:
            for key, value in metadata.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()  # Ensure it's a CPU tensor
                    data[key] = value.tolist()  # Convert to list for DataFrame compatibility
                else:
                    data[key] = value

        df = pd.DataFrame(data)
        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

    def flatten_tensor(self, tensor):
        # Convert the tensor to numpy and reshape
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            tensor = tensor.detach().cpu().numpy()
        
        if tensor.ndim == 3:
            tensor = tensor.reshape(-1, tensor.shape[2])
        return [t for t in tensor]

    def flatten_degenerate(self, tensor):
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            tensor = tensor.detach().cpu().numpy()

        tensor = np.squeeze(tensor, axis=-1)

        if tensor.ndim == 2:
            tensor = tensor.reshape(-1)
        
        return tensor.tolist()
    