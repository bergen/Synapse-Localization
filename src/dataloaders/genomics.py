# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
# Adapted from https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
from pathlib import Path
from typing import Any, List, Union
import os
import glob
import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import Dataset

from src.dataloaders.base import SequenceDataset, default_data_path
from src.dataloaders.fault_tolerant_sampler import RandomFaultTolerantSampler
from src.dataloaders.fault_tolerant_sampler import FaultTolerantDistributedSampler
# genomics datasets
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.datasets.mrna_dataset import TranscriptDataset
from src.dataloaders.datasets.synapse_classification import SynapseClassificationDataset
from src.dataloaders.datasets.mutation_scoring import MutationScoring
from src.dataloaders.datasets.utr_classification import UTRClassificationDataset
from src.dataloaders.datasets.synapse_regression import SynapseRegressionDataset
from src.dataloaders.datasets.codon_scoring import CodonScoring
from src.dataloaders.datasets.translation_efficiency import TranslationEfficiencyDataset

"""

Dataloaders for genomics datasets, including pretraining and downstream tasks.  First works in HyenaDNA project, May 2023.

"""

class MRNADataloader(SequenceDataset):
    """
    Base class, other dataloaders can inherit from this class.

    You must implement the following functions:
        - __init__
        - setup

    You can then use (already have access to) the following functions:
        - train_dataloader
        - val_dataloader
        - test_dataloader

    """
    _name_ = "mrna"  # this name is how the dataset config finds the right dataloader

    def __init__(self, fasta_directory, tokenizer_name=None, dataset_config_name=None, max_length=1024, d_output=2, rc_aug=False,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357, use_fixed_len_val=False,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, use_padding=True,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.fasta_directory = fasta_directory
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.use_fixed_len_val = use_fixed_len_val
        self.fault_tolerant = fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        self.use_padding=use_padding

        

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("**using pretrained AIRI tokenizer**")
            self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()  # creates the datasets.  You can also just create this inside the setup() here.


    def init_datasets(self):
        """Init the datasets (separate from the tokenizer)"""

        # delete old datasets to free memory
        if hasattr(self, 'dataset_train'):
            del self.dataset_train

        if hasattr(self, 'dataset_val'):
            del self.dataset_val

        # Initialize empty lists to store the datasets for each split
        datasets_train, datasets_val = [], []

        fasta_train = os.path.join(self.fasta_directory, "filtered_training.fasta")

        datasets_train.append(TranscriptDataset(fasta_file=fasta_train,
                                                max_length=self.max_length,
                                                tokenizer=self.tokenizer,
                                                tokenizer_name=self.tokenizer_name,
                                                add_eos=self.add_eos,
                                                use_padding=self.use_padding))

        fasta_val = os.path.join(self.fasta_directory, "pretraining_validation.fasta")

        datasets_val.append(TranscriptDataset(fasta_file=fasta_val, 
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            use_padding=self.use_padding))

        # Concatenate the datasets for each split
        self.dataset_train = torch.utils.data.ConcatDataset(datasets_train)
        self.dataset_val = torch.utils.data.ConcatDataset(datasets_val)
        # fix this: we currently don't have a test set
        self.dataset_test = torch.utils.data.ConcatDataset(datasets_val)

        return

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet



class SynapseClassificationDataloader(SequenceDataset):
    """
    You must implement the following functions:
        - __init__
        - setup

    You can then use (already have access to) the following functions:
        - train_dataloader
        - val_dataloader
        - test_dataloader

    """
    _name_ = "synapse_classification"  # this name is how the dataset config finds the right dataloader
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, csv_path=None, train_fasta_path=None, validation_fasta_path=None, test_fasta_path=None, predict_fasta_path=None, synapse_label_name=False, tokenizer_name=None, dataset_config_name=None, max_length=1024, d_output=2, rc_aug=False,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357, use_fixed_len_val=False,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, upsample=False, use_padding=True,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.csv_path = csv_path
        self.train_fasta_path = train_fasta_path
        self.validation_fasta_path = validation_fasta_path
        self.predict_fasta_path = predict_fasta_path
        self.synapse_label_name = synapse_label_name
        self.test_fasta_path = test_fasta_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.use_fixed_len_val = use_fixed_len_val
        self.fault_tolerant = fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        self.upsample=upsample
        self.use_padding=use_padding

        

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("**using pretrained AIRI tokenizer**")
            self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()  # creates the datasets.  You can also just create this inside the setup() here.


    def init_datasets(self):
        """Init the datasets (separate from the tokenizer)"""

        # delete old datasets to free memory
        if hasattr(self, 'dataset_train'):
            del self.dataset_train

        if hasattr(self, 'dataset_val'):
            del self.dataset_val

        # Initialize empty lists to store the datasets for each split


        return

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        self.dataset_train = SynapseClassificationDataset(csv_path=self.csv_path,
                                                fasta_path=self.train_fasta_path,
                                                synapse_label_name=self.synapse_label_name,
                                                max_length=self.max_length,
                                                tokenizer=self.tokenizer,
                                                tokenizer_name=self.tokenizer_name,
                                                add_eos=self.add_eos,
                                                upsample=self.upsample,
                                                use_padding=self.use_padding)
        
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler,
                                 )

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        self.dataset_val = SynapseClassificationDataset(csv_path=self.csv_path,
                                            fasta_path=self.validation_fasta_path,
                                            synapse_label_name=self.synapse_label_name,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            upsample=False,
                                            use_padding=self.use_padding)
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        self.dataset_test = SynapseClassificationDataset(csv_path=self.csv_path,
                                            fasta_path=self.test_fasta_path,
                                            synapse_label_name=self.synapse_label_name,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            upsample=False,
                                            use_padding=self.use_padding)
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)
    
    def predict_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        self.dataset_predict = SynapseClassificationDataset(csv_path=self.csv_path,
                                            fasta_path=self.predict_fasta_path,
                                            synapse_label_name=self.synapse_label_name,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            upsample=False,
                                            use_padding=self.use_padding)
        return self._data_loader(self.dataset_predict, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet



class SynapseRegressionDataloader(SequenceDataset):
    """
    You must implement the following functions:
        - __init__
        - setup

    You can then use (already have access to) the following functions:
        - train_dataloader
        - val_dataloader
        - test_dataloader

    """
    _name_ = "synapse_regression"  # this name is how the dataset config finds the right dataloader
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, csv_path=None, train_fasta_path=None, validation_fasta_path=None, test_fasta_path=None, predict_fasta_path=None, synapse_label_name=False, tokenizer_name=None, dataset_config_name=None, max_length=1024, d_output=2, rc_aug=False,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357, use_fixed_len_val=False,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, upsample=False, use_padding=True,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.csv_path = csv_path
        self.train_fasta_path = train_fasta_path
        self.validation_fasta_path = validation_fasta_path
        self.predict_fasta_path = predict_fasta_path
        self.synapse_label_name = synapse_label_name
        self.test_fasta_path = test_fasta_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.use_fixed_len_val = use_fixed_len_val
        self.fault_tolerant = fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        self.upsample=upsample
        self.use_padding=use_padding

        

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("**using pretrained AIRI tokenizer**")
            self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()  # creates the datasets.  You can also just create this inside the setup() here.


    def init_datasets(self):
        """Init the datasets (separate from the tokenizer)"""

        # delete old datasets to free memory
        if hasattr(self, 'dataset_train'):
            del self.dataset_train

        if hasattr(self, 'dataset_val'):
            del self.dataset_val

        # Initialize empty lists to store the datasets for each split


        return

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        self.dataset_train = SynapseRegressionDataset(csv_path=self.csv_path,
                                                fasta_path=self.train_fasta_path,
                                                synapse_label_name=self.synapse_label_name,
                                                max_length=self.max_length,
                                                tokenizer=self.tokenizer,
                                                tokenizer_name=self.tokenizer_name,
                                                add_eos=self.add_eos,
                                                upsample=self.upsample,
                                                use_padding=self.use_padding)
        
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler,
                                 )

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        self.dataset_val = SynapseRegressionDataset(csv_path=self.csv_path,
                                            fasta_path=self.validation_fasta_path,
                                            synapse_label_name=self.synapse_label_name,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            upsample=False,
                                            use_padding=self.use_padding)
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        self.dataset_test = SynapseRegressionDataset(csv_path=self.csv_path,
                                            fasta_path=self.test_fasta_path,
                                            synapse_label_name=self.synapse_label_name,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            upsample=False,
                                            use_padding=self.use_padding)
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)
    
    def predict_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        self.dataset_predict = SynapseRegressionDataset(csv_path=self.csv_path,
                                            fasta_path=self.predict_fasta_path,
                                            synapse_label_name=self.synapse_label_name,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            upsample=False,
                                            use_padding=self.use_padding)
        return self._data_loader(self.dataset_predict, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet




class TranslationEfficiencyDataloader(SequenceDataset):
    """
    You must implement the following functions:
        - __init__
        - setup

    You can then use (already have access to) the following functions:
        - train_dataloader
        - val_dataloader
        - test_dataloader

    """
    _name_ = "translation_efficiency"  # this name is how the dataset config finds the right dataloader
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, csv_path=None, train_fasta_path=None, validation_fasta_path=None, test_fasta_path=None, predict_fasta_path=None, synapse_label_name=False, tokenizer_name=None, dataset_config_name=None, max_length=1024, d_output=2, rc_aug=False,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357, use_fixed_len_val=False,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, upsample=False, use_padding=True,
                 aggregation_mode="mean",region=None,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.csv_path = csv_path
        self.train_fasta_path = train_fasta_path
        self.validation_fasta_path = validation_fasta_path
        self.predict_fasta_path = predict_fasta_path
        self.synapse_label_name = synapse_label_name
        self.test_fasta_path = test_fasta_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.use_fixed_len_val = use_fixed_len_val
        self.fault_tolerant = fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        self.upsample=upsample
        self.use_padding=use_padding
        self.aggregation_mode=aggregation_mode
        self.region=region

        

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("**using pretrained AIRI tokenizer**")
            self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()  # creates the datasets.  You can also just create this inside the setup() here.


    def init_datasets(self):
        """Init the datasets (separate from the tokenizer)"""

        # delete old datasets to free memory
        if hasattr(self, 'dataset_train'):
            del self.dataset_train

        if hasattr(self, 'dataset_val'):
            del self.dataset_val

        # Initialize empty lists to store the datasets for each split


        return

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        self.dataset_train = TranslationEfficiencyDataset(csv_path=self.csv_path,
                                                fasta_path=self.train_fasta_path,
                                                max_length=self.max_length,
                                                tokenizer=self.tokenizer,
                                                tokenizer_name=self.tokenizer_name,
                                                add_eos=self.add_eos,
                                                upsample=self.upsample,
                                                use_padding=self.use_padding,
                                                aggregation_mode=self.aggregation_mode,
                                                region=self.region)
        
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler,
                                 )

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        self.dataset_val = TranslationEfficiencyDataset(csv_path=self.csv_path,
                                            fasta_path=self.validation_fasta_path,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            upsample=False,
                                            use_padding=self.use_padding,
                                            aggregation_mode=self.aggregation_mode,
                                            region=self.region)
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        self.dataset_test = TranslationEfficiencyDataset(csv_path=self.csv_path,
                                            fasta_path=self.test_fasta_path,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            upsample=False,
                                            use_padding=self.use_padding,
                                            aggregation_mode=self.aggregation_mode,
                                            region=self.region)
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)
    
    def predict_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        self.dataset_predict = TranslationEfficiencyDataset(csv_path=self.csv_path,
                                            fasta_path=self.predict_fasta_path,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            upsample=False,
                                            use_padding=self.use_padding,
                                            aggregation_mode=self.aggregation_mode,
                                            region=self.region)
        return self._data_loader(self.dataset_predict, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet


class UTRClassificationDataloader(SequenceDataset):
    """
    You must implement the following functions:
        - __init__
        - setup

    You can then use (already have access to) the following functions:
        - train_dataloader
        - val_dataloader
        - test_dataloader

    """
    _name_ = "utr_classification"  # this name is how the dataset config finds the right dataloader
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, csv_path=None, train_fasta_path=None, validation_fasta_path=None, test_fasta_path=None, predict_fasta_path=None, synapse_label_name=False, tokenizer_name=None, dataset_config_name=None, max_length=1024, d_output=2, rc_aug=False,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357, use_fixed_len_val=False,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, upsample=False, use_padding=True,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.csv_path = csv_path
        self.train_fasta_path = train_fasta_path
        self.validation_fasta_path = validation_fasta_path
        self.predict_fasta_path = predict_fasta_path
        self.synapse_label_name = synapse_label_name
        self.test_fasta_path = test_fasta_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.use_fixed_len_val = use_fixed_len_val
        self.fault_tolerant = fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        self.upsample=upsample
        self.use_padding=use_padding

        

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("**using pretrained AIRI tokenizer**")
            self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()  # creates the datasets.  You can also just create this inside the setup() here.


    def init_datasets(self):
        """Init the datasets (separate from the tokenizer)"""

        # delete old datasets to free memory
        if hasattr(self, 'dataset_train'):
            del self.dataset_train

        if hasattr(self, 'dataset_val'):
            del self.dataset_val

        # Initialize empty lists to store the datasets for each split


        return

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        self.dataset_train = UTRClassificationDataset(csv_path=self.csv_path,
                                                fasta_path=self.train_fasta_path,
                                                synapse_label_name=self.synapse_label_name,
                                                max_length=self.max_length,
                                                tokenizer=self.tokenizer,
                                                tokenizer_name=self.tokenizer_name,
                                                add_eos=self.add_eos,
                                                upsample=self.upsample,
                                                use_padding=self.use_padding)
        
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler,
                                 )

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        self.dataset_val = UTRClassificationDataset(csv_path=self.csv_path,
                                            fasta_path=self.validation_fasta_path,
                                            synapse_label_name=self.synapse_label_name,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            upsample=False,
                                            use_padding=self.use_padding)
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        self.dataset_test = UTRClassificationDataset(csv_path=self.csv_path,
                                            fasta_path=self.test_fasta_path,
                                            synapse_label_name=self.synapse_label_name,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            upsample=False,
                                            use_padding=self.use_padding)
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)
    
    def predict_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        self.dataset_predict = UTRClassificationDataset(csv_path=self.csv_path,
                                            fasta_path=self.predict_fasta_path,
                                            synapse_label_name=self.synapse_label_name,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            upsample=False,
                                            use_padding=self.use_padding)
        return self._data_loader(self.dataset_predict, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet




class MutationScoringDataloader(SequenceDataset):
    """
    You must implement the following functions:
        - __init__
        - setup


    """
    _name_ = "mutation_scoring"  # this name is how the dataset config finds the right dataloader
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, csv_path=None, fasta_path=None, predict_fasta_path=None, synapse_label_name=False, tokenizer_name=None, dataset_config_name=None, max_length=1024, d_output=2, rc_aug=False,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357, use_fixed_len_val=False,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, upsample=False, use_padding=True,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.csv_path = csv_path
        self.fasta_path = fasta_path
        self.predict_fasta_path = predict_fasta_path
        self.synapse_label_name = synapse_label_name
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.use_fixed_len_val = use_fixed_len_val
        self.fault_tolerant = fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        self.upsample=upsample
        self.use_padding=use_padding

        

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("**using pretrained AIRI tokenizer**")
            self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')

        self.vocab_size = len(self.tokenizer)



    def predict_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        self.dataset_predict = MutationScoring(csv_path=self.csv_path,
                                               fasta_path=self.fasta_path,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            use_padding=self.use_padding)
        return self._data_loader(self.dataset_predict, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )




class CodonScoringDataloader(SequenceDataset):
    """
    You must implement the following functions:
        - __init__
        - setup


    """
    _name_ = "codon_scoring"  # this name is how the dataset config finds the right dataloader
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, csv_path=None, train_fasta_path=None, validation_fasta_path=None, test_fasta_path=None, predict_fasta_path=None, synapse_label_name=False, tokenizer_name=None, dataset_config_name=None, max_length=1024, d_output=2, rc_aug=False,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357, use_fixed_len_val=False,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, upsample=False, use_padding=True,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.csv_path = csv_path
        self.train_fasta_path = train_fasta_path
        self.validation_fasta_path = validation_fasta_path
        self.predict_fasta_path = predict_fasta_path
        self.synapse_label_name = synapse_label_name
        self.test_fasta_path = test_fasta_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.use_fixed_len_val = use_fixed_len_val
        self.fault_tolerant = fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        self.upsample=upsample
        self.use_padding=use_padding

        

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("**using pretrained AIRI tokenizer**")
            self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')

        self.vocab_size = len(self.tokenizer)



    def predict_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        self.dataset_predict = CodonScoring(fasta_path=self.predict_fasta_path,
                                            max_length=self.max_length_val,
                                            tokenizer=self.tokenizer,
                                            tokenizer_name=self.tokenizer_name,
                                            add_eos=self.add_eos,
                                            use_padding=self.use_padding)
        return self._data_loader(self.dataset_predict, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
