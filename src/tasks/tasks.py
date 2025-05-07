from typing import Optional, List, Tuple
import math
import functools
from functools import reduce
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import ListConfig
from src.models.nn.components import ReversibleInstanceNorm1dInput, ReversibleInstanceNorm1dOutput, \
    TSNormalization, TSInverseNormalization

from src.models.nn.adaptive_softmax import AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
import src.tasks.metrics as M
from src.tasks.torchmetrics import torchmetric_fns as tm_mine
import src.models.nn.utils as U
import torchmetrics as tm
from src.utils.config import to_list, instantiate
from torchmetrics import MetricCollection

class BaseTask:
    """ Abstract class that takes care of:
    - loss function
    - arbitrary metrics
    - forward pass
    - (optional) encoder module that interfaces with dataset (inputs) and model
    - (optional) decoder module that interfaces with dataset (targets) and model
    """
    encoder = None
    decoder = None

    def __init__(self, dataset=None, model=None, loss=None, weights=None, loss_val=None, metrics=None, torchmetrics=None, **kwargs):
        """ This class is allowed to grab attributes directly off a constructed dataset and model object """
        self.dataset = dataset
        self.model = model
        if metrics is None: metrics = []
        self.metric_names = to_list(metrics)

        if torchmetrics is None: torchmetrics = []
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

        # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        new_loss = instantiate(M.output_metric_fns, loss, partial=True)
        lambda_loss = lambda logits, y, ignore_index=-100: new_loss(logits, y, weights=weights, ignore_index=ignore_index)
        self.loss = U.discard_kwargs(lambda_loss)
        if loss_val is not None:
            self.loss_val = instantiate(M.output_metric_fns, loss_val, partial=True)
            self.loss_val = U.discard_kwargs(self.loss_val)
        torchmetrics = MetricCollection(self._init_torchmetrics())
        self.train_torchmetrics = torchmetrics.clone(prefix='train/')
        self.val_torchmetrics = torchmetrics.clone(prefix='val/')
        self.test_torchmetrics = torchmetrics.clone(prefix='test/')

    def _init_torchmetrics(self):
        """
        Instantiate torchmetrics.
        """
        tracked_torchmetrics = {}

        for name in self.torchmetric_names:
            if name in tm_mine:
                tracked_torchmetrics[name] = tm_mine[name]()
            elif name in ['AUROC', 'StatScores', 'Precision', 'Recall', 'F1', 'F1Score', "Specificity"]:
                tracked_torchmetrics[name] = getattr(tm, name)(task='binary', average='macro', num_classes=self.dataset.d_output)
            elif '@' in name:
                k = int(name.split('@')[1])
                mname = name.split('@')[0]
                tracked_torchmetrics[name] = getattr(tm, mname)(average='macro', num_classes=self.dataset.d_output, top_k=k)
            else:
                tracked_torchmetrics[name] = getattr(tm, name)()

        return tracked_torchmetrics

    def _reset_torchmetrics(self, prefix=None):
        """
        Reset torchmetrics for a prefix
        associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the start of an epoch.
        """
        all_prefixes = [prefix] if prefix is not None else self._tracked_torchmetrics

        for prefix in all_prefixes:
            if prefix in self._tracked_torchmetrics:
                self._tracked_torchmetrics[prefix].reset()

    def get_torchmetrics(self, prefix):
        """
        Compute torchmetrics for a prefix associated with
        a particular dataloader (e.g. train, val, test).

        Generally do this at the end of an epoch.
        """
        return {name: self._tracked_torchmetrics[prefix][name].compute() for name in self.torchmetric_names}

    def torchmetrics(self, x, y, prefix, loss=None):
        """
        Update torchmetrics with new x, y .
        Prefix corresponds to a particular dataloader (e.g. train, val, test).

        Generally call this every batch.
        """
        if prefix not in self._tracked_torchmetrics:
            self._init_torchmetrics(prefix)
        self._tracked_torchmetrics[prefix](x, y, loss=loss)

        # for name in self.torchmetric_names:
        #     if name.startswith('Accuracy'):
        #         if len(x.shape) > 2:
        #             # Multi-dimensional, multi-class
        #             self._tracked_torchmetrics[prefix][name].update(x.transpose(1, 2), y.squeeze())
        #             continue
        #     self._tracked_torchmetrics[prefix][name].update(x, y)

    def get_torchmetrics(self, prefix):
        return self._tracked_torchmetrics[prefix]

    def metrics(self, x, y, **kwargs):
        """
        Metrics are just functions
        output metrics are a function of output and target
        loss metrics are a function of loss (e.g. perplexity)
        """
        output_metrics = {
            name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }
        return {**output_metrics, **loss_metrics}

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        x, w = encoder(x) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x, state = model(x, **w, state=_state)
        self._state = state
        x, w = decoder(x, state=state)
        return x, y, w


class Scalar(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c = c
    def forward(self, x):
        return x * self.c

class LMTask(BaseTask):
    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]
        x, w = encoder(x, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x, state = model(x, **w, state=_state)
        self._state = state
        x, w = decoder(x, state=state, **z)

        x = x.logits
        x = rearrange(x, '... C -> (...) C')
        y = rearrange(y, '... -> (...)')

        return x, y, w



class MLMTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # mlm related probabilities

        self.mask_prob = kwargs['mask_prob']
        self.replace_prob = kwargs['replace_prob']

        self.num_tokens = kwargs['num_tokens']
        self.random_token_prob = kwargs['random_token_prob']

        # token ids

        self.pad_token_id = kwargs['pad_token_id']
        self.mask_token_id = kwargs['mask_token_id']
        self.mask_ignore_token_ids = set([*kwargs['mask_ignore_token_ids'], kwargs['pad_token_id']])


    def prob_mask_like(self, t, prob):
        return torch.zeros_like(t).float().uniform_(0, 1) < prob

    def mask_with_tokens(self, t, token_ids):
        init_no_mask = torch.full_like(t, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
        return mask

    def get_mask_subset_with_prob(self, mask, prob):
        batch, seq_len, device = *mask.shape, mask.device
        max_masked = math.ceil(prob * seq_len)

        num_tokens = mask.sum(dim=-1, keepdim=True)
        mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
        mask_excess = mask_excess[:, :max_masked]

        rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
        _, sampled_indices = rand.topk(max_masked, dim=-1)
        sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

        new_mask = torch.zeros((batch, seq_len + 1), device=device)
        new_mask.scatter_(-1, sampled_indices, 1)
        return new_mask[:, 1:].bool()

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random

        no_mask = self.mask_with_tokens(x, self.mask_ignore_token_ids)
        mask = self.get_mask_subset_with_prob(~no_mask, self.mask_prob)

        masked_seq = x.clone().detach()

        # derive labels to predict

        labels = x.masked_fill(~mask, self.pad_token_id)

        # if random token probability > 0 for mlm

        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
            random_token_prob = self.prob_mask_like(x, self.random_token_prob)
            random_tokens = torch.randint(7, self.num_tokens+7, x.shape, device=x.device) #add 7, because all tokens below 7 are reserved for special tokens
            random_no_mask = self.mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)

            # remove tokens that were substituted randomly from being [mask]ed later
            mask = mask & ~random_token_prob

        # [mask] input
            
        # see here for interpretation of params: https://github.com/lucidrains/mlm-pytorch/issues/6#issuecomment-1030529603

        replace_prob = self.prob_mask_like(x, self.replace_prob)
        masked_seq = masked_seq.masked_fill(mask * replace_prob, self.mask_token_id)
        
        x, w = encoder(masked_seq, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x, state = model(x)
        self._state = state
        x, w = decoder(x, state=state, **z)

        x = rearrange(x, '... C -> (...) C')
        labels = rearrange(labels, '... -> (...)')

        return x, labels, w

class MultiClass(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.continual_metrics = {}
        for name in self.metric_names:
            if name.endswith('_per_class'):
                for spec_idx, spec in enumerate(self.dataset.species):
                    self.continual_metrics[name + '_' + spec] = M.output_metric_fns[name](spec_idx)

    def metrics(self, x, y, **kwargs):
        output_metrics = {}
        for name in self.metric_names:
            if name in M.output_metric_fns:
                if name.endswith('_per_class'):
                    for spec_idx, spec in enumerate(self.dataset.species):
                        self.continual_metrics[name + '_' + spec] = self.continual_metrics[name + '_' + spec].to(x.device)
                        self.continual_metrics[name + '_' + spec].update(x, y)
                        output_metrics[name + '_' + spec] = self.continual_metrics[name + '_' + spec].compute()
                elif name in ['precision']:
                    self.continual_metrics[name] = self.continual_metrics[name].to(x.device)
                    output_metrics[name] = self.continual_metrics[name](x, y)
                else:
                    output_metrics[name] = U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)

        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }

        return {**output_metrics, **loss_metrics}

    def _reset_torchmetrics(self, prefix=None):
        super()._reset_torchmetrics(prefix)
        for name in self.metric_names:
            if name.endswith('_per_class'):
                for spec_idx, spec in enumerate(self.dataset.species):
                    self.continual_metrics[name + '_' + spec].reset()


class Regression(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        new_loss = instantiate(M.output_metric_fns, kwargs['loss'], partial=True)
        lambda_loss = lambda outs, y: new_loss(outs, y)
        self.loss = U.discard_kwargs(lambda_loss)


class MultiOutputRegression(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        new_loss = instantiate(M.output_metric_fns, kwargs['loss'], partial=True)

        lambda_loss = lambda outs, y, mask=None: new_loss(outs, y, mask=mask)
        self.loss = U.discard_kwargs(lambda_loss)

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        x, w = encoder(x) 
        x, state = model(x, **w, state=_state)
        self._state = state
        x, w = decoder(x, state=state)
        return x, y, z



registry = {
    'base': BaseTask,
    'multiclass': MultiClass,
    'binary': MultiClass,
    'lm': LMTask,
    'mlm': MLMTask,
    'regression': Regression,
    'multioutput_regression': MultiOutputRegression
}
