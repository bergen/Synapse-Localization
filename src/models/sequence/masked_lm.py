import re
from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from transformers import BertConfig

from flash_attn.models.bert import (BertPreTrainedModel,
    BertModel,
    _init_weights
)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        linear_cls = nn.Linear
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        approximate = (
            "tanh"
            if config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"]
            else "none"
        )
        self.transform_act_fn = nn.GELU(approximate=approximate)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        linear_cls = nn.Linear

        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = linear_cls(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states





class BertForMLM(BertPreTrainedModel):
    def __init__(self, **args):

        config = BertConfig(**args)
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.predictions = BertLMPredictionHead(config)

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, initializer_range=config.initializer_range))

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
    ):
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=None,
            masked_tokens_mask=None,
        )
        sequence_output = outputs.last_hidden_state

        prediction_scores = self.predictions(sequence_output)


        return prediction_scores, None