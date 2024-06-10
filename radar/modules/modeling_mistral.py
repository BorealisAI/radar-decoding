# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2023 Mistral AI and the HuggingFace Inc. team.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on Huggingface's Mistral from https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py by HuggingFace  # noqa: E501
# Huggingface's transformers repository is under the Apache 2.0 License at https://github.com/huggingface/transformers/blob/main/LICENSE  # noqa: E501
####################################################################################


import logging
from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    apply_rotary_pos_emb,
)

from radar.radar_cache import RadarCache

logger = logging.getLogger(__name__)


def mistral_forward(
    self: MistralAttention,
    cache_config,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    if past_key_value is None:
        past_key_value = RadarCache(cache_config)
    elif not isinstance(past_key_value, RadarCache):
        raise ValueError(
            f"`past_key_value` should be of type `RadarCache`, but is of type {type(past_key_value)}."
        )

    if not cache_config.extend_context:
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    cache_kwargs = {
        "position_ids": position_ids,
        "query_states": query_states,
        "rotary_emb": self.rotary_emb,
        "attention_mask": attention_mask,
        "attention_dropout": self.attention_dropout,
        "training": self.training,
    }
    attn_output = past_key_value.update(
        key_states, value_states, self.layer_idx, cache_kwargs
    )

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
