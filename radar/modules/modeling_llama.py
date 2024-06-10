# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2022 EleutherAI and the HuggingFace Inc. team.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on Huggingface's Llama from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py by HuggingFace  # noqa: E501
# Huggingface's transformers repository is under the Apache 2.0 License at https://github.com/huggingface/transformers/blob/main/LICENSE  # noqa: E501
####################################################################################


import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from radar.radar_cache import RadarCache

logger = logging.getLogger(__name__)


class SequentialLlamaRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            device=device,
            scaling_factor=scaling_factor,
            rope_type=rope_type,
            config=config,
        )

        inv_freq_expanded = self.inv_freq[:, None].float()  # (dim / 2, 1)
        position_ids = torch.arange(
            0, config.max_position_embeddings, dtype=torch.long
        ).to(device)
        position_ids_expanded = position_ids[None, :].float()  # (1, max_len)
        precomputed_freqs = (
            inv_freq_expanded.float() @ position_ids_expanded.float()
        ).transpose(
            0, 1
        )  # (max_len, dim / 2)

        with torch.autocast(enabled=False, device_type=device.type):
            self.register_buffer(
                "_precomputed_cos", precomputed_freqs.cos()
            )  # (max_len, dim / 2)
            self.register_buffer(
                "_precomputed_sin", precomputed_freqs.sin()
            )  # (max_len, dim / 2)

    @classmethod
    def from_original(cls, original: LlamaRotaryEmbedding):
        new = cls(
            dim=original.inv_freq.size(-1) * 2,
            max_position_embeddings=original.original_max_seq_len,
            base=original.rope_kwargs.get("base", original.config.rope_theta),
            device=original.inv_freq.device,
            scaling_factor=original.rope_kwargs.get("scaling_factor", 1.0),
            rope_type=original.rope_type,
            config=original.config,
        )
        return new

    @torch.no_grad()
    def forward(self, x, position_ids=None, batch=None, length=None):
        if position_ids is not None:
            # Not our call, skip for efficiency reasons
            return None, None
        batch = batch
        length = length
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(length, device=x.device)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type):
            cos = self._precomputed_cos[:length]
            cos = torch.cat([cos, cos], dim=-1).unsqueeze(
                0
            )  # (1, max_len, dim / 2) -> (1, max_len, dim)
            sin = self._precomputed_sin[:length]
            sin = torch.cat([sin, sin], dim=-1).unsqueeze(
                0
            )  # (1, max_len, dim / 2) -> (1, max_len, dim)

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def llama_forward(
    self,
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

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
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

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
