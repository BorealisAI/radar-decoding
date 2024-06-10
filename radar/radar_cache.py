# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from transformers import CacheConfig, DynamicCache

torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)

PROJ_REUSE = {}


def set_logging_level(level: Union[str, int]) -> None:
    logger.setLevel(level)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@dataclass
class RadarCacheConfig(CacheConfig):
    cache_implementation: str = "radar"

    def __init__(
        self,
        hdim: int,
        projection_dim: Optional[int] = None,
        residual_length: Optional[int] = 1024,
        compute_dtype: Optional[torch.dtype] = torch.bfloat16,
        device: Optional[str] = "cpu",
        num_heads: Optional[int] = 8,
        num_kv_heads: Optional[int] = 4,
        num_sink_tokens: Optional[int] = 1,
        num_layers: Optional[int] = 1,
        enabled: Optional[bool] = True,
        ablation: Optional[str] = "none",
        topk: Optional[int] = None,
        target_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        aspect_ratio: Optional[int] = 2,
        extend_context: Optional[bool] = False,
    ):
        if enabled:
            assert topk is not None or target_tokens is not None
            assert topk is None or target_tokens is None
            assert max_length is not None

        self.hdim = hdim
        self.residual_length = residual_length
        self.compute_dtype = compute_dtype
        self.device = device
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = hdim // num_heads
        self.num_sink_tokens = num_sink_tokens
        self.num_layers = num_layers
        self.enabled = enabled
        self.ablation = ablation
        self.topk = topk
        self.target_tokens = target_tokens
        self.max_length = max_length
        self.aspect_ratio = aspect_ratio
        self.extend_context = extend_context

        self.projected_dim = (
            projection_dim if projection_dim is not None else self.head_dim * 16
        )

    def get_cluster_size_floor(self, ntokens):
        return math.floor(math.sqrt(ntokens) / self.aspect_ratio)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    assert k.shape[-2] == cos.shape[-2]
    length_q = q.shape[-2]
    q_embed = (q * cos[..., -length_q:, :]) + (rotate_half(q) * sin[..., -length_q:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Projector:
    def __init__(
        self,
        batch,
        hdim,
        projected_dim,
        layer_idx=0,
        eps=1e-18,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        self.hdim = hdim
        self.projected_dim = projected_dim
        self.device = device
        self.batch = batch

        self.eps = eps
        self.d = hdim**0.5
        self.dtype = dtype
        self.layer_idx = layer_idx

        self.normalizers = None

        global PROJ_REUSE
        if PROJ_REUSE.get(hdim, None) is not None:
            self.projection_matrix = PROJ_REUSE[hdim]
        else:
            PROJ_REUSE[hdim] = self.projection_matrix = torch.randn(
                (hdim, projected_dim), device=device
            ).unsqueeze(0)

    def centroids(self, x, normalize=False, mem_efficient=False):
        assert x.ndim == 4
        pnormalizer = torch.zeros((1, 1, 1), device=self.device)
        nnormalizer = torch.zeros((1, 1, 1), device=self.device)
        if mem_efficient:
            proj_results = []
            for i in range(x.shape[1]):
                px, normx = self.get_projection(x[:, i])
                _pnormalizer = torch.amax(px - normx, dim=(1, 2), keepdim=True)
                _nnormalizer = torch.amax(-px - normx, dim=(1, 2), keepdim=True)
                pnormalizer = torch.max(pnormalizer, _pnormalizer)
                nnormalizer = torch.max(nnormalizer, _nnormalizer)
                proj_results.append((px, normx))

            results = []
            for i in range(x.shape[1]):
                px, normx = proj_results[i]
                feature = self.get_transformation(
                    px, normx, (pnormalizer, nnormalizer)
                ).mean(dim=-2)
                results.append(feature)
            return torch.stack(results, dim=1), (pnormalizer, nnormalizer)
        else:
            batch, num_blocks, block_size, hdim = x.shape
            x = x.reshape(batch, num_blocks * block_size, hdim)
            return (
                self(x, normalize=normalize)
                .reshape(batch, num_blocks, block_size, self.projected_dim)
                .mean(dim=-2),
                None,
            ), None

    def get_projection(self, x):
        x = x.float()
        dim = x.shape[-1]
        x = x * (dim**-0.25)
        normx = torch.sum(x**2, dim=-1, keepdim=True) / 2

        x = torch.bmm(x, self.projection_matrix.expand(x.shape[0], -1, -1))

        return x, normx

    def get_transformation(self, px, normx, normalizers):
        pnormalizer, nnormalizer = normalizers
        feature = torch.cat(
            [
                torch.exp(px - normx - pnormalizer) + self.eps,
                torch.exp(-px - normx - nnormalizer) + self.eps,
            ],
            dim=-1,
        ) * ((2 * self.projected_dim) ** -0.5)
        return feature

    @torch.no_grad()
    def __call__(self, x, normalize=False):

        x, normx = self.get_projection(x)

        if normalize is None:
            pnormalizer = nnormalizer = 0.0
        elif isinstance(normalize, bool):
            pnormalizer = torch.amax(x - normx, dim=(-1, -2), keepdim=True)
            nnormalizer = torch.amax(-x - normx, dim=(-1, -2), keepdim=True)
        elif isinstance(normalize, tuple):
            pnormalizer, nnormalizer = normalize
        else:
            raise ValueError("Invalid normalize argument")

        return self.get_transformation(x, normx, (pnormalizer, nnormalizer))


def get_random_features(projection_matrix, x):
    x = x.float()
    dim = x.shape[-1]
    proj_dim = projection_matrix.shape[-1]

    x = x * (dim**-0.25)
    normx = torch.sum(x**2, dim=-1, keepdim=True) / 2
    x = torch.bmm(x, projection_matrix.expand(x.shape[0], -1, -1))

    pnormalizer = torch.amax(x - normx, dim=(1, 2), keepdim=True)
    nnormalizer = torch.amax(-x - normx, dim=(1, 2), keepdim=True)

    feature = torch.cat(
        [
            torch.exp(x - normx - pnormalizer) + 1e-18,
            torch.exp(-x - normx - nnormalizer) + 1e-18,
        ],
        dim=-1,
    ) * ((2 * proj_dim) ** -0.5)

    return feature


def search_fn(
    extend_context, topk, projection, qs, stored_keys, stored_values, reps, keys, values
):
    num_heads, num_query, _ = qs.shape
    block_size = stored_keys.size(-2)
    head_dim = stored_keys.size(-1)

    medot = torch.matmul(get_random_features(projection, qs), reps.transpose(1, 2))
    block_idx = torch.topk(medot, k=topk, dim=-1, sorted=False).indices
    if extend_context:
        block_idx = torch.sort(block_idx, -1).values

    block_idx = block_idx[..., None, None].expand(-1, -1, -1, block_size, head_dim)

    stored_keys = stored_keys.unsqueeze(1).expand(-1, num_query, -1, -1, -1)
    stored_keys = torch.gather(stored_keys, 2, block_idx)
    stored_keys = stored_keys.reshape(num_heads, num_query, -1, head_dim)
    stored_values = stored_values.unsqueeze(1).expand(-1, num_query, -1, -1, -1)
    stored_values = torch.gather(stored_values, 2, block_idx)
    stored_values = stored_values.reshape(num_heads, num_query, -1, head_dim)

    keys = torch.cat([stored_keys, keys], dim=-2)
    values = torch.cat([stored_values, values], dim=-2)

    return keys, values


def centroids_fn(projection, x):
    x = x.float()
    num_heads, num_blocks, block_size, hdim = x.shape
    proj_dim = projection.shape[-1]

    x = x.reshape(num_heads, num_blocks * block_size, hdim)
    x = x * (hdim**-0.25)

    normx = torch.sum(x**2, dim=-1, keepdim=True) / 2

    pnormalizer = torch.amax(x - normx, dim=(1, 2), keepdim=True)
    nnormalizer = torch.amax(-x - normx, dim=(1, 2), keepdim=True)

    x = torch.bmm(x, projection.expand(x.shape[0], -1, -1))

    x = torch.cat(
        [
            torch.exp(x - normx - pnormalizer) + 1e-18,
            torch.exp(-x - normx - nnormalizer) + 1e-18,
        ],
        dim=-1,
    ) * ((2 * proj_dim) ** -0.5)
    x = x.reshape(num_heads, num_blocks, block_size, -1).mean(dim=-2)
    return x, (pnormalizer, nnormalizer)


class Manager:
    def __init__(self, config: RadarCacheConfig, layer_idx: int = 0):
        self.num_envs = config.num_kv_heads
        self.ntotal = 0
        self.config = config
        self.layer_idx = layer_idx

        self._stored_keys = torch.empty(
            (self.num_envs, 0, 0, config.head_dim),
            device=config.device,
            dtype=config.compute_dtype,
        )
        self._stored_values = torch.empty(
            (self.num_envs, 0, 0, config.head_dim),
            device=config.device,
            dtype=config.compute_dtype,
        )
        self._remainder_keys = torch.empty(
            (self.num_envs, 0, config.head_dim),
            device=config.device,
            dtype=config.compute_dtype,
        )
        self._remainder_values = torch.empty(
            (self.num_envs, 0, config.head_dim),
            device=config.device,
            dtype=config.compute_dtype,
        )
        self._reps = None
        self.nclusters = 0
        self.size_per_cluster = -1
        self.aspect_ratio = self.config.aspect_ratio

        self._normalizers = None
        self._debt = 0

        self.proj = Projector(
            batch=config.num_kv_heads,
            hdim=config.head_dim,
            projected_dim=config.projected_dim,
            device=config.device,
            layer_idx=layer_idx,
            dtype=config.compute_dtype,
        )

    def reorganize(self):
        if self._debt == 0:
            return

        if (
            self.config.topk
            and (self.config.topk / self.aspect_ratio) ** 2 > self.ntotal
        ):
            return
        if self.config.target_tokens and self.config.target_tokens > self.ntotal:
            return

        to_rebuild = False
        if self.size_per_cluster < 0:
            to_rebuild = True
        else:
            nclusters_after_insert = self.ntotal // self.size_per_cluster
            if (
                nclusters_after_insert
                >= self.size_per_cluster * (self.aspect_ratio**2) * 2
            ):
                to_rebuild = True

        self._debt = 0

        if not to_rebuild:
            remainder_length = self._remainder_keys.shape[1]
            if remainder_length < self.size_per_cluster:
                return
            new_clusters = remainder_length // self.size_per_cluster
            new_tokens = new_clusters * self.size_per_cluster
            num_envs = self._remainder_keys.shape[0]

            self.nclusters += new_clusters
            self._stored_keys = torch.cat(
                [
                    self._stored_keys,
                    self._remainder_keys[:, :new_tokens].view(
                        num_envs, new_clusters, self.size_per_cluster, -1
                    ),
                ],
                dim=1,
            )
            self._stored_values = torch.cat(
                [
                    self._stored_values,
                    self._remainder_values[:, :new_tokens].view(
                        num_envs, new_clusters, self.size_per_cluster, -1
                    ),
                ],
                dim=1,
            )
            for i in range(1, new_clusters + 1):
                self._reps = torch.cat(
                    [
                        self._reps,
                        self.proj(self._stored_keys[:, -i], self._normalizers).mean(
                            dim=-2, keepdim=True
                        ),
                    ],
                    dim=1,
                )

            self._remainder_keys = self._remainder_keys[:, new_tokens:]
            self._remainder_values = self._remainder_values[:, new_tokens:]
        else:
            assert (
                self.ntotal
                == self._stored_keys.shape[1] * self.size_per_cluster
                + self._remainder_keys.shape[1]
            )

            self.size_per_cluster = self.config.get_cluster_size_floor(self.ntotal)
            self.nclusters = self.ntotal // self.size_per_cluster
            if self.layer_idx == 0:
                logger.debug(
                    f"Rebuilding cache with {self.size_per_cluster} tokens per cluster for {self.ntotal} tokens"
                )
            stored_length = self.nclusters * self.size_per_cluster
            remainder_length = self.ntotal - stored_length

            all_keys = self._stored_keys.reshape(
                self.num_envs, -1, self.config.head_dim
            )
            self._stored_keys = None
            all_keys = torch.cat([all_keys, self._remainder_keys], dim=1)
            all_keys, self._remainder_keys = (
                all_keys[:, :stored_length],
                all_keys[:, stored_length:],
            )

            all_values = self._stored_values.reshape(
                self.num_envs, -1, self.config.head_dim
            )
            self._stored_values = None
            all_values = torch.cat([all_values, self._remainder_values], dim=1)
            all_values, self._remainder_values = (
                all_values[:, :stored_length],
                all_values[:, stored_length:],
            )

            assert all_keys.shape == all_values.shape
            assert all_keys.shape[1] == stored_length
            assert self._remainder_keys.shape[1] == remainder_length

            self._stored_keys = all_keys.reshape(
                self.num_envs, self.nclusters, self.size_per_cluster, -1
            )
            self._stored_values = all_values.reshape(
                self.num_envs, self.nclusters, self.size_per_cluster, -1
            )

            self._reps, self._normalizers = self.proj.centroids(
                self._stored_keys, mem_efficient=True, normalize=True
            )

    def add(self, keys, values):
        num_envs, num_data, _ = keys.shape
        assert num_envs == self.config.num_kv_heads
        assert values.shape == keys.shape
        keys = keys.clone()
        values = values.clone()
        self._remainder_keys = torch.cat([self._remainder_keys, keys], dim=1)
        self._remainder_values = torch.cat([self._remainder_values, values], dim=1)

        self.ntotal += num_data
        self._debt += num_data

    def search(self, qs):
        self.reorganize()
        topk = self.config.topk or int(
            self.config.target_tokens / self.size_per_cluster
        )
        num_envs, num_query, _ = qs.shape
        if self._remainder_keys.shape[1] > 0:
            keys = self._remainder_keys[:, None, ...].expand(-1, num_query, -1, -1)
            values = self._remainder_values[:, None, ...].expand(-1, num_query, -1, -1)
        else:
            keys = torch.empty(
                (num_envs, num_query, 0, self.config.head_dim),
                device=self.config.device,
                dtype=self.config.compute_dtype,
            )
            values = torch.empty(
                (num_envs, num_query, 0, self.config.head_dim),
                device=self.config.device,
                dtype=self.config.compute_dtype,
            )

        if self._stored_keys.shape[1] > 0:
            if topk > self.nclusters:
                keys = torch.cat(
                    [
                        self._stored_keys.reshape(
                            num_envs, 1, -1, self.config.head_dim
                        ).expand(-1, num_query, -1, -1),
                        keys,
                    ],
                    dim=-2,
                )
                values = torch.cat(
                    [
                        self._stored_values.reshape(
                            num_envs, 1, -1, self.config.head_dim
                        ).expand(-1, num_query, -1, -1),
                        values,
                    ],
                    dim=-2,
                )
                return keys, values
            if self.config.ablation == "none":
                return search_fn(
                    self.config.extend_context,
                    topk,
                    self.proj.projection_matrix,
                    qs,
                    self._stored_keys,
                    self._stored_values,
                    self._reps,
                    keys,
                    values,
                )
            elif self.config.ablation == "bottom":
                medot = torch.matmul(self.proj(qs, True), self._reps.transpose(1, 2))
                _, block_idx = torch.topk(
                    medot, dim=-1, k=topk, largest=False, sorted=False
                )
                if self.config.extend_context:
                    block_idx = block_idx.sort(dim=-1).values
            elif self.config.ablation == "random":
                block_idx = torch.arange(
                    self.nclusters, device=self.config.device
                ).repeat(num_envs, num_query, 1)
                block_idx = torch.gather(
                    block_idx,
                    2,
                    torch.randint(
                        0,
                        self.nclusters,
                        (num_envs, num_query, topk),
                        device=self.config.device,
                    ),
                )
                if self.config.extend_context:
                    block_idx = block_idx.sort(dim=-1).values
            elif self.config.ablation == "last":
                block_idx = (
                    torch.arange(
                        self.nclusters - topk, self.nclusters, device=self.config.device
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(num_envs, num_query, -1)
                )
            elif self.config.ablation == "exact":
                stored_keys = self._stored_keys
                dot = (
                    torch.einsum("hnbd,hqd->hqnb", stored_keys, qs)
                    / self.config.head_dim**0.5
                )
                dot = dot - dot.max()
                dot = torch.exp(dot).sum(dim=-1)
                block_idx = torch.topk(dot, dim=-1, k=topk, sorted=False).indices
                if self.config.extend_context:
                    block_idx = block_idx.sort(dim=-1).values
            else:
                raise ValueError(f"Invalid ablation option: {self.config.ablation}")
            block_idx = block_idx[..., None, None].expand(
                -1, -1, -1, self.size_per_cluster, self.config.head_dim
            )
            stored_keys = self._stored_keys[:, None, ...].expand(
                -1, num_query, -1, -1, -1
            )
            stored_keys = torch.gather(stored_keys, 2, block_idx)
            stored_keys = stored_keys.reshape(
                num_envs, num_query, -1, self.config.head_dim
            )
            stored_values = self._stored_values[:, None, ...].expand(
                -1, num_query, -1, -1, -1
            )
            stored_values = torch.gather(stored_values, 2, block_idx)
            stored_values = stored_values.reshape(
                num_envs, num_query, -1, self.config.head_dim
            )

            keys = torch.cat([stored_keys, keys], dim=-2)
            values = torch.cat([stored_values, values], dim=-2)

        return keys, values


class RadarCache(DynamicCache):
    def __init__(self, cache_config: RadarCacheConfig) -> None:
        self.device = cache_config.device
        self.hdim = cache_config.hdim
        self.num_heads = cache_config.num_heads
        self.head_dim = self.hdim // self.num_heads
        self.num_kv_heads = cache_config.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.residual_length = cache_config.residual_length
        self.num_sink_tokens = cache_config.num_sink_tokens
        self.projected_dim = cache_config.projected_dim
        self.enabled = cache_config.enabled

        self.config = cache_config

        super().__init__()

        self.mgrs = []

        self.sink_keys = []
        self.sink_values = []

    @classmethod
    def convert_from_cache(
        cls, cache: DynamicCache, cache_config: RadarCacheConfig
    ) -> "RadarCache":
        new_cache = cls(cache_config)
        new_cache._seen_tokens = cache._seen_tokens
        new_cache.key_cache = cache.key_cache
        new_cache.value_cache = cache.value_cache
        new_cache.sink_keys = [None] * len(cache.key_cache)
        new_cache.sink_values = [None] * len(cache.key_cache)
        for layer_idx in range(len(cache.key_cache)):
            new_cache._init_mgr_for_layer(layer_idx)
        return new_cache

    def _init_mgr_for_layer(self, layer_idx: int) -> None:
        self.mgrs.append(Manager(self.config, layer_idx))

    def get_max_length(self) -> int:
        return self.config.max_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        query_states = cache_kwargs["query_states"]

        if query_states.shape[0] > 1:
            raise NotImplementedError("Cache only supports batch size 1")

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.sink_keys.append(None)
            self.sink_values.append(None)
            self._init_mgr_for_layer(layer_idx)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        key_states, value_states = (
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
        )
        value_states = value_states[:, :, -key_states.shape[-2] :]

        if self.sink_keys[layer_idx] is not None:
            key_states = torch.cat(
                [self.sink_keys[layer_idx].unsqueeze(0), key_states], dim=-2
            )
            value_states = torch.cat(
                [self.sink_values[layer_idx].unsqueeze(0), value_states], dim=-2
            )

        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)

        batch, num_heads, slen, head_dim = key_states.shape
        max_len = self.get_max_length() - 1
        qlen = query_states.shape[-2]

        if qlen > 1:

            if qlen > max_len and self.config.extend_context:

                if layer_idx == 0:
                    logger.debug(
                        f"Query length {qlen} is greater than max length {max_len}"
                    )
                first_query_states = query_states[..., :max_len, :]
                first_key_states = key_states[..., :max_len, :]
                cos, sin = cache_kwargs["rotary_emb"](
                    value_states,
                    position_ids=None,
                    batch=key_states.shape[0],
                    length=max_len,
                )
                first_query_states, first_key_states = apply_rotary_pos_emb(
                    first_query_states, first_key_states, cos, sin
                )

                first_results = torch.nn.functional.scaled_dot_product_attention(
                    first_query_states,
                    first_key_states,
                    value_states[..., :max_len, :],
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                )

                second_results = []
                num_sink_tokens = self.num_sink_tokens
                for i in range(qlen - max_len):
                    cq = query_states[..., max_len + i : max_len + i + 1, :]
                    ck = key_states[..., num_sink_tokens : max_len + i + 1, :]
                    cv = value_states[..., num_sink_tokens : max_len + i + 1, :]
                    attention = cq @ ck.transpose(-2, -1)

                    attention = torch.topk(
                        attention, max_len - num_sink_tokens, dim=-1, sorted=False
                    ).indices.squeeze(-2)
                    if self.config.extend_context:
                        attention = torch.sort(attention, -1).values

                    cv = cv.gather(
                        -2, attention[..., None].expand(-1, -1, -1, head_dim)
                    )
                    ck = ck.gather(
                        -2, attention[..., None].expand(-1, -1, -1, head_dim)
                    )
                    ck = torch.cat([key_states[..., :num_sink_tokens, :], ck], dim=-2)
                    cv = torch.cat([value_states[..., :num_sink_tokens, :], cv], dim=-2)
                    cq, ck = apply_rotary_pos_emb(cq, ck, cos, sin)
                    cv = torch.nn.functional.scaled_dot_product_attention(
                        cq,
                        ck,
                        cv,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                    )
                    second_results.append(cv)
                second_results = torch.cat(second_results, dim=-2)
                value_states = torch.cat([first_results, second_results], dim=-2)
            else:
                if qlen > max_len:
                    logger.warning(
                        f"Query length {qlen} is greater than max length {max_len}, which is not tested"
                    )
                if self.config.extend_context:
                    cos, sin = cache_kwargs["rotary_emb"](
                        value_states,
                        position_ids=None,
                        batch=key_states.shape[0],
                        length=qlen,
                    )
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin
                    )
                value_states = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                )
        else:
            if self.mgrs[layer_idx].ntotal > 0 and query_states.shape[-2] == 1:
                mgr_keys, mgr_values = self.mgrs[layer_idx].search(
                    query_states.reshape(self.num_kv_heads, -1, self.head_dim)
                )
                mgr_keys = mgr_keys.reshape(1, self.num_heads, -1, self.head_dim)
                mgr_values = mgr_values.reshape(1, self.num_heads, -1, self.head_dim)
                key_states = torch.cat(
                    [key_states[:, :, :1], mgr_keys, key_states[:, :, 1:]], dim=-2
                )
                value_states = torch.cat(
                    [value_states[:, :, :1], mgr_values, value_states[:, :, 1:]], dim=-2
                )

            if self.config.extend_context:
                cos, sin = cache_kwargs["rotary_emb"](
                    value_states,
                    position_ids=None,
                    batch=key_states.shape[0],
                    length=key_states.shape[-2],
                )

                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

            if layer_idx == 0:
                logger.debug(
                    f"Key length: {key_states.shape[-2]}, Value length: {value_states.shape[-2]}"
                )
            value_states = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

        if (
            self.num_sink_tokens
            and self.key_cache[layer_idx].shape[-2] > self.num_sink_tokens
        ):
            if self.sink_keys[layer_idx] is None:
                self.sink_keys[layer_idx] = self.key_cache[layer_idx][
                    0, :, : self.num_sink_tokens
                ]
                self.sink_values[layer_idx] = self.value_cache[layer_idx][
                    0, :, : self.num_sink_tokens
                ]
                self.key_cache[layer_idx] = self.key_cache[layer_idx][
                    :, :, self.num_sink_tokens :
                ]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][
                    :, :, self.num_sink_tokens :
                ]

        if self.key_cache[layer_idx].shape[-2] > self.residual_length:
            if self.mgrs[layer_idx].config.enabled:
                self.mgrs[layer_idx].add(
                    self.key_cache[layer_idx][0, :, : -self.residual_length],
                    self.value_cache[layer_idx][0, :, : -self.residual_length],
                )
            self.key_cache[layer_idx] = self.key_cache[layer_idx][
                :, :, -self.residual_length :
            ]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][
                :, :, -self.residual_length :
            ]

        return value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1
