# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import functools
import inspect
import logging
import types

import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRotaryEmbedding,
)
from transformers.models.mistral.modeling_mistral import MistralAttention

from radar.modules.modeling_llama import SequentialLlamaRotaryEmbedding, llama_forward
from radar.modules.modeling_mistral import mistral_forward
from radar.radar_cache import RadarCache, RadarCacheConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_logger_level(level):
    logger.setLevel(level)


def replace_dynamic_cache_(model, cache_config):
    def wrapper(forward_fn):
        def wrapper_forward(*args, **kwargs):
            signature = inspect.signature(forward_fn)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            if "past_key_values" in bound_args.arguments:
                past_key_values = bound_args.arguments["past_key_values"]
                if not isinstance(past_key_values, RadarCache):
                    bound_args.arguments["past_key_values"] = RadarCache(cache_config)
            else:
                bound_args.arguments["past_key_values"] = RadarCache(cache_config)
            return forward_fn(*bound_args.args, **bound_args.kwargs)

        return wrapper_forward

    model.forward = wrapper(model.forward)
    return model


def _delete_module(module):
    for _name, _param in module.named_parameters():
        # Save device memory by clearing parameters
        setattr(module, _name, None)
        del _param
    del module


def convert_(model, config: RadarCacheConfig, forward_fn=None) -> None:
    def recursive_convert_(parent_name, parent_module):
        converted = False
        for name, module in parent_module.named_children():
            if isinstance(module, LlamaAttention):
                module.forward = types.MethodType(
                    functools.partial(forward_fn, cache_config=config), module
                )
                if config.extend_context:
                    module.rotary_emb = SequentialLlamaRotaryEmbedding.from_original(
                        module.rotary_emb,
                    )
                converted = True
            elif isinstance(module, MistralAttention):
                module.forward = types.MethodType(
                    functools.partial(mistral_forward, cache_config=config), module
                )
                converted = True
            elif isinstance(module, LlamaRotaryEmbedding) and config.extend_context:
                setattr(
                    parent_module,
                    name,
                    SequentialLlamaRotaryEmbedding.from_original(module),
                )
                _delete_module(module)

            elif isinstance(module, torch.nn.Module):
                subconverted = recursive_convert_(parent_name + "." + name, module)
                converted = converted or subconverted
        return converted

    if forward_fn is None:
        forward_fn = llama_forward
    converted = recursive_convert_("", model)

    if not converted:
        logger.warning(
            "No Llama or Mistral attention modules found in model. Model not converted."
        )
    else:
        replace_dynamic_cache_(model, config)

    if config.extend_context:
        extended_length = (config.max_length / config.topk) ** 2 * config.aspect_ratio
        model.config.max_position_embeddings = int(extended_length)
    return model


__all__ = ["convert_", "set_logger_level", "RadarCacheConfig", "RadarCache"]
