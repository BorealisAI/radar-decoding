# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import transformers

SUPPORTED_METHODS = ["radar", "naive", "streaming", "snapkv", "landmark"]


def load_model(
    model_id,
    dtype=None,
    method="naive",
    residual_length=1024,
    target_tokens=None,
    topk=None,
    aspect_ratio=1,
    projection_dim=None,
    num_sink_tokens=1,
    ablation="none",
):
    config = transformers.AutoConfig.from_pretrained(model_id)

    if dtype is not None:
        dtype = getattr(torch, dtype)
    else:
        dtype = config.torch_dtype

    if method == "naive":
        return transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )

    if method == "radar" or method == "streaming":
        import radar

        model_id = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        config = model_id.config
        return radar.convert_(
            model_id,
            config=radar.RadarCacheConfig(
                hdim=config.hidden_size,
                residual_length=residual_length + (target_tokens or 0 if method == "streaming" else 0),
                compute_dtype=dtype,
                device="cuda" if torch.cuda.is_available() else "cpu",
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                num_sink_tokens=num_sink_tokens,
                projection_dim=projection_dim,
                num_layers=config.num_hidden_layers,
                enabled=method == "radar",
                ablation=ablation,
                topk=topk,
                target_tokens=target_tokens,
                max_length=config.max_position_embeddings,
                aspect_ratio=aspect_ratio,
            ),
        )
    elif method == "snapkv":
        from snapkv.monkeypatch.monkeypatch import (  # type: ignore
            replace_llama,
            replace_mistral,
            replace_mixtral,
        )

        replace_llama()
        replace_mistral()
        replace_mixtral()

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        layers = len(model.model.layers)
        window_sizes = [residual_length] * layers
        max_capacity_prompts = [target_tokens + residual_length] * layers
        kernel_sizes = [7] * layers
        pooling = "maxpool"
        for i in range(layers):
            model.model.layers[i].self_attn.config.window_size = window_sizes[i]
            model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
            model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
            model.model.layers[i].self_attn.config.pooling = pooling
        return model
    elif method == "landmark":
        assert "landmark" in model_id
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        return model
    else:
        raise NotImplementedError(f"Method {method} is not supported.")

