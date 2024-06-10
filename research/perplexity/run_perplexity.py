# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import logging
import time
from typing import Iterable, Optional

import datasets
import torch
import transformers
import wandb
from tqdm.auto import tqdm
from transformers.cache_utils import DynamicCache

from research.utils import SUPPORTED_METHODS, load_model


def setup_logging(level):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=getattr(logging, level.upper()),
    )


@torch.inference_mode()
def compute_perplexity(
    args,
    model,
    tokenizer,
    dataset,
    data_column: str = "text",
    sample_ids: Optional[Iterable[int]] = [0],
    prefill: int = 0,
) -> None:
    loss_fn = torch.nn.functional.cross_entropy
    dataset = dataset.select(sample_ids)[data_column]
    text = "\n\n".join(dataset)
    model.eval()

    # build experiment name
    model_name = args.model_id.split("/")[-1].replace("-", "_")
    cache_name = args.cache_implementation
    sliding_window = args.residual_length
    num_sink_tokens = args.num_sink_tokens
    ablation = args.ablation
    if cache_name == "radar" and ablation != "none":
        cache_name = f"radar({ablation})"

    experiment = f"{cache_name}-{model_name}-{num_sink_tokens}-{sliding_window}"
    wandb.init(project="perplexity", name=experiment, config=vars(args))

    encodings = tokenizer(text, return_tensors="pt")
    encodings = encodings.to(model.device)

    seq_len = encodings.input_ids.size(1)
    pbar = tqdm(range(0, seq_len - 1), dynamic_ncols=True)
    all_nlls = torch.empty(seq_len - 1, dtype=torch.float32, device=model.device)
    past_key_values = (
        DynamicCache() if "radar" not in args.cache_implementation else None
    )

    # Pre-fill the cache
    if prefill > 0:
        input_ids = encodings.input_ids[:, :prefill]
        outputs = model(input_ids, use_cache=True, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        label = encodings.input_ids[:, 1 : prefill + 1]
        logits = outputs.logits.view(-1, model.config.vocab_size)
        label = label.view(-1)
        neg_log_likelihood = loss_fn(logits, label, reduction="none")
        all_nlls[:prefill] = neg_log_likelihood

    pbar.update(prefill)
    elapsed = 0.0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute perplexity
    for idx in range(prefill, seq_len - 1):
        model.eval()
        input_ids = encodings.input_ids[:, idx : idx + 1]
        if not args.ignore_latency:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_t = time.perf_counter()
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        if not args.ignore_latency:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - start_t
        else:
            latency = 0.0
        logits = outputs.logits.view(-1, model.config.vocab_size)
        past_key_values = outputs.past_key_values
        label = encodings.input_ids[:, idx + 1 : idx + 2].view(-1)
        neg_log_likelihood = loss_fn(logits, label, reduction="none")
        all_nlls[idx] = neg_log_likelihood

        # Gather data for logging
        elapsed += latency
        step_data = {
            "nll": neg_log_likelihood.item(),
            "latency": latency,
            "memory": torch.cuda.memory_allocated() / 1024**3,
            "mean_nll": all_nlls[: idx + 1].mean().item(),
            "perplexity": all_nlls[: idx + 1].mean().exp().item(),
            "elapsed": elapsed,
            "length": idx + 1,
        }

        # Log data to wandb for every 10 steps
        if idx % 10 == 0 or idx == seq_len - 2:
            wandb.log(step_data, step=idx + 1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update progress bar
        pbar.set_description(
            f"Loss: {step_data['nll']:.2f} | Log PPL: {step_data['mean_nll']:.2f}"
        )
        pbar.update(1)

    wandb.summary.update(step_data)
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cache_implementation",
        type=str,
        default="naive",
        choices=SUPPORTED_METHODS,
    )

    # Model args
    parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
    )

    # Dataset args
    parser.add_argument("--dataset_name", type=str, default="emozilla/pg19-test")
    parser.add_argument("--data_column", type=str, default="text")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument(
        "--split", type=str, default="test", choices=["validation", "test", "train"]
    )
    parser.add_argument("--sample_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--dtype", type=str, default=None)

    parser.add_argument("--residual_length", type=int, default=1024)
    parser.add_argument("--num_sink_tokens", type=int, default=1)
    parser.add_argument("--ablation", type=str, default="none")
    parser.add_argument("--topk", type=int)
    parser.add_argument("--prefill", type=int, default=0)
    parser.add_argument("--aspect-ratio", type=float, default=4.0)
    parser.add_argument("--extend-context", action="store_true")
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--ignore-latency", action="store_true")
    parser.add_argument("--projection-dim", type=int, default=None)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["INFO", "DEBUG", "ERROR", "WARNING"],
    )

    args = parser.parse_args()

    setup_logging(args.log_level)

    model = load_model(
        args.model_id,
        dtype=args.dtype,
        method=args.cache_implementation,
        residual_length=args.residual_length,
        target_tokens=args.target_tokens,
        topk=args.topk,
        aspect_ratio=args.aspect_ratio,
        num_sink_tokens=args.num_sink_tokens,
        ablation=args.ablation,
        projection_dim=args.projection_dim,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=False,
    )

    # Set up the dataset
    dataset = datasets.load_dataset(args.dataset_name, split=args.split)
    if args.data_dir is not None:
        dataset = dataset.filter(lambda ex: ex["lang"].lower() in args.data_dir)

    compute_perplexity(
        args,
        model,
        tokenizer,
        dataset,
        data_column=args.data_column,
        sample_ids=args.sample_ids,
        prefill=args.prefill,
    )


if __name__ == "__main__":
    main()
