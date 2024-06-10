# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import logging
from typing import List

import torch
import transformers
import wandb

from research.longbench import longbench
from research.utils import SUPPORTED_METHODS, load_model


def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = load_model(
        model_id=args.model,
        method=args.method,
        dtype=args.dtype,
        residual_length=args.residual_length,
        num_sink_tokens=args.num_sink_tokens,
        ablation=args.ablation,
        projection_dim=args.projection_dim,
        target_tokens=args.target_tokens,
        aspect_ratio=args.aspect_ratio,
    )
    if torch.cuda.is_available():
        model = model.cuda()

    exp_name = args.exp_name or f"{args.method}_{args.model}"
    wandb.init(project="longbench", config=vars(args), name=exp_name)

    main_table = wandb.Table(columns=["task", "score", "std", "task_type", "metric"])

    for task_name in longbench.available_tasks(
        language=args.language,
        category=args.category,
        version_e=args.version_e,
        task_name=args.task,
    ):
        res = longbench.evalute(
            task_name=task_name,
            model=model,
            version_e=args.version_e,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        main_table.add_data(
            task_name, res["mean"], res["std"], res["task_type"], res["metric"]
        )
        individual_results = res.pop("details")
        task_table = wandb.Table(
            columns=list(individual_results[0].keys()),
            data=[
                [
                    " | ".join(item) if isinstance(item, List) else item
                    for item in row.values()
                ]
                for row in individual_results
            ],
        )
        wandb.log(
            {
                "score": res["mean"],
                "std": res["std"],
                task_name: task_table,
            }
        )
        wandb.summary.update({task_name: res["mean"]})
    wandb.log({"results": main_table})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--method", type=str, required=True, choices=SUPPORTED_METHODS
    )
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-e", "--exp-name", type=str)
    parser.add_argument("-t", "--task", type=str, default=None, nargs="+")
    parser.add_argument("-cat", "--category", type=str, default=None, nargs="+")
    parser.add_argument("-lang", "--language", type=str, default=None, nargs="+")
    parser.add_argument("--version-e", action="store_true")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--residual-length", type=int, default=32)
    parser.add_argument("--num-sink-tokens", type=int, default=1)
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "last", "random", "exact"],
    )
    parser.add_argument("--projection-dim", type=int, default=None)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"]
    )
    parser.add_argument("--target-tokens", type=int, default=1024)
    parser.add_argument("--aspect-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["INFO", "DEBUG", "ERROR", "WARNING"],
    )

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    main(args)
