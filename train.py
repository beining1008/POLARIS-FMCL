from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from data.benchmarks import BENCHMARKS
from harness.config import ExperimentConfig
from harness.registry import available_methods
from harness.runner import FederatedContinualRunner
from models.backbone import BACKBONES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("--method", default="polaris", choices=available_methods())
    parser.add_argument("--backbone", default="llava-1.5-7b", choices=sorted(BACKBONES))
    parser.add_argument("--benchmark", default="coin6", choices=sorted(BENCHMARKS))
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--rounds-per-task", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        backbone=args.backbone,
        benchmark=args.benchmark,
        method=args.method,
        num_experts=args.experts,
        top_k=args.top_k,
        num_clients=args.clients,
        dirichlet_beta=args.beta,
        rounds_per_task=args.rounds_per_task,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


def default_output(config: ExperimentConfig) -> str:
    return (
        f"results/{config.method}_{config.backbone}_{config.benchmark}"
        f"_beta{config.dirichlet_beta}_seed{config.seed}.json"
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    runner = FederatedContinualRunner(config)
    matrix = runner.run()
    summary = matrix.summary()
    payload = {
        "config": asdict(config),
        "tasks": list(BENCHMARKS[config.benchmark]),
        "zero_shot": matrix.zero_shot,
        "accuracy_matrix": matrix.values,
        "metrics": summary,
    }
    output = Path(args.output or default_output(config))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(
        f"{config.method} | {config.backbone} | {config.benchmark} | "
        f"beta={config.dirichlet_beta} | seed={config.seed} | "
        f"AA={summary['AA']:.2f} BWT={summary['BWT']:.2f} FWT={summary['FWT']:.2f} | {output}"
    )


if __name__ == "__main__":
    main()
