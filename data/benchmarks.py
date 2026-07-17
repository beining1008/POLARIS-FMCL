from __future__ import annotations

from typing import Dict, List, Tuple

from data.partition import dirichlet_partition

COIN6: Tuple[str, ...] = (
    "scienceqa",
    "textvqa",
    "imagenet",
    "gqa",
    "vqav2",
    "ocrvqa",
)

COIN_LONG10: Tuple[str, ...] = (
    "docvqa",
    "okvqa",
    "chartqa",
    "pathvqa",
    "aokvqa",
    "infovqa",
    "iconqa",
    "vizwiz",
    "clevr-math",
    "ai2d",
)

BENCHMARKS: Dict[str, Tuple[str, ...]] = {
    "coin6": COIN6,
    "coin-long10": COIN_LONG10,
}


def build_task_dataset(task: str, split: str, tokenizer=None):
    raise NotImplementedError


def build_loader(dataset, batch_size: int, shuffle: bool):
    raise NotImplementedError


def build_client_loaders(
    task: str,
    num_clients: int,
    beta: float,
    batch_size: int,
    seed: int,
    tokenizer=None,
) -> List:
    dataset = build_task_dataset(task, split="train", tokenizer=tokenizer)
    shards = dirichlet_partition(len(dataset), num_clients=num_clients, beta=beta, seed=seed)
    loaders = []
    for indices in shards:
        subset = dataset.select(indices)
        loaders.append(build_loader(subset, batch_size=batch_size, shuffle=True))
    return loaders


def build_eval_loader(task: str, batch_size: int, tokenizer=None):
    dataset = build_task_dataset(task, split="test", tokenizer=tokenizer)
    return build_loader(dataset, batch_size=batch_size, shuffle=False)
