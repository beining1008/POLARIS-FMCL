from __future__ import annotations

from typing import List

import torch


def dirichlet_partition(num_samples: int, num_clients: int, beta: float, seed: int) -> List[List[int]]:
    generator = torch.Generator().manual_seed(seed)
    proportions = torch.distributions.Dirichlet(
        torch.full((num_clients,), beta)
    ).sample()
    counts = torch.floor(proportions * num_samples).to(torch.long)
    remainder = num_samples - int(counts.sum())
    for _ in range(remainder):
        counts[int(torch.randint(num_clients, (1,), generator=generator))] += 1
    permutation = torch.randperm(num_samples, generator=generator).tolist()
    shards: List[List[int]] = []
    cursor = 0
    for count in counts.tolist():
        shard = sorted(permutation[cursor : cursor + count])
        shards.append(shard)
        cursor += count
    return shards
