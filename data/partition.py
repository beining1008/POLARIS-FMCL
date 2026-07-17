from __future__ import annotations

from typing import List, Optional, Sequence

import torch


def _split_counts(num_items: int, num_clients: int, beta: float, generator: torch.Generator) -> torch.Tensor:
    proportions = torch.distributions.Dirichlet(torch.full((num_clients,), beta)).sample()
    counts = torch.floor(proportions * num_items).to(torch.long)
    remainder = num_items - int(counts.sum())
    for _ in range(remainder):
        counts[int(torch.randint(num_clients, (1,), generator=generator))] += 1
    return counts


def dirichlet_partition(
    num_samples: int,
    num_clients: int,
    beta: float,
    seed: int,
    labels: Optional[Sequence[int]] = None,
) -> List[List[int]]:
    generator = torch.Generator().manual_seed(seed)
    if labels is None:
        counts = _split_counts(num_samples, num_clients, beta, generator)
        permutation = torch.randperm(num_samples, generator=generator).tolist()
        shards: List[List[int]] = []
        cursor = 0
        for count in counts.tolist():
            shards.append(sorted(permutation[cursor : cursor + count]))
            cursor += count
        return shards
    label_tensor = torch.tensor(list(labels), dtype=torch.long)
    grouped: List[List[int]] = [[] for _ in range(num_clients)]
    for value in torch.unique(label_tensor).tolist():
        members = (label_tensor == value).nonzero(as_tuple=True)[0]
        order = torch.randperm(members.numel(), generator=generator)
        members = members[order]
        counts = _split_counts(members.numel(), num_clients, beta, generator)
        cursor = 0
        for client_id, count in enumerate(counts.tolist()):
            grouped[client_id].extend(members[cursor : cursor + count].tolist())
            cursor += count
    return [sorted(shard) for shard in grouped]
