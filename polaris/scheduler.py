from __future__ import annotations

from typing import Dict, Optional

import torch


def effective_rank(eigenvalues: torch.Tensor, threshold: float = 1e-3) -> int:
    if eigenvalues.numel() == 0:
        return 0
    leading = float(eigenvalues.max().clamp_min(1e-12))
    return int((eigenvalues >= threshold * leading).sum())


class InterferenceScheduler:
    def __init__(self, budget_ratio: float = 0.75, rank_threshold: float = 1e-3) -> None:
        self.budget_ratio = budget_ratio
        self.rank_threshold = rank_threshold

    def uniform(self, layer_names, per_layer_budget: int) -> Dict[str, int]:
        return {layer: per_layer_budget for layer in layer_names}

    def capacity_caps(self, per_task_ranks: Dict[str, int]) -> Dict[str, int]:
        return {
            layer: max(int(self.budget_ratio * rank), 1)
            for layer, rank in per_task_ranks.items()
        }

    def allocate(
        self,
        gamma: Dict[str, float],
        caps: Dict[str, int],
        total_budget: Optional[int] = None,
    ) -> Dict[str, int]:
        layers = [layer for layer in caps if layer in gamma]
        if not layers:
            return dict(caps)
        cap_vector = {layer: max(caps[layer], 0) for layer in layers}
        budget = int(total_budget) if total_budget is not None else sum(cap_vector.values())
        budget = min(budget, sum(cap_vector.values()))
        active = {layer for layer in layers if cap_vector[layer] > 0}
        allocation = {layer: 0.0 for layer in layers}
        remaining = float(budget)
        while active and remaining > 1e-9:
            energy = sum(gamma[layer] ** 2 for layer in active)
            if energy <= 0.0:
                share = remaining / len(active)
                proposals = {layer: allocation[layer] + share for layer in active}
            else:
                proposals = {
                    layer: allocation[layer] + remaining * (gamma[layer] ** 2) / energy
                    for layer in active
                }
            clipped = {layer for layer in active if proposals[layer] >= cap_vector[layer]}
            if not clipped:
                allocation.update(proposals)
                break
            for layer in clipped:
                remaining -= cap_vector[layer] - allocation[layer]
                allocation[layer] = float(cap_vector[layer])
                active.discard(layer)
        return self._integerize(allocation, cap_vector, budget)

    @staticmethod
    def _integerize(
        allocation: Dict[str, float], caps: Dict[str, int], budget: int
    ) -> Dict[str, int]:
        floored = {layer: min(int(allocation[layer]), caps[layer]) for layer in allocation}
        deficit = budget - sum(floored.values())
        remainders = sorted(
            allocation.keys(),
            key=lambda layer: allocation[layer] - floored[layer],
            reverse=True,
        )
        for layer in remainders:
            if deficit <= 0:
                break
            if floored[layer] < caps[layer]:
                floored[layer] += 1
                deficit -= 1
        return floored
