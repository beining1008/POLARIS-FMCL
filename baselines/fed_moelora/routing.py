from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from models.moe_lora import iter_adapters


@dataclass(frozen=True)
class RoutingTelemetry:
    expert_mass: List[float]
    entropy: float
    dominant_expert: int
    dominant_share: float
    min_share: float
    load_imbalance: float
    num_batches: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "routing_mass": list(self.expert_mass),
            "routing_entropy": self.entropy,
            "dominant_expert": self.dominant_expert,
            "dominant_share": self.dominant_share,
            "min_share": self.min_share,
            "load_imbalance": self.load_imbalance,
            "routing_batches": self.num_batches,
        }


def batch_routing_mass(model: torch.nn.Module, num_experts: int) -> torch.Tensor:
    per_adapter: List[torch.Tensor] = []
    for _, adapter in iter_adapters(model):
        routing = adapter.last_routing
        if routing is None:
            continue
        flat = routing.reshape(-1, num_experts).to(torch.float32)
        per_adapter.append(flat.mean(dim=0).detach().to("cpu"))
    if not per_adapter:
        return torch.zeros(num_experts, dtype=torch.float32)
    return torch.stack(per_adapter, dim=0).mean(dim=0)


def routing_entropy(mass: List[float]) -> float:
    total = 0.0
    for value in mass:
        if value > 0.0:
            total -= value * math.log(value)
    return total


def expert_load_imbalance(mass: List[float], num_experts: int) -> float:
    if not mass:
        return 1.0
    return float(num_experts) * max(mass)


def aggregate_expert_mass(
    masses: List[List[float]], weights: List[float], num_experts: int
) -> List[float]:
    total = [0.0 for _ in range(num_experts)]
    for weight, mass in zip(weights, masses):
        if not mass:
            continue
        span = min(num_experts, len(mass))
        for index in range(span):
            total[index] += float(weight) * float(mass[index])
    return total


class RoutingMeter:
    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts
        self._accumulator = torch.zeros(num_experts, dtype=torch.float32)
        self._batches = 0

    def update(self, model: torch.nn.Module) -> None:
        self._accumulator = self._accumulator + batch_routing_mass(model, self.num_experts)
        self._batches += 1

    def _normalized(self) -> List[float]:
        uniform = [1.0 / self.num_experts for _ in range(self.num_experts)]
        if self._batches == 0:
            return uniform
        mass = self._accumulator / float(self._batches)
        total = float(mass.sum())
        if total <= 0.0:
            return uniform
        return [float(value) for value in (mass / total).tolist()]

    def result(self) -> RoutingTelemetry:
        mass = self._normalized()
        dominant = max(range(self.num_experts), key=lambda index: mass[index])
        return RoutingTelemetry(
            expert_mass=mass,
            entropy=routing_entropy(mass),
            dominant_expert=int(dominant),
            dominant_share=mass[dominant],
            min_share=min(mass),
            load_imbalance=expert_load_imbalance(mass, self.num_experts),
            num_batches=self._batches,
        )
