from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from models.moe_lora import MoELoRALinear, layer_label
from polaris.pefosu import union_key


class GammaAccumulator:
    def __init__(self) -> None:
        self.numerator: Dict[str, float] = {}
        self.denominator: Dict[str, float] = {}

    def accumulate(self, adapters: Dict[str, MoELoRALinear], bases: Dict[str, torch.Tensor]) -> None:
        for layer_name, adapter in adapters.items():
            label = layer_label(adapter, layer_name)
            routing = adapter.last_routing
            if routing is None:
                continue
            expert_mass = routing.to(torch.float32).reshape(-1, adapter.num_experts).mean(dim=0)
            for expert_index in range(adapter.num_experts):
                basis = bases.get(union_key(layer_name, expert_index))
                lora_a, lora_b = adapter.expert_parameters(expert_index)
                if basis is None or lora_a.grad is None or lora_b.grad is None:
                    continue
                mass = float(expert_mass[expert_index])
                if mass <= 0.0:
                    continue
                basis32 = basis.to(torch.float32)
                ga = lora_a.grad.to(torch.float32)
                gb = lora_b.grad.to(torch.float32)
                inside = float((ga @ basis32).square().sum() + (basis32.transpose(0, 1) @ gb).square().sum())
                total = float(ga.square().sum() + gb.square().sum())
                self.numerator[label] = self.numerator.get(label, 0.0) + mass * inside
                self.denominator[label] = self.denominator.get(label, 0.0) + mass * total

    def export(self) -> Dict[str, Tuple[float, float]]:
        return {
            layer: (self.numerator[layer], self.denominator.get(layer, 0.0))
            for layer in self.numerator
        }


class OnlineGammaEstimator:
    def __init__(self) -> None:
        self._per_task: List[Dict[str, float]] = []

    @staticmethod
    def merge_clients(
        client_statistics: List[Tuple[float, Dict[str, Tuple[float, float]]]]
    ) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for mix_weight, statistics in client_statistics:
            for layer, (numerator, denominator) in statistics.items():
                if denominator <= 0.0:
                    continue
                merged[layer] = merged.get(layer, 0.0) + mix_weight * (numerator / denominator)
        return merged

    def update_task(self, merged: Dict[str, float]) -> None:
        if merged:
            self._per_task.append(dict(merged))

    def landscape(self) -> Dict[str, float]:
        running: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for task_values in self._per_task:
            for layer, value in task_values.items():
                running[layer] = running.get(layer, 0.0) + value
                counts[layer] = counts.get(layer, 0) + 1
        return {layer: running[layer] / counts[layer] for layer in running}

    def tasks_observed(self) -> int:
        return len(self._per_task)

    def latest(self) -> Optional[Dict[str, float]]:
        return self._per_task[-1] if self._per_task else None
