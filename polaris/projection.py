from __future__ import annotations

from typing import Dict

import torch

from models.moe_lora import MoELoRALinear
from polaris.pefosu import union_key


class BilateralProjector:
    def __init__(self) -> None:
        self.bases: Dict[str, torch.Tensor] = {}

    def load(self, bases: Dict[str, torch.Tensor]) -> None:
        self.bases = dict(bases)

    def has_bases(self) -> bool:
        return len(self.bases) > 0

    @staticmethod
    def warmup_strength(step: int, warmup_steps: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(float(step + 1) / float(warmup_steps), 1.0)

    @staticmethod
    def _project_pair(
        grad_a: torch.Tensor, grad_b: torch.Tensor, basis: torch.Tensor, strength: float
    ) -> None:
        basis32 = basis.to(torch.float32)
        ga = grad_a.to(torch.float32)
        gb = grad_b.to(torch.float32)
        ga_projected = ga - strength * ((ga @ basis32) @ basis32.transpose(0, 1))
        gb_projected = gb - strength * (basis32 @ (basis32.transpose(0, 1) @ gb))
        grad_a.copy_(ga_projected.to(grad_a.dtype))
        grad_b.copy_(gb_projected.to(grad_b.dtype))

    def apply(self, adapters: Dict[str, MoELoRALinear], strength: float) -> int:
        if not self.bases or strength <= 0.0:
            return 0
        projected_experts = 0
        for layer_name, adapter in adapters.items():
            for expert_index in range(adapter.num_experts):
                basis = self.bases.get(union_key(layer_name, expert_index))
                if basis is None:
                    continue
                lora_a, lora_b = adapter.expert_parameters(expert_index)
                if lora_a.grad is None or lora_b.grad is None:
                    continue
                self._project_pair(lora_a.grad, lora_b.grad, basis, strength)
                projected_experts += 1
        return projected_experts

    def residual_alignment(self, adapters: Dict[str, MoELoRALinear]) -> Dict[str, float]:
        alignment: Dict[str, float] = {}
        for layer_name, adapter in adapters.items():
            layer_total = 0.0
            layer_inside = 0.0
            for expert_index in range(adapter.num_experts):
                basis = self.bases.get(union_key(layer_name, expert_index))
                lora_a, lora_b = adapter.expert_parameters(expert_index)
                if basis is None or lora_a.grad is None or lora_b.grad is None:
                    continue
                basis32 = basis.to(torch.float32)
                ga = lora_a.grad.to(torch.float32)
                gb = lora_b.grad.to(torch.float32)
                layer_inside += float((ga @ basis32).square().sum() + (basis32.transpose(0, 1) @ gb).square().sum())
                layer_total += float(ga.square().sum() + gb.square().sum())
            if layer_total > 0.0:
                alignment[layer_name] = layer_inside / layer_total
        return alignment
