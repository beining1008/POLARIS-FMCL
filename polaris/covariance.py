from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch

from models.moe_lora import MoELoRALinear


class ExpertCovariance:
    def __init__(self, dim: int, device: Optional[torch.device] = None) -> None:
        self.dim = dim
        self.matrix = torch.zeros(dim, dim, dtype=torch.float32, device=device)
        self.weight_mass = 0.0

    def update(self, grad_a: torch.Tensor, grad_b: torch.Tensor, routing_weight: float) -> None:
        if routing_weight <= 0.0:
            return
        ga = grad_a.to(torch.float32)
        gb = grad_b.to(torch.float32)
        self.matrix.add_(ga.transpose(0, 1) @ ga, alpha=routing_weight)
        self.matrix.add_(gb @ gb.transpose(0, 1), alpha=routing_weight)
        self.weight_mass += routing_weight

    def spectrum(self) -> torch.Tensor:
        eigenvalues = torch.linalg.eigvalsh(self.matrix)
        return torch.flip(eigenvalues, dims=(0,)).clamp_min(0.0)

    def factorize(self, rank: int) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(self.matrix)
        order = torch.argsort(eigenvalues, descending=True)
        top = order[: max(rank, 1)]
        basis = eigenvectors[:, top]
        scale = eigenvalues[top].clamp_min(0.0).sqrt()
        return basis * scale.unsqueeze(0)

    def reset(self) -> None:
        self.matrix.zero_()
        self.weight_mass = 0.0


class CovarianceBank:
    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts
        self._store: Dict[str, List[Optional[ExpertCovariance]]] = {}

    def _slot(self, layer_name: str, expert_index: int, dim: int, device: torch.device) -> ExpertCovariance:
        if layer_name not in self._store:
            self._store[layer_name] = [None for _ in range(self.num_experts)]
        slot = self._store[layer_name][expert_index]
        if slot is None:
            slot = ExpertCovariance(dim, device)
            self._store[layer_name][expert_index] = slot
        return slot

    def accumulate(self, adapters: Dict[str, MoELoRALinear]) -> None:
        for layer_name, adapter in adapters.items():
            routing = adapter.last_routing
            if routing is None:
                continue
            expert_mass = routing.to(torch.float32).reshape(-1, adapter.num_experts).mean(dim=0)
            for expert_index in range(adapter.num_experts):
                lora_a, lora_b = adapter.expert_parameters(expert_index)
                if lora_a.grad is None or lora_b.grad is None:
                    continue
                mass = float(expert_mass[expert_index])
                slot = self._slot(layer_name, expert_index, lora_a.shape[1], lora_a.device)
                slot.update(lora_a.grad, lora_b.grad, mass)

    def factorize(self, layer_name: str, expert_index: int, rank: int) -> Optional[torch.Tensor]:
        slots = self._store.get(layer_name)
        if slots is None or slots[expert_index] is None:
            return None
        return slots[expert_index].factorize(rank)

    def pooled_matrix(self, layer_name: str) -> Optional[torch.Tensor]:
        slots = self._store.get(layer_name)
        if slots is None:
            return None
        pooled: Optional[torch.Tensor] = None
        for slot in slots:
            if slot is None:
                continue
            pooled = slot.matrix.clone() if pooled is None else pooled + slot.matrix
        return pooled

    def layers(self) -> Iterable[str]:
        return self._store.keys()

    def reset(self) -> None:
        for slots in self._store.values():
            for slot in slots:
                if slot is not None:
                    slot.reset()


def merge_pooled_matrices(
    banks: List[Tuple[float, CovarianceBank]], layer_name: str
) -> Optional[torch.Tensor]:
    merged: Optional[torch.Tensor] = None
    for mix_weight, bank in banks:
        pooled = bank.pooled_matrix(layer_name)
        if pooled is None:
            continue
        scaled = pooled * mix_weight
        merged = scaled if merged is None else merged + scaled
    return merged
