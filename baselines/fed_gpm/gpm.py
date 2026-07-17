from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import torch

from models.moe_lora import iter_adapters

EPS = 1e-12


def _frobenius_sq(matrix: torch.Tensor) -> torch.Tensor:
    return torch.sum(matrix.to(torch.float32).pow(2))


def _energy_rank(
    singular_values: torch.Tensor,
    threshold: float,
    total_energy: torch.Tensor,
    captured_energy: torch.Tensor,
) -> int:
    threshold_energy = threshold * total_energy
    if bool(captured_energy >= threshold_energy):
        return 0
    squared = singular_values.to(torch.float32).pow(2)
    cumulative = torch.cumsum(squared, dim=0) + captured_energy
    reached = torch.nonzero(cumulative >= threshold_energy, as_tuple=False)
    if reached.numel() > 0:
        return int(reached[0].item()) + 1
    return int(singular_values.shape[0])


@dataclass
class GradientProjectionMemory:
    energy_threshold_base: float
    energy_threshold_increment: float
    bases: Dict[str, torch.Tensor] = field(default_factory=dict)

    def threshold_for(self, task_id: int) -> float:
        return self.energy_threshold_base + self.energy_threshold_increment * float(task_id)

    def has_basis(self, layer_name: str) -> bool:
        basis = self.bases.get(layer_name)
        return basis is not None and basis.numel() > 0

    def total_columns(self) -> int:
        return int(sum(basis.shape[1] for basis in self.bases.values()))

    def update_memory(self, representations: Dict[str, torch.Tensor], task_id: int) -> None:
        threshold = self.threshold_for(task_id)
        for layer_name, representation in representations.items():
            matrix = representation.to(torch.float32)
            existing = self.bases.get(layer_name)
            if existing is None or existing.numel() == 0:
                self.bases[layer_name] = self._initial_basis(matrix, threshold)
            else:
                addition = self._incremental_basis(matrix, existing.to(torch.float32), threshold)
                if addition.numel() > 0:
                    self.bases[layer_name] = torch.cat([existing.to(torch.float32), addition], dim=1)

    def _initial_basis(self, matrix: torch.Tensor, threshold: float) -> torch.Tensor:
        left, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
        total_energy = torch.sum(singular.pow(2)).clamp_min(EPS)
        zero = total_energy.new_zeros(())
        rank = _energy_rank(singular, threshold, total_energy, zero)
        return left[:, :rank].contiguous()

    def _incremental_basis(
        self, matrix: torch.Tensor, basis: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        residual = matrix - basis @ (basis.t() @ matrix)
        total_energy = _frobenius_sq(matrix).clamp_min(EPS)
        captured_energy = (total_energy - _frobenius_sq(residual)).clamp_min(0.0)
        left, singular, _ = torch.linalg.svd(residual, full_matrices=False)
        rank = _energy_rank(singular, threshold, total_energy, captured_energy)
        if rank == 0:
            return matrix.new_zeros((matrix.shape[0], 0))
        candidate = left[:, :rank]
        candidate = candidate - basis @ (basis.t() @ candidate)
        orthonormal, _ = torch.linalg.qr(candidate)
        return orthonormal[:, : candidate.shape[1]].contiguous()

    def project(self, layer_name: str, grad: torch.Tensor) -> torch.Tensor:
        basis = self.bases.get(layer_name)
        if basis is None or basis.numel() == 0:
            return grad
        working = grad.to(torch.float32)
        projected = working - (working @ basis) @ basis.t()
        return projected.to(grad.dtype)


def _representation_batch_count(batch) -> int:
    if isinstance(batch, dict):
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.shape[0])
    elif isinstance(batch, (list, tuple)):
        for value in batch:
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.shape[0])
    elif isinstance(batch, torch.Tensor) and batch.dim() > 0:
        return int(batch.shape[0])
    return 1


def _make_capture_hook(
    layer_name: str,
    captures: Dict[str, List[torch.Tensor]],
    counts: Dict[str, int],
    sample_cap: int,
) -> Callable:
    def hook(module, inputs, output) -> None:
        if not inputs:
            return
        activation = inputs[0]
        if not isinstance(activation, torch.Tensor):
            return
        remaining = sample_cap - counts[layer_name]
        if remaining <= 0:
            return
        flattened = activation.detach().reshape(-1, activation.shape[-1]).to(torch.float32)
        if flattened.shape[0] > remaining:
            selection = torch.randperm(flattened.shape[0], device=flattened.device)[:remaining]
            flattened = flattened.index_select(0, selection)
        captures[layer_name].append(flattened)
        counts[layer_name] += int(flattened.shape[0])

    return hook


def run_activation_forward(model, batch) -> None:
    raise NotImplementedError(
        "activation forward pass depends on the concrete multimodal backbone"
    )


def collect_representation_batches(model, loader, sample_cap: int) -> int:
    was_training = model.training
    model.eval()
    seen = 0
    with torch.no_grad():
        for batch in loader:
            run_activation_forward(model, batch)
            seen += _representation_batch_count(batch)
            if seen >= sample_cap:
                break
    if was_training:
        model.train()
    return seen


def get_representation_matrix(model, loader, sample_cap: int) -> Dict[str, torch.Tensor]:
    captures: Dict[str, List[torch.Tensor]] = collections.defaultdict(list)
    counts: Dict[str, int] = collections.defaultdict(int)
    handles = []
    for layer_name, adapter in iter_adapters(model):
        handles.append(
            adapter.register_forward_hook(
                _make_capture_hook(layer_name, captures, counts, sample_cap)
            )
        )
    try:
        collect_representation_batches(model, loader, sample_cap)
    finally:
        for handle in handles:
            handle.remove()
    representations: Dict[str, torch.Tensor] = {}
    for layer_name, chunks in captures.items():
        if not chunks:
            continue
        features = torch.cat(chunks, dim=0)
        representations[layer_name] = features.t().contiguous()
    return representations
