from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import torch

from models.moe_lora import iter_adapters

EPS = 1e-12


def _principal_rank(singular: torch.Tensor, energy: float) -> Tuple[int, float]:
    squared = singular.to(torch.float32).pow(2)
    total = torch.sum(squared).clamp_min(EPS)
    cumulative = torch.cumsum(squared, dim=0) / total
    reached = torch.nonzero(cumulative >= float(energy), as_tuple=False)
    if reached.numel() > 0:
        rank = int(reached[0].item()) + 1
    else:
        rank = int(singular.shape[0])
    rank = max(1, min(rank, int(singular.shape[0])))
    captured = float(cumulative[rank - 1].item())
    return rank, captured


@dataclass
class PrincipalResidualSplit:
    basis_out: torch.Tensor
    basis_in: torch.Tensor
    rank: int
    captured_energy: float

    @classmethod
    def from_weight(cls, weight: torch.Tensor, energy: float) -> PrincipalResidualSplit:
        matrix = weight.detach().to(torch.float32)
        left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        rank, captured = _principal_rank(singular, energy)
        basis_out = left[:, :rank].contiguous()
        basis_in = right[:rank, :].t().contiguous()
        return cls(basis_out=basis_out, basis_in=basis_in, rank=rank, captured_energy=captured)

    @property
    def residual_input_dim(self) -> int:
        return int(self.basis_in.shape[0]) - self.rank

    @property
    def residual_output_dim(self) -> int:
        return int(self.basis_out.shape[0]) - self.rank

    def project_input(self, grad: torch.Tensor) -> torch.Tensor:
        return grad - (grad @ self.basis_in) @ self.basis_in.t()

    def project_output(self, grad: torch.Tensor) -> torch.Tensor:
        return grad - self.basis_out @ (self.basis_out.t() @ grad)


@dataclass
class HistoricalFeatureDirections:
    history_rank: int
    covariance: Dict[str, torch.Tensor] = field(default_factory=dict)
    directions: Dict[str, torch.Tensor] = field(default_factory=dict)

    def has_basis(self, layer_name: str) -> bool:
        basis = self.directions.get(layer_name)
        return basis is not None and basis.numel() > 0

    def total_columns(self) -> int:
        return int(sum(basis.shape[1] for basis in self.directions.values()))

    def update_memory(self, feature_grams: Dict[str, torch.Tensor]) -> None:
        for layer_name, gram in feature_grams.items():
            matrix = gram.to(torch.float32)
            if layer_name in self.covariance:
                self.covariance[layer_name] = self.covariance[layer_name] + matrix
            else:
                self.covariance[layer_name] = matrix
            symmetric = 0.5 * (self.covariance[layer_name] + self.covariance[layer_name].t())
            _, eigenvectors = torch.linalg.eigh(symmetric)
            columns = int(eigenvectors.shape[1])
            width = min(int(self.history_rank), columns)
            self.directions[layer_name] = eigenvectors[:, columns - width :].contiguous()

    def project_input(self, layer_name: str, grad: torch.Tensor) -> torch.Tensor:
        basis = self.directions.get(layer_name)
        if basis is None or basis.numel() == 0:
            return grad
        return grad - (grad @ basis) @ basis.t()


def _make_covariance_hook(layer_name: str, grams: Dict[str, torch.Tensor]) -> Callable:
    def hook(module, inputs, output) -> None:
        if not inputs:
            return
        activation = inputs[0]
        if not isinstance(activation, torch.Tensor):
            return
        flattened = activation.detach().reshape(-1, activation.shape[-1]).to(torch.float32)
        gram = flattened.t() @ flattened
        if layer_name in grams:
            grams[layer_name].add_(gram)
        else:
            grams[layer_name] = gram

    return hook


def run_feature_forward(model, batch) -> None:
    raise NotImplementedError(
        "feature forward pass depends on the concrete multimodal backbone"
    )


def _collect_feature_batches(model, loader, batch_cap: int) -> int:
    was_training = model.training
    model.eval()
    seen = 0
    with torch.no_grad():
        for batch in loader:
            run_feature_forward(model, batch)
            seen += 1
            if seen >= batch_cap:
                break
    if was_training:
        model.train()
    return seen


def collect_feature_covariance(model, loader, batch_cap: int) -> Dict[str, torch.Tensor]:
    grams: Dict[str, torch.Tensor] = {}
    handles = []
    for layer_name, adapter in iter_adapters(model):
        handles.append(adapter.register_forward_hook(_make_covariance_hook(layer_name, grams)))
    try:
        _collect_feature_batches(model, loader, batch_cap)
    finally:
        for handle in handles:
            handle.remove()
    return grams
