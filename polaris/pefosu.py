from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class SubspaceUnionResult:
    basis: torch.Tensor
    eigenvalues: torch.Tensor
    spectral_cut: float
    retained_variance_ratio: float
    carry_over_residual: float
    retention_bound: float


class PEFOSUAggregator:
    def __init__(self) -> None:
        self.bases: Dict[str, torch.Tensor] = {}
        self.eigenvalues: Dict[str, torch.Tensor] = {}
        self.last_diagnostics: Dict[str, SubspaceUnionResult] = {}

    @staticmethod
    def _concatenate_factors(
        carry_basis: Optional[torch.Tensor],
        carry_eigenvalues: Optional[torch.Tensor],
        client_factors: List[Tuple[float, torch.Tensor]],
    ) -> torch.Tensor:
        columns = []
        if carry_basis is not None and carry_eigenvalues is not None:
            columns.append(carry_basis * carry_eigenvalues.clamp_min(0.0).sqrt().unsqueeze(0))
        for mix_weight, factor in client_factors:
            columns.append(factor.to(torch.float32) * (mix_weight ** 0.5))
        return torch.cat(columns, dim=1)

    def union(
        self,
        key: str,
        client_factors: List[Tuple[float, torch.Tensor]],
        rank: int,
    ) -> SubspaceUnionResult:
        carry_basis = self.bases.get(key)
        carry_eigenvalues = self.eigenvalues.get(key)
        stacked = self._concatenate_factors(carry_basis, carry_eigenvalues, client_factors)
        left, singular_values, _ = torch.linalg.svd(stacked, full_matrices=False)
        spectrum = singular_values.square()
        keep = min(rank, left.shape[1])
        new_basis = left[:, :keep].contiguous()
        new_eigenvalues = spectrum[:keep].contiguous()
        spectral_cut = float(spectrum[keep]) if spectrum.numel() > keep else 0.0
        total_mass = float(spectrum.sum().clamp_min(1e-12))
        retained_ratio = float(new_eigenvalues.sum()) / total_mass
        carry_residual = 0.0
        if carry_basis is not None:
            projector_residual = carry_basis - new_basis @ (new_basis.transpose(0, 1) @ carry_basis)
            carry_residual = float(projector_residual.norm(dim=0).square().max())
        retention_bound = 0.0
        if carry_eigenvalues is not None and carry_eigenvalues.numel() > 0:
            leading = float(new_eigenvalues[0]) if new_eigenvalues.numel() > 0 else 0.0
            carry_floor = float(carry_eigenvalues.min())
            if carry_floor <= spectral_cut or leading <= spectral_cut:
                retention_bound = 1.0
            else:
                retention_bound = max(0.0, (leading - carry_floor) / (leading - spectral_cut))
        result = SubspaceUnionResult(
            basis=new_basis,
            eigenvalues=new_eigenvalues,
            spectral_cut=spectral_cut,
            retained_variance_ratio=retained_ratio,
            carry_over_residual=carry_residual,
            retention_bound=retention_bound,
        )
        self.bases[key] = new_basis
        self.eigenvalues[key] = new_eigenvalues
        self.last_diagnostics[key] = result
        return result

    def retention_certificate(self, key: str, carry_over_mass: float) -> float:
        eigenvalues = self.eigenvalues.get(key)
        diagnostics = self.last_diagnostics.get(key)
        if eigenvalues is None or diagnostics is None or eigenvalues.numel() == 0:
            return 1.0
        leading = float(eigenvalues[0])
        cut = diagnostics.spectral_cut
        if carry_over_mass <= cut or leading <= cut:
            return 1.0
        return max(0.0, (leading - carry_over_mass) / (leading - cut))

    def basis_for(self, key: str) -> Optional[torch.Tensor]:
        return self.bases.get(key)

    def export_bases(self) -> Dict[str, torch.Tensor]:
        return dict(self.bases)

    def load_bases(self, bases: Dict[str, torch.Tensor], eigenvalues: Dict[str, torch.Tensor]) -> None:
        self.bases = dict(bases)
        self.eigenvalues = dict(eigenvalues)


def union_key(layer_name: str, expert_index: int) -> str:
    return f"{layer_name}::expert{expert_index}"
