from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch

from models.moe_lora import MoELoRALinear, iter_adapters


def read_layer_activation(module: MoELoRALinear, hook_input: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    raise NotImplementedError


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


@dataclass(frozen=True)
class LayerSplit:
    minor_basis: torch.Tensor
    minor_energy: torch.Tensor
    r_split: int


class SubspaceSplitState:
    def __init__(self, split_alpha: float, split_temperature: float, activation_batches: int) -> None:
        self.split_alpha = float(split_alpha)
        self.split_temperature = float(split_temperature)
        self.activation_batches = int(activation_batches)
        self.cur_matrix: Dict[str, torch.Tensor] = {}
        self.feature_list: Dict[str, LayerSplit] = {}
        self.old_weight: Dict[str, torch.Tensor] = {}
        self._token_count: Counter = Counter()

    def has_split(self) -> bool:
        return len(self.feature_list) > 0

    def has_split_for(self, name: str) -> bool:
        return name in self.feature_list

    def old_weight_for(self, name: str) -> Optional[torch.Tensor]:
        return self.old_weight.get(name)

    def _accumulate(self, name: str, feats: torch.Tensor) -> None:
        x = feats.detach().to(torch.float32)
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        gram = x.t().mm(x)
        buffer = self.cur_matrix.get(name)
        if buffer is None:
            self.cur_matrix[name] = gram
        else:
            buffer.add_(gram)
        self._token_count[name] += int(x.shape[0])

    def _make_capture_hook(self, name: str):
        def hook(module: MoELoRALinear, args: Tuple[torch.Tensor, ...]):
            feats = read_layer_activation(module, args)
            self._accumulate(name, feats)
            return None

        return hook

    def _register_capture(self, model: torch.nn.Module):
        handles = []
        for name, adapter in iter_adapters(model):
            handles.append(adapter.register_forward_pre_hook(self._make_capture_hook(name)))
        return handles

    def _capture(self, model: torch.nn.Module, loader: Iterable, device: torch.device) -> None:
        handles = self._register_capture(model)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index >= self.activation_batches:
                    break
                model(**move_batch(batch, device))
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    def update_grad_before_train(self, model: torch.nn.Module, loader: Iterable, device: torch.device) -> None:
        self.cur_matrix.clear()
        self._token_count.clear()
        self._capture(model, loader, device)

    def update_grad_after_train(self, model: torch.nn.Module, loader: Iterable, device: torch.device) -> None:
        self._capture(model, loader, device)

    def split(self, task_id: int) -> None:
        self.feature_list.clear()
        for name, matrix in self.cur_matrix.items():
            count = max(self._token_count.get(name, 0), 1)
            cov = matrix.to(torch.float32) / float(count)
            basis, singular, _ = torch.linalg.svd(cov, full_matrices=False)
            singular = singular.to(torch.float32)
            total = int(singular.numel())
            if total < 2:
                continue
            mass = torch.clamp(singular.sum(), min=1e-12)
            cumulative_ratio = torch.cumsum(singular, dim=0) / mass
            counts = torch.arange(1, total + 1, dtype=torch.float32, device=singular.device)
            scores = float(task_id + 1) * (1.0 - cumulative_ratio) - self.split_alpha * counts
            r_split = int(torch.argmin(scores).item()) + 1
            r_split = max(1, min(r_split, total - 1))
            self.feature_list[name] = LayerSplit(
                minor_basis=basis[:, r_split:].contiguous(),
                minor_energy=singular[r_split:].contiguous(),
                r_split=r_split,
            )

    def minor_subspace_init(self, name: str, rank: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        split = self.feature_list[name]
        basis = split.minor_basis.to(torch.float32)
        width = basis.shape[1]
        gain = torch.softmax(split.minor_energy.to(torch.float32) / self.split_temperature, dim=0)
        coefficients = torch.randn(rank, width, dtype=torch.float32, device=basis.device) * gain.unsqueeze(0)
        rows = coefficients.mm(basis.t())
        rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-8)
        rows = rows * math.sqrt(1.0 / rows.shape[1])
        return rows.to(dtype=dtype, device=device)

    def project_to_minor(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        split = self.feature_list.get(name)
        if split is None:
            return grad
        basis = split.minor_basis.to(torch.float32)
        gradient = grad.to(torch.float32)
        projected = gradient.mm(basis).mm(basis.t())
        return projected.to(dtype=grad.dtype)

    def fold_old_weight(self, name: str, delta: torch.Tensor) -> None:
        value = delta.to(torch.float32)
        buffer = self.old_weight.get(name)
        if buffer is None:
            self.old_weight[name] = value
        else:
            buffer.add_(value)
