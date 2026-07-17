from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RankPoolState:
    top_rank: int
    train_rank: int
    shared_ids: List[int] = field(default_factory=list)
    static_ids: List[int] = field(default_factory=list)
    trainable_ids: List[int] = field(default_factory=list)

    @classmethod
    def initial(cls, top_rank: int, train_rank: int) -> "RankPoolState":
        base_train = top_rank + train_rank
        trainable = list(range(base_train, base_train + train_rank))
        return cls(top_rank, train_rank, [], [], trainable)

    @property
    def pool_rank(self) -> int:
        return self.top_rank + 2 * self.train_rank

    @property
    def frozen_ids(self) -> List[int]:
        return self.shared_ids + self.static_ids

    @property
    def active_ids(self) -> List[int]:
        return self.shared_ids + self.static_ids + self.trainable_ids

    def repartition(self, counts: Tuple[int, int, int]) -> None:
        n_shared, n_static, n_train = counts
        base_static = self.top_rank
        base_train = self.top_rank + self.train_rank
        self.shared_ids = list(range(0, n_shared))
        self.static_ids = list(range(base_static, base_static + n_static))
        self.trainable_ids = list(range(base_train, base_train + n_train))

    def occupancy(self) -> Dict[str, int]:
        return {
            "shared": len(self.shared_ids),
            "static": len(self.static_ids),
            "trainable": len(self.trainable_ids),
        }


class RankPoolLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, pool_rank: int, scaling: float) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.pool_rank = pool_rank
        self.scaling = scaling
        dtype = base.weight.dtype
        self.lora_A_pool = nn.Parameter(torch.zeros(pool_rank, self.in_features, dtype=dtype))
        self.lora_B_pool = nn.Parameter(torch.zeros(self.out_features, pool_rank, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A_pool, a=math.sqrt(5))
        self.register_buffer("train_mask", torch.zeros(pool_rank, dtype=torch.float32))
        self.lora_A_pool.register_hook(self._grad_mask_a)
        self.lora_B_pool.register_hook(self._grad_mask_b)

    def _grad_mask_a(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self.train_mask.to(grad.dtype).view(-1, 1)

    def _grad_mask_b(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self.train_mask.to(grad.dtype).view(1, -1)

    def set_trainable(self, trainable_ids: List[int]) -> None:
        mask = torch.zeros(self.pool_rank, dtype=torch.float32, device=self.train_mask.device)
        if trainable_ids:
            index = torch.tensor(sorted(trainable_ids), dtype=torch.long)
            mask.index_fill_(0, index, 1.0)
        self.train_mask.copy_(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        a = self.lora_A_pool.to(x.dtype)
        b = self.lora_B_pool.to(x.dtype)
        delta = F.linear(F.linear(x, a), b)
        return base_out + delta * self.scaling


def iter_rank_pool(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, RankPoolLoRALinear):
            yield name, module


def _layer_index(name: str) -> int:
    parts = name.split(".")
    for left, right in zip(parts, parts[1:]):
        if left in ("layers", "h", "blocks") and right.isdigit():
            return int(right)
    return -1


def _parent_and_leaf(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_rank_pool_lora(
    model: nn.Module,
    adapter_layers: Tuple[int, ...],
    pool_rank: int,
    scaling: float,
    target_modules: Tuple[str, ...],
) -> Dict[str, RankPoolLoRALinear]:
    wrapped: Dict[str, RankPoolLoRALinear] = {}
    hits = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] in target_modules and _layer_index(name) in adapter_layers:
            hits.append((name, module))
    for name, module in hits:
        adapter = RankPoolLoRALinear(module, pool_rank, scaling)
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, adapter)
        wrapped[name] = adapter
    return wrapped
