from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

TensorDict = Dict[str, torch.Tensor]


@dataclass
class ClientPayload:
    client_id: int
    num_samples: int
    weights: TensorDict
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BroadcastPackage:
    weights: TensorDict
    extras: Dict[str, Any] = field(default_factory=dict)


class FederatedMethod(ABC):
    name: str = "base"

    def __init__(self, config, backbone_spec) -> None:
        self.config = config
        self.backbone_spec = backbone_spec
        self._broadcast: Optional[BroadcastPackage] = None

    @abstractmethod
    def build_model(self) -> nn.Module:
        ...

    @abstractmethod
    def local_update(
        self,
        client_id: int,
        task_id: int,
        round_id: int,
        model: nn.Module,
        loader: Iterable,
        broadcast: Optional[BroadcastPackage],
    ) -> ClientPayload:
        ...

    @abstractmethod
    def aggregate(self, payloads: List[ClientPayload], task_id: int, round_id: int) -> BroadcastPackage:
        ...

    def on_task_start(self, task_id: int) -> None:
        return None

    def on_task_end(self, task_id: int, payloads: List[ClientPayload]) -> None:
        return None

    def current_broadcast(self) -> Optional[BroadcastPackage]:
        return self._broadcast

    def set_broadcast(self, package: BroadcastPackage) -> None:
        self._broadcast = package

    def trainable_parameters(self, model: nn.Module) -> List[nn.Parameter]:
        return [p for p in model.parameters() if p.requires_grad]

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.trainable_parameters(model),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def build_lr_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int):
        if self.config.lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
