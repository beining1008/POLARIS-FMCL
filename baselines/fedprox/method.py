from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.fedprox.proximal import ProximalAdamW
from harness.aggregation import extract_trainable, fedavg, load_weights
from harness.interfaces import BroadcastPackage, ClientPayload, FederatedMethod
from models.backbone import load_backbone
from models.moe_lora import inject_moe_lora

IGNORE_INDEX = -100


@dataclass(frozen=True)
class FedProxHyperparams:
    proximal_mu: float = 0.01
    mu_search_grid: Tuple[float, ...] = (0.001, 0.01, 0.1, 0.5, 1.0)


def _steps_per_epoch(loader: Any) -> int:
    try:
        return max(int(len(loader)), 1)
    except TypeError:
        return 1


def _batch_size(batch: Mapping[str, Any]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            return int(value.size(0))
    return 1


def _forward_loss(model: nn.Module, batch: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = batch["labels"]
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    outputs = model(**inputs)
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, dict):
        logits = outputs["logits"]
    else:
        logits = outputs
    vocab = logits.size(-1)
    shift_logits = logits[..., :-1, :].reshape(-1, vocab).to(torch.float32)
    shift_labels = labels[..., 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)
    return loss, logits


class FedProx(FederatedMethod):
    name = "fedprox"

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hparams = FedProxHyperparams()

    def build_model(self) -> nn.Module:
        model = load_backbone(self.backbone_spec, self.config.resolve_dtype())
        inject_moe_lora(
            model,
            self.backbone_spec.adapter_layers,
            self.config.num_experts,
            self.config.lora_rank,
            self.config.lora_alpha,
            self.config.top_k,
            self.config.routing,
        )
        return model

    def build_optimizer(self, model: nn.Module) -> ProximalAdamW:
        return ProximalAdamW(
            self.trainable_parameters(model),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            proximal_mu=self.hparams.proximal_mu,
        )

    def _snapshot_anchors(
        self, model: nn.Module, broadcast: Optional[BroadcastPackage]
    ) -> Dict[int, torch.Tensor]:
        source = broadcast.weights if broadcast is not None else extract_trainable(model)
        anchors: Dict[int, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in source:
                anchors[id(param)] = source[name].detach().to(param.device, torch.float32).clone()
        return anchors

    def _proximal_distance(self, model: nn.Module, anchors: Dict[int, torch.Tensor]) -> float:
        total = 0.0
        with torch.no_grad():
            for param in model.parameters():
                anchor = anchors.get(id(param))
                if anchor is None:
                    continue
                total += float((param.detach().to(torch.float32) - anchor).pow(2).sum())
        return total

    def local_update(
        self,
        client_id: int,
        task_id: int,
        round_id: int,
        model: nn.Module,
        loader: Iterable,
        broadcast: Optional[BroadcastPackage],
    ) -> ClientPayload:
        if broadcast is not None:
            load_weights(model, broadcast.weights)
        anchors = self._snapshot_anchors(model, broadcast)

        optimizer = self.build_optimizer(model)
        optimizer.set_anchors(anchors)
        total_steps = _steps_per_epoch(loader) * self.config.local_epochs
        scheduler = self.build_lr_scheduler(optimizer, total_steps)

        model.train()
        loss_total = 0.0
        steps = 0
        samples = 0
        for _ in range(self.config.local_epochs):
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss, _ = _forward_loss(model, batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_total += float(loss.detach())
                steps += 1
                samples += _batch_size(batch)

        stats: Dict[str, Any] = {
            "loss": loss_total / max(steps, 1),
            "proximal_mu": self.hparams.proximal_mu,
            "proximal_distance": self._proximal_distance(model, anchors),
            "local_steps": steps,
        }
        return ClientPayload(
            client_id=client_id,
            num_samples=samples,
            weights=extract_trainable(model),
            stats=stats,
        )

    def aggregate(self, payloads: List[ClientPayload], task_id: int, round_id: int) -> BroadcastPackage:
        merged = fedavg(payloads)
        return BroadcastPackage(weights=merged)
