from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.aggregation import client_weights, extract_trainable, fedavg, load_weights
from harness.interfaces import BroadcastPackage, ClientPayload, FederatedMethod
from models.backbone import load_backbone
from models.moe_lora import inject_moe_lora

from baselines.fed_moelora.routing import (
    RoutingMeter,
    aggregate_expert_mass,
    expert_load_imbalance,
)


@dataclass(frozen=True)
class FedMoELoRAHyperparams:
    expert_num: int = 4
    total_rank: int = 32
    gate_mode: str = "dense"
    published_reference_rank: int = 128
    published_reference_alpha: int = 256
    published_reference_experts: int = 8


def _model_device(model: nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _move_batch(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def _num_batches(loader: Iterable) -> int:
    try:
        return int(len(loader))
    except TypeError:
        return 1


def _dataset_size(loader: Iterable) -> int:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        try:
            return int(len(dataset))
        except TypeError:
            pass
    return _num_batches(loader)


def compute_language_modeling_loss(model: nn.Module, batch: Any) -> torch.Tensor:
    labels = batch["labels"]
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    outputs = model(**inputs)
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, dict):
        logits = outputs["logits"]
    else:
        logits = outputs
    shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


class FedMoELoRA(FederatedMethod):
    name = "fed-moelora"

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hparams = FedMoELoRAHyperparams()

    def build_model(self) -> nn.Module:
        hparams = self.hparams
        model = load_backbone(self.backbone_spec, self.config.resolve_dtype())
        inject_moe_lora(
            model,
            self.backbone_spec.adapter_layers,
            num_experts=hparams.expert_num,
            rank=hparams.total_rank // hparams.expert_num,
            alpha=self.config.lora_alpha,
            top_k=self.config.top_k,
            routing=hparams.gate_mode,
        )
        return model

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
        device = _model_device(model)
        optimizer = self.build_optimizer(model)
        total_steps = max(self.config.local_epochs * _num_batches(loader), 1)
        scheduler = self.build_lr_scheduler(optimizer, total_steps)
        meter = RoutingMeter(self.hparams.expert_num)
        model.train()
        running_loss = 0.0
        steps = 0
        for _ in range(self.config.local_epochs):
            for batch in loader:
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = compute_language_modeling_loss(model, batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                meter.update(model)
                running_loss += float(loss.detach())
                steps += 1
        telemetry = meter.result()
        stats: Dict[str, Any] = {
            "loss": running_loss / max(steps, 1),
            "local_steps": steps,
            "gate_mode": self.hparams.gate_mode,
            "expert_num": self.hparams.expert_num,
            "expert_rank": self.hparams.total_rank // self.hparams.expert_num,
            "task_id": task_id,
            "round_id": round_id,
        }
        stats.update(telemetry.as_dict())
        return ClientPayload(
            client_id=client_id,
            num_samples=_dataset_size(loader),
            weights=extract_trainable(model),
            stats=stats,
        )

    def aggregate(
        self, payloads: List[ClientPayload], task_id: int, round_id: int
    ) -> BroadcastPackage:
        merged = fedavg(payloads)
        mix = client_weights(payloads)
        masses = [list(payload.stats.get("routing_mass", [])) for payload in payloads]
        routing_mass = aggregate_expert_mass(masses, mix, self.hparams.expert_num)
        extras: Dict[str, Any] = {
            "task_id": task_id,
            "round_id": round_id,
            "gate_mode": self.hparams.gate_mode,
            "routing_mass": routing_mass,
            "load_imbalance": expert_load_imbalance(routing_mass, self.hparams.expert_num),
        }
        return BroadcastPackage(weights=merged, extras=extras)
