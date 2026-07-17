from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.aggregation import extract_trainable, fedavg, load_weights
from harness.interfaces import BroadcastPackage, ClientPayload, FederatedMethod
from models.backbone import load_backbone
from models.moe_lora import inject_moe_lora, iter_adapters

from baselines.fed_keeplora.subspace import (
    HistoricalFeatureDirections,
    PrincipalResidualSplit,
    collect_feature_covariance,
)


@dataclass(frozen=True)
class FedKeepLoRAHyperparams:
    principal_energy: float = 0.5
    history_rank: int = 8
    covariance_batches: int = 32


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


class FedKeepLoRA(FederatedMethod):
    name = "fed-keeplora"
    max_grad_norm: float = 1.0

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hparams = FedKeepLoRAHyperparams()
        self._pretrained_split: Dict[str, PrincipalResidualSplit] = {}
        self._client_history: Dict[int, HistoricalFeatureDirections] = {}

    def build_model(self) -> nn.Module:
        model = load_backbone(self.backbone_spec, self.config.resolve_dtype())
        inject_moe_lora(
            model,
            adapter_layers=self.backbone_spec.adapter_layers,
            num_experts=self.config.num_experts,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            top_k=self.config.top_k,
            routing=self.config.routing,
        )
        return model

    def _client_history_for(self, client_id: int) -> HistoricalFeatureDirections:
        history = self._client_history.get(client_id)
        if history is None:
            history = HistoricalFeatureDirections(history_rank=self.hparams.history_rank)
            self._client_history[client_id] = history
        return history

    def _ensure_pretrained_split(self, model: nn.Module) -> None:
        for name, adapter in iter_adapters(model):
            if name not in self._pretrained_split:
                self._pretrained_split[name] = PrincipalResidualSplit.from_weight(
                    adapter.base.weight, self.hparams.principal_energy
                )

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
        self._ensure_pretrained_split(model)
        history = self._client_history_for(client_id)
        device = _model_device(model)
        optimizer = self.build_optimizer(model)
        total_steps = max(self.config.local_epochs * _num_batches(loader), 1)
        scheduler = self.build_lr_scheduler(optimizer, total_steps)
        model.train()
        running_loss = 0.0
        steps = 0
        projected_layers = 0
        for _ in range(self.config.local_epochs):
            for batch in loader:
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = compute_language_modeling_loss(model, batch)
                loss.backward()
                projected_layers = self._project_gradients(model, history)
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_parameters(model), self.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                running_loss += float(loss.detach())
                steps += 1
        if round_id == self.config.rounds_per_task - 1:
            self._update_history(client_id, model, loader)
        stats: Dict[str, Any] = {
            "loss": running_loss / max(steps, 1),
            "projected_layers": projected_layers,
            "principal_columns": self._principal_columns(),
            "residual_columns": self._residual_columns(),
            "principal_energy": self._mean_captured_energy(),
            "history_columns": history.total_columns(),
            "task_id": task_id,
            "round_id": round_id,
        }
        return ClientPayload(
            client_id=client_id,
            num_samples=_dataset_size(loader),
            weights=extract_trainable(model),
            stats=stats,
        )

    def _project_gradients(self, model: nn.Module, history: HistoricalFeatureDirections) -> int:
        touched = 0
        for name, adapter in iter_adapters(model):
            split = self._pretrained_split.get(name)
            if split is None:
                continue
            for expert in adapter.experts:
                grad_a = expert.lora_A.grad
                if grad_a is not None:
                    residual = split.project_input(grad_a.to(torch.float32))
                    residual = history.project_input(name, residual)
                    grad_a.copy_(residual.to(grad_a.dtype))
                grad_b = expert.lora_B.grad
                if grad_b is not None:
                    residual = split.project_output(grad_b.to(torch.float32))
                    grad_b.copy_(residual.to(grad_b.dtype))
            touched += 1
        return touched

    def _update_history(self, client_id: int, model: nn.Module, loader: Iterable) -> None:
        history = self._client_history_for(client_id)
        feature_grams = collect_feature_covariance(model, loader, self.hparams.covariance_batches)
        history.update_memory(feature_grams)

    def _principal_columns(self) -> int:
        return int(sum(split.rank for split in self._pretrained_split.values()))

    def _residual_columns(self) -> int:
        return int(sum(split.residual_input_dim for split in self._pretrained_split.values()))

    def _mean_captured_energy(self) -> float:
        if not self._pretrained_split:
            return 0.0
        total = sum(split.captured_energy for split in self._pretrained_split.values())
        return total / len(self._pretrained_split)

    def aggregate(
        self, payloads: List[ClientPayload], task_id: int, round_id: int
    ) -> BroadcastPackage:
        merged = fedavg(payloads)
        return BroadcastPackage(
            weights=merged, extras={"task_id": task_id, "round_id": round_id}
        )
