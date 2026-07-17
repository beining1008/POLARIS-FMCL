from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.fed_ewc.consolidation import (
    accumulate_fisher,
    consolidation_penalty,
    diagonal_fisher,
    fisher_trace,
)
from harness.aggregation import extract_trainable, fedavg, load_weights
from harness.interfaces import (
    BroadcastPackage,
    ClientPayload,
    FederatedMethod,
    TensorDict,
)
from models.backbone import load_backbone
from models.moe_lora import inject_moe_lora

IGNORE_INDEX = -100


@dataclass(frozen=True)
class FedEWCHyperparams:
    ewc_lambda: float = 400.0
    fisher_batches: int = 64
    fisher_accumulate: bool = True


def _device_of(model: nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        for key in ("input_ids", "labels", "attention_mask"):
            value = batch.get(key)
            if torch.is_tensor(value):
                return int(value.shape[0])
        for value in batch.values():
            if torch.is_tensor(value):
                return int(value.shape[0])
        return 0
    if torch.is_tensor(batch):
        return int(batch.shape[0])
    return len(batch)


class FedEWC(FederatedMethod):
    name = "fed-ewc"

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hparams = FedEWCHyperparams()
        self._client_fisher: Dict[int, TensorDict] = {}
        self._anchor: Optional[TensorDict] = None
        self._fisher_tasks_seen = 0
        self._active_client = -1

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

    def on_task_start(self, task_id: int) -> None:
        broadcast = self.current_broadcast()
        if broadcast is None:
            self._anchor = None
            return
        self._anchor = {
            name: value.detach().clone().to(torch.float32)
            for name, value in broadcast.weights.items()
        }

    def on_task_end(self, task_id: int, payloads: List[ClientPayload]) -> None:
        self._fisher_tasks_seen += 1

    def _prepare_inputs(self, model: nn.Module, batch: Any) -> Dict[str, Any]:
        device = _device_of(model)
        dtype = self.config.resolve_dtype()
        prepared: Dict[str, Any] = {}
        for key, value in batch.items():
            if not torch.is_tensor(value):
                prepared[key] = value
            elif torch.is_floating_point(value):
                prepared[key] = value.to(device=device, dtype=dtype)
            else:
                prepared[key] = value.to(device=device)
        return prepared

    def _forward_loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        inputs = self._prepare_inputs(model, batch)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        shift_logits = logits[..., :-1, :].contiguous().to(torch.float32)
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

    def _consolidation_penalty(self, model: nn.Module) -> torch.Tensor:
        return consolidation_penalty(
            model,
            self._anchor,
            self._client_fisher.get(self._active_client),
            self.hparams.ewc_lambda,
            _device_of(model),
        )

    def _estimate_diagonal_fisher(
        self,
        model: nn.Module,
        loader: Iterable,
        max_batches: int,
    ) -> TensorDict:
        return diagonal_fisher(model, loader, self._forward_loss, max_batches)

    def _total_steps(self, loader: Iterable) -> int:
        try:
            return max(self.config.local_epochs * len(loader), 1)
        except TypeError:
            return 1

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
        self._active_client = client_id
        model.train()
        optimizer = self.build_optimizer(model)
        scheduler = self.build_lr_scheduler(optimizer, self._total_steps(loader))
        running_task = 0.0
        running_penalty = 0.0
        steps = 0
        num_samples = 0
        for epoch in range(self.config.local_epochs):
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                task_loss = self._forward_loss(model, batch)
                penalty = self._consolidation_penalty(model)
                loss = task_loss + penalty
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_task += float(task_loss.detach())
                running_penalty += float(penalty.detach())
                steps += 1
                if epoch == 0:
                    num_samples += _batch_size(batch)
        if round_id == self.config.rounds_per_task - 1:
            estimate = self._estimate_diagonal_fisher(
                model, loader, self.hparams.fisher_batches
            )
            self._client_fisher[client_id] = accumulate_fisher(
                self._client_fisher.get(client_id),
                estimate,
                self.hparams.fisher_accumulate,
            )
        stats = {
            "task_loss": running_task / max(steps, 1),
            "consolidation_penalty": running_penalty / max(steps, 1),
            "fisher_tasks_seen": self._fisher_tasks_seen,
            "fisher_trace": fisher_trace(self._client_fisher.get(client_id)),
            "has_anchor": self._anchor is not None,
        }
        return ClientPayload(
            client_id=client_id,
            num_samples=num_samples,
            weights=extract_trainable(model),
            stats=stats,
        )

    def aggregate(
        self,
        payloads: List[ClientPayload],
        task_id: int,
        round_id: int,
    ) -> BroadcastPackage:
        merged = fedavg(payloads)
        package = BroadcastPackage(
            weights=merged,
            extras={"method": self.name, "task_id": task_id, "round_id": round_id},
        )
        self.set_broadcast(package)
        return package
