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

from baselines.fed_gpm.gpm import GradientProjectionMemory, get_representation_matrix


@dataclass(frozen=True)
class FedGPMHyperparams:
    energy_threshold_base: float = 0.97
    energy_threshold_increment: float = 0.003
    representation_samples: int = 125


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


class FedGPM(FederatedMethod):
    name = "fed-gpm"
    max_grad_norm: float = 1.0

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hparams = FedGPMHyperparams()
        self._client_memories: Dict[int, GradientProjectionMemory] = {}

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

    def _client_memory(self, client_id: int) -> GradientProjectionMemory:
        memory = self._client_memories.get(client_id)
        if memory is None:
            memory = GradientProjectionMemory(
                energy_threshold_base=self.hparams.energy_threshold_base,
                energy_threshold_increment=self.hparams.energy_threshold_increment,
            )
            self._client_memories[client_id] = memory
        return memory

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
        memory = self._client_memory(client_id)
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
                projected_layers = self._project_gradients(model, memory)
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_parameters(model), self.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                running_loss += float(loss.detach())
                steps += 1
        if round_id == self.config.rounds_per_task - 1:
            self._update_gpm(client_id, task_id, model, loader)
        stats: Dict[str, Any] = {
            "loss": running_loss / max(steps, 1),
            "projected_layers": projected_layers,
            "memory_columns": memory.total_columns(),
            "task_id": task_id,
            "round_id": round_id,
        }
        return ClientPayload(
            client_id=client_id,
            num_samples=_dataset_size(loader),
            weights=extract_trainable(model),
            stats=stats,
        )

    def _project_gradients(self, model: nn.Module, memory: GradientProjectionMemory) -> int:
        touched = 0
        for name, adapter in iter_adapters(model):
            if not memory.has_basis(name):
                continue
            for expert in adapter.experts:
                grad = expert.lora_A.grad
                if grad is not None:
                    grad.copy_(memory.project(name, grad))
            touched += 1
        return touched

    def _update_gpm(
        self, client_id: int, task_id: int, model: nn.Module, loader: Iterable
    ) -> None:
        memory = self._client_memory(client_id)
        representations = get_representation_matrix(
            model, loader, self.hparams.representation_samples
        )
        memory.update_memory(representations, task_id)

    def aggregate(
        self, payloads: List[ClientPayload], task_id: int, round_id: int
    ) -> BroadcastPackage:
        merged = fedavg(payloads)
        return BroadcastPackage(
            weights=merged, extras={"task_id": task_id, "round_id": round_id}
        )
