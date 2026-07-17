from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.aggregation import client_weights, extract_trainable, fedavg, load_weights
from harness.interfaces import (
    BroadcastPackage,
    ClientPayload,
    FederatedMethod,
    TensorDict,
)
from models.backbone import load_backbone

from baselines.fed_eproj.projectors import (
    VISION_HIDDEN,
    ProjectorLibrary,
    ProjectorMLP,
    TaskKeyStore,
)

LIBRARY_MODULE = "eproj_projector_library"
KEY_MODULE = "eproj_key_store"
LIBRARY_PREFIX = LIBRARY_MODULE + "."
KEY_PREFIX = KEY_MODULE + "."


@dataclass(frozen=True)
class FedEProjHyperparams:
    similarity_reg_lambda: float = 1e8
    key_pull_lambda: float = 0.1
    projector_hidden_multiplier: int = 1
    published_reference_lr: float = 1e-5
    published_reference_epochs: int = 5


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


class FedEProj(FederatedMethod):
    name = "fed-eproj"

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hparams = FedEProjHyperparams()
        self._dtype = config.resolve_dtype()
        self._vision_hidden = VISION_HIDDEN[backbone_spec.name]
        self._embed_dim = self._vision_hidden + backbone_spec.hidden_size
        self._library = ProjectorLibrary()
        self._keys = TaskKeyStore(self._embed_dim)
        self._references: Dict[int, torch.Tensor] = {}
        self._importance_max: Dict[int, float] = {}
        self._similarity: Dict[int, float] = {}
        self._param_importance: Dict[str, torch.Tensor] = {}
        self._current_centroid: Optional[torch.Tensor] = None
        self._active_task = 0
        self._active_projector: Optional[ProjectorMLP] = None
        self._active_key: Optional[nn.Parameter] = None
        self._centroid_estimate: Optional[torch.Tensor] = None

    def build_model(self) -> nn.Module:
        model = load_backbone(self.backbone_spec, self._dtype)
        for param in model.parameters():
            param.requires_grad_(False)
        model.add_module(LIBRARY_MODULE, ProjectorLibrary())
        model.add_module(KEY_MODULE, TaskKeyStore(self._embed_dim))
        return model

    def encode_task_sample(self, batch: Any) -> torch.Tensor:
        raise NotImplementedError

    def on_task_start(self, task_id: int) -> None:
        task = str(task_id)
        self._library.allocate(
            task,
            self._vision_hidden,
            self.backbone_spec.hidden_size,
            self.hparams.projector_hidden_multiplier,
            self._dtype,
        )
        self._keys.allocate(task)
        self._library.set_active(task)
        for previous in range(task_id):
            self._library.freeze_task(str(previous))
            self._keys.freeze_key(str(previous))
        self._library.set_trainable(task, True)
        self._keys.set_trainable(task, True)
        self._param_importance = {}
        self._current_centroid = None
        self._active_task = task_id

    def _prepare_model(
        self,
        model: nn.Module,
        task_id: int,
        broadcast: Optional[BroadcastPackage],
    ) -> None:
        library: ProjectorLibrary = getattr(model, LIBRARY_MODULE)
        keys: TaskKeyStore = getattr(model, KEY_MODULE)
        for previous in range(task_id + 1):
            task = str(previous)
            library.allocate(
                task,
                self._vision_hidden,
                self.backbone_spec.hidden_size,
                self.hparams.projector_hidden_multiplier,
                self._dtype,
            )
            keys.allocate(task)
        self._copy_module(library, self._library)
        self._copy_module(keys, self._keys)
        library.set_active(str(task_id))
        for previous in range(task_id + 1):
            trainable = previous == task_id
            library.set_trainable(str(previous), trainable)
            keys.set_trainable(str(previous), trainable)
        if broadcast is not None:
            load_weights(model, broadcast.weights)
        self._active_projector = library.active_module()
        self._active_key = keys.key(str(task_id))

    def _copy_module(self, destination: nn.Module, source: nn.Module) -> None:
        target = dict(destination.named_parameters())
        with torch.no_grad():
            for name, value in source.named_parameters():
                if name in target:
                    target[name].copy_(value.to(target[name].dtype))

    def _flatten_projector(self, projector: ProjectorMLP) -> torch.Tensor:
        parts = [param.reshape(-1).to(torch.float32) for _, param in sorted(projector.named_parameters())]
        return torch.cat(parts)

    def _key_pull_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        target = embeddings.detach().to(torch.float32).mean(dim=0)
        return F.mse_loss(self._active_key.to(torch.float32), target)

    def _task_similarity(self, task_id: int) -> float:
        if self._centroid_estimate is None:
            return 0.0
        stored = self._keys.centroid(str(task_id)).to(torch.float32)
        score = F.cosine_similarity(self._centroid_estimate.to(torch.float32), stored, dim=0)
        return float(score)

    def _similarity_regularizer(self) -> torch.Tensor:
        reg = torch.zeros((), dtype=torch.float32)
        if self._active_projector is None:
            return reg
        current = self._flatten_projector(self._active_projector)
        for task_id in range(self._active_task):
            reference = self._references.get(task_id)
            if reference is None:
                continue
            gate = 1.0 - self._task_similarity(task_id)
            importance = self._importance_max.get(task_id, 0.0)
            reg = reg + gate * importance * torch.sum((current - reference) ** 2)
        return reg

    def _update_importance(self, projector: ProjectorMLP) -> None:
        for name, param in projector.named_parameters():
            if param.grad is None:
                continue
            squared = param.grad.detach().to(torch.float32) ** 2
            previous = self._param_importance.get(name)
            self._param_importance[name] = squared if previous is None else torch.maximum(previous, squared)

    def local_update(
        self,
        client_id: int,
        task_id: int,
        round_id: int,
        model: nn.Module,
        loader: Iterable,
        broadcast: Optional[BroadcastPackage],
    ) -> ClientPayload:
        self._active_task = task_id
        self._prepare_model(model, task_id, broadcast)
        device = _model_device(model)
        model.train()
        optimizer = self.build_optimizer(model)
        total_steps = max(self.config.local_epochs * _num_batches(loader), 1)
        scheduler = self.build_lr_scheduler(optimizer, total_steps)
        centroid_sum = torch.zeros(self._embed_dim, dtype=torch.float32)
        seen = 0
        running_task = 0.0
        steps = 0
        for _ in range(self.config.local_epochs):
            for batch in loader:
                batch = _move_batch(batch, device)
                embeddings = self.encode_task_sample(batch)
                centroid_sum = centroid_sum + embeddings.detach().to(torch.float32).sum(dim=0)
                seen += int(embeddings.shape[0])
                running_centroid = centroid_sum / max(seen, 1)
                self._centroid_estimate = (
                    self._current_centroid if self._current_centroid is not None else running_centroid
                )
                task_loss = compute_language_modeling_loss(model, batch)
                pull = self._key_pull_loss(embeddings)
                regularizer = self._similarity_regularizer()
                total = (
                    task_loss.to(torch.float32)
                    + self.hparams.key_pull_lambda * pull
                    + self.hparams.similarity_reg_lambda * regularizer
                )
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                self._update_importance(self._active_projector)
                optimizer.step()
                scheduler.step()
                running_task += float(task_loss.detach())
                steps += 1
        local_centroid = centroid_sum / max(seen, 1)
        r_max = max((float(value.max()) for value in self._param_importance.values()), default=0.0)
        stats: Dict[str, Any] = {
            "task_id": task_id,
            "round_id": round_id,
            "task_loss": running_task / max(steps, 1),
            "local_centroid": local_centroid,
            "r_max": r_max,
        }
        return ClientPayload(
            client_id=client_id,
            num_samples=_dataset_size(loader),
            weights=extract_trainable(model),
            stats=stats,
        )

    def aggregate(
        self,
        payloads: List[ClientPayload],
        task_id: int,
        round_id: int,
    ) -> BroadcastPackage:
        names = list(payloads[0].weights.keys())
        merged = fedavg(payloads, keys=names)
        self._writeback(merged)
        mix = client_weights(payloads)
        centroid = torch.zeros(self._embed_dim, dtype=torch.float32)
        for weight, payload in zip(mix, payloads):
            local = payload.stats.get("local_centroid")
            if local is not None:
                centroid = centroid + weight * local.to(torch.float32)
        self._current_centroid = centroid
        self._keys.set_centroid(str(task_id), centroid)
        for previous in range(task_id):
            stored = self._keys.centroid(str(previous)).to(torch.float32)
            self._similarity[previous] = float(F.cosine_similarity(centroid, stored, dim=0))
        extras: Dict[str, Any] = {
            "task_id": task_id,
            "round_id": round_id,
            "similarity": dict(self._similarity),
            "centroid": centroid,
            "library": self._full_state_weights(),
        }
        return BroadcastPackage(weights=merged, extras=extras)

    def _writeback(self, weights: TensorDict) -> None:
        library = dict(self._library.named_parameters())
        keys = dict(self._keys.named_parameters())
        with torch.no_grad():
            for name, value in weights.items():
                if name.startswith(LIBRARY_PREFIX):
                    target = library.get(name[len(LIBRARY_PREFIX):])
                elif name.startswith(KEY_PREFIX):
                    target = keys.get(name[len(KEY_PREFIX):])
                else:
                    target = None
                if target is not None:
                    target.copy_(value.to(target.dtype))

    def _full_state_weights(self) -> TensorDict:
        state: TensorDict = {}
        for name, param in self._library.named_parameters():
            state[LIBRARY_PREFIX + name] = param.detach().clone()
        for name, param in self._keys.named_parameters():
            state[KEY_PREFIX + name] = param.detach().clone()
        return state

    def on_task_end(self, task_id: int, payloads: List[ClientPayload]) -> None:
        task = str(task_id)
        self._library.freeze_task(task)
        self._keys.freeze_key(task)
        if self._current_centroid is not None:
            self._keys.set_centroid(task, self._current_centroid)
        self._references[task_id] = self._flatten_projector(self._library.module(task)).detach()
        tracked = max((float(value.max()) for value in self._param_importance.values()), default=0.0)
        reported = max((float(payload.stats.get("r_max", 0.0)) for payload in payloads), default=0.0)
        self._importance_max[task_id] = max(tracked, reported)
        self.set_broadcast(
            BroadcastPackage(
                weights=self._full_state_weights(),
                extras={"task_id": task_id, "library_broadcast": True},
            )
        )
