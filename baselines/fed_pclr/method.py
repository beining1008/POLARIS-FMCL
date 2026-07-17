from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.aggregation import client_weights, extract_trainable, fedavg, load_weights
from harness.interfaces import BroadcastPackage, ClientPayload, FederatedMethod
from models.backbone import load_backbone
from models.moe_lora import TARGET_MODULES

from baselines.fed_pclr.rank_pool import (
    RankPoolState,
    inject_rank_pool_lora,
    iter_rank_pool,
)


@dataclass(frozen=True)
class FedPCLRHyperparams:
    train_rank: int = 32
    top_rank: int = 32
    loss_weight3: float = 0.2
    published_reference_train_rank: int = 64
    published_reference_top_rank: int = 64


def _device(model: nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _first_tensor(batch: Any) -> Optional[torch.Tensor]:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        values: Iterable[Any] = batch.values()
    elif isinstance(batch, (list, tuple)):
        values = batch
    else:
        return None
    for value in values:
        if isinstance(value, torch.Tensor):
            return value
    return None


def _batch_size(batch: Any, default: int) -> int:
    tensor = _first_tensor(batch)
    if tensor is not None and tensor.dim() > 0:
        return int(tensor.shape[0])
    return default


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(v.to(device) if isinstance(v, torch.Tensor) else v for v in batch)
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def _loader_length(loader: Iterable) -> int:
    try:
        return max(int(len(loader)), 1)
    except TypeError:
        return 1


def _forward_loss(model: nn.Module, batch: Any) -> torch.Tensor:
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


class FedPCLR(FederatedMethod):
    name = "fed-pclr"

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hp = FedPCLRHyperparams()
        self._scaling = float(config.lora_alpha) / float(max(config.lora_rank, 1))
        self._pool_rank = self.hp.top_rank + 2 * self.hp.train_rank
        self._states: Dict[str, RankPoolState] = {}

    def build_model(self) -> nn.Module:
        dtype = self.config.resolve_dtype()
        backbone = load_backbone(self.backbone_spec, dtype)
        wrapped = inject_rank_pool_lora(
            backbone,
            self.backbone_spec.adapter_layers,
            self._pool_rank,
            self._scaling,
            TARGET_MODULES,
        )
        if not self._states:
            for name in wrapped:
                self._states[name] = RankPoolState.initial(self.hp.top_rank, self.hp.train_rank)
        self._sync_masks(backbone)
        return backbone

    def _sync_masks(self, model: nn.Module) -> None:
        for name, module in iter_rank_pool(model):
            state = self._states.get(name)
            if state is None:
                continue
            module.set_trainable(state.trainable_ids)

    def _pool_occupancy(self) -> Dict[str, int]:
        if not self._states:
            return {"shared": 0, "static": 0, "trainable": 0}
        return next(iter(self._states.values())).occupancy()

    def _interference_regularizer(self, model: nn.Module) -> Optional[torch.Tensor]:
        terms: List[torch.Tensor] = []
        for name, module in iter_rank_pool(model):
            state = self._states.get(name)
            if state is None:
                continue
            frozen = state.frozen_ids
            trainable = state.trainable_ids
            if not frozen or not trainable:
                continue
            device = module.lora_A_pool.device
            fro = torch.tensor(frozen, dtype=torch.long, device=device)
            tr = torch.tensor(trainable, dtype=torch.long, device=device)
            a_pool = module.lora_A_pool.to(torch.float32)
            b_pool = module.lora_B_pool.to(torch.float32)
            a_train = a_pool.index_select(0, tr)
            a_frozen = a_pool.index_select(0, fro)
            b_train = b_pool.index_select(1, tr).t()
            b_frozen = b_pool.index_select(1, fro).t()
            cross_a = a_train @ a_frozen.t()
            cross_b = b_train @ b_frozen.t()
            terms.append(cross_a.pow(2).mean() + cross_b.pow(2).mean())
        if not terms:
            return None
        return self.hp.loss_weight3 * torch.stack(terms).mean()

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
        self._sync_masks(model)
        device = _device(model)
        model.train()
        optimizer = self.build_optimizer(model)
        total_steps = max(self.config.local_epochs * _loader_length(loader), 1)
        scheduler = self.build_lr_scheduler(optimizer, total_steps)

        seen = 0
        batches = 0
        loss_sum = 0.0
        reg_sum = 0.0
        for epoch in range(self.config.local_epochs):
            for batch in loader:
                batch = _to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                task_loss = _forward_loss(model, batch)
                reg = self._interference_regularizer(model)
                loss = task_loss if reg is None else task_loss + reg
                loss.backward()
                nn.utils.clip_grad_norm_(self.trainable_parameters(model), 1.0)
                optimizer.step()
                scheduler.step()
                batches += 1
                loss_sum += float(task_loss.detach())
                reg_sum += 0.0 if reg is None else float(reg.detach())
                if epoch == 0:
                    seen += _batch_size(batch, self.config.batch_size)

        occupancy = self._pool_occupancy()
        stats: Dict[str, Any] = {
            "loss": loss_sum / max(batches, 1),
            "interference": reg_sum / max(batches, 1),
            "shared": occupancy["shared"],
            "static": occupancy["static"],
            "trainable": occupancy["trainable"],
            "pool_rank": self._pool_rank,
        }
        return ClientPayload(
            client_id=client_id,
            num_samples=max(seen, 1),
            weights=extract_trainable(model),
            stats=stats,
        )

    def _aggregate_pool(
        self,
        payloads: List[ClientPayload],
        key: str,
        ids: List[int],
        axis: int,
    ) -> torch.Tensor:
        reference = payloads[0].weights[key]
        if not ids:
            return reference.detach().clone()
        mix = client_weights(payloads)
        accumulator = reference.detach().to(torch.float32).clone()
        index = torch.tensor(sorted(ids), dtype=torch.long, device=accumulator.device)
        accumulator.index_fill_(axis, index, 0.0)
        for weight, payload in zip(mix, payloads):
            selected = payload.weights[key].index_select(axis, index).to(torch.float32)
            accumulator.index_add_(axis, index, selected * weight)
        return accumulator.to(reference.dtype)

    def aggregate(self, payloads: List[ClientPayload], task_id: int, round_id: int) -> BroadcastPackage:
        reference = payloads[0].weights
        merged: Dict[str, torch.Tensor] = {}
        for key in reference.keys():
            if key.endswith(".lora_A_pool") or key.endswith(".lora_B_pool"):
                name = key.rsplit(".", 1)[0]
                state = self._states.get(name)
                trainable = state.trainable_ids if state is not None else []
                axis = 0 if key.endswith(".lora_A_pool") else 1
                merged[key] = self._aggregate_pool(payloads, key, trainable, axis)
            else:
                merged[key] = fedavg(payloads, keys=[key])[key]
        extras = {"partition": {n: dict(s.occupancy()) for n, s in self._states.items()}}
        return BroadcastPackage(weights=merged, extras=extras)

    def _compression_stage(self, weights: Dict[str, torch.Tensor], task_id: int) -> None:
        for ordinal, (name, state) in enumerate(self._states.items()):
            a_key = name + ".lora_A_pool"
            b_key = name + ".lora_B_pool"
            if a_key not in weights or b_key not in weights:
                continue
            dtype = weights[a_key].dtype
            a_pool = weights[a_key].to(torch.float32)
            b_pool = weights[b_key].to(torch.float32)
            device = a_pool.device
            in_features = a_pool.shape[1]
            out_features = b_pool.shape[0]

            group = state.shared_ids + state.trainable_ids
            index = torch.tensor(group, dtype=torch.long, device=device)
            a_group = a_pool.index_select(0, index)
            b_group = b_pool.index_select(1, index)
            delta = b_group @ a_group
            left, singular, right = torch.linalg.svd(delta, full_matrices=False)
            root = singular.clamp_min(0.0).sqrt()

            top = min(self.hp.top_rank, singular.shape[0])
            static = min(self.hp.train_rank, max(singular.shape[0] - top, 0))
            new_a = torch.zeros(state.pool_rank, in_features, dtype=torch.float32, device=device)
            new_b = torch.zeros(out_features, state.pool_rank, dtype=torch.float32, device=device)
            if top > 0:
                new_a[0:top] = root[:top].unsqueeze(1) * right[:top]
                new_b[:, 0:top] = left[:, :top] * root[:top].unsqueeze(0)
            if static > 0:
                base_static = self.hp.top_rank
                lo, hi = top, top + static
                new_a[base_static:base_static + static] = root[lo:hi].unsqueeze(1) * right[lo:hi]
                new_b[:, base_static:base_static + static] = left[:, lo:hi] * root[lo:hi].unsqueeze(0)

            base_train = self.hp.top_rank + self.hp.train_rank
            generator = torch.Generator().manual_seed(self.config.seed * 100003 + task_id * 97 + ordinal)
            bound = 1.0 / math.sqrt(max(in_features, 1))
            fresh = (torch.rand(self.hp.train_rank, in_features, generator=generator) * 2.0 - 1.0) * bound
            new_a[base_train:base_train + self.hp.train_rank] = fresh.to(device)

            weights[a_key] = new_a.to(dtype)
            weights[b_key] = new_b.to(dtype)
            state.repartition((top, static, self.hp.train_rank))

    def on_task_end(self, task_id: int, payloads: List[ClientPayload]) -> None:
        if self._broadcast is None:
            return
        self._compression_stage(self._broadcast.weights, task_id)
        self._broadcast.extras["partition"] = {
            name: dict(state.occupancy()) for name, state in self._states.items()
        }
