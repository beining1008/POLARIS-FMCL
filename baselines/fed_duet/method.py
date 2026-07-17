from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.fed_duet.pathway import DuetPathwayLinear
from harness.aggregation import extract_trainable, fedavg, load_weights
from harness.interfaces import BroadcastPackage, ClientPayload, FederatedMethod, TensorDict
from models.backbone import load_backbone
from models.moe_lora import TARGET_MODULES

IGNORE_INDEX = -100

_SHARED_MARKERS: Tuple[str, ...] = (".shared_experts.", ".gate.", ".prompts")


@dataclass(frozen=True)
class FedDuetHyperparams:
    num_shared_experts: int = 2
    num_local_experts: int = 1
    num_semantic_prompts: int = 8
    expert_rank: int = 8
    fusion_temperature: float = 1.0


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


def _layer_index(name: str) -> int:
    parts = name.split(".")
    for left, right in zip(parts, parts[1:]):
        if left in ("layers", "h", "blocks") and right.isdigit():
            return int(right)
    return -1


def _locate(model: nn.Module, qualified: str) -> Tuple[nn.Module, str]:
    parts = qualified.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _forward_loss(model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
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
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)


class FedDuet(FederatedMethod):
    name = "fed-duet"

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hparams = FedDuetHyperparams()
        self._client_parametric: Dict[int, TensorDict] = {}

    def build_model(self) -> nn.Module:
        model = load_backbone(self.backbone_spec, self.config.resolve_dtype())
        self._inject(model)
        return model

    def _inject(self, model: nn.Module) -> Dict[str, DuetPathwayLinear]:
        hidden_size = self.backbone_spec.hidden_size
        adapter_layers = self.backbone_spec.adapter_layers
        targets: List[Tuple[str, nn.Linear, int]] = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            leaf = name.split(".")[-1]
            index = _layer_index(name)
            if leaf in TARGET_MODULES and index in adapter_layers:
                targets.append((name, module, index))
        wrapped: Dict[str, DuetPathwayLinear] = {}
        for name, module, index in targets:
            layer = DuetPathwayLinear(
                base=module,
                hidden_size=hidden_size,
                num_shared=self.hparams.num_shared_experts,
                num_local=self.hparams.num_local_experts,
                num_prompts=self.hparams.num_semantic_prompts,
                rank=self.hparams.expert_rank,
                alpha=self.config.lora_alpha,
                fusion_temperature=self.hparams.fusion_temperature,
                layer_index=index,
            )
            parent, leaf = _locate(model, name)
            setattr(parent, leaf, layer)
            wrapped[name] = layer
        return wrapped

    def _split_shared_local(self, weights: TensorDict) -> Tuple[TensorDict, TensorDict]:
        shared: TensorDict = {}
        local: TensorDict = {}
        for name, tensor in weights.items():
            if any(marker in name for marker in _SHARED_MARKERS):
                shared[name] = tensor
            else:
                local[name] = tensor
        return shared, local

    def _swap_in(self, model: nn.Module, client_id: int) -> None:
        stored = self._client_parametric.get(client_id)
        if stored is None:
            _, local = self._split_shared_local(extract_trainable(model))
            self._client_parametric[client_id] = local
        else:
            load_weights(model, stored)

    def _swap_out(self, model: nn.Module, client_id: int) -> Tuple[TensorDict, TensorDict]:
        shared, local = self._split_shared_local(extract_trainable(model))
        self._client_parametric[client_id] = local
        return shared, local

    def _collect_mixture(self, model: nn.Module) -> List[float]:
        masses = [
            module.last_mixture
            for module in model.modules()
            if isinstance(module, DuetPathwayLinear) and module.last_mixture is not None
        ]
        if not masses:
            return [0.0] * (self.hparams.num_shared_experts + self.hparams.num_local_experts)
        averaged = torch.stack(masses, dim=0).mean(dim=0)
        return [float(value) for value in averaged]

    def _collect_fusion(self, model: nn.Module) -> List[float]:
        weights = [
            module.last_fusion
            for module in model.modules()
            if isinstance(module, DuetPathwayLinear) and module.last_fusion is not None
        ]
        if not weights:
            return [0.0, 0.0]
        averaged = torch.stack(weights, dim=0).mean(dim=0)
        return [float(value) for value in averaged]

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
        self._swap_in(model, client_id)
        device = _model_device(model)
        optimizer = self.build_optimizer(model)
        total_steps = max(self.config.local_epochs * _num_batches(loader), 1)
        scheduler = self.build_lr_scheduler(optimizer, total_steps)
        model.train()
        running_loss = 0.0
        steps = 0
        for _ in range(self.config.local_epochs):
            for batch in loader:
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = _forward_loss(model, batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += float(loss.detach())
                steps += 1
        shared, local = self._swap_out(model, client_id)
        stats: Dict[str, Any] = {
            "loss": running_loss / max(steps, 1),
            "local_steps": steps,
            "task_id": task_id,
            "round_id": round_id,
            "shared_params": int(sum(tensor.numel() for tensor in shared.values())),
            "local_params": int(sum(tensor.numel() for tensor in local.values())),
            "semantic_mixture": self._collect_mixture(model),
            "pathway_fusion": self._collect_fusion(model),
        }
        return ClientPayload(
            client_id=client_id,
            num_samples=_dataset_size(loader),
            weights=shared,
            stats=stats,
        )

    def aggregate(self, payloads: List[ClientPayload], task_id: int, round_id: int) -> BroadcastPackage:
        merged = fedavg(payloads)
        extras: Dict[str, Any] = {
            "task_id": task_id,
            "round_id": round_id,
            "num_shared_experts": self.hparams.num_shared_experts,
            "num_local_experts": self.hparams.num_local_experts,
        }
        return BroadcastPackage(weights=merged, extras=extras)
