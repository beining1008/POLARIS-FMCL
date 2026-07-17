from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.fed_olora.olora_layers import (
    _fresh_pair,
    _grow_stack,
    inject_olora_moe,
    iter_olora,
)
from harness.aggregation import extract_trainable, fedavg, load_weights
from harness.interfaces import BroadcastPackage, ClientPayload, FederatedMethod, TensorDict
from models.backbone import load_backbone
from models.moe_lora import TARGET_MODULES

_LORANEW_A = ".loranew_A"
_LORANEW_B = ".loranew_B"


@dataclass(frozen=True)
class FedOLoRAHyperparams:
    lamda_1: float = 0.5
    lamda_2: float = 0.0


class FedOLoRA(FederatedMethod):
    name = "fed-olora"

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hparams = FedOLoRAHyperparams()
        self._dtype = config.resolve_dtype()
        self._rank = int(config.lora_rank)
        self._stacks: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def build_model(self) -> nn.Module:
        model = load_backbone(self.backbone_spec, self._dtype)
        inject_olora_moe(
            model,
            adapter_layers=self.backbone_spec.adapter_layers,
            num_experts=self.config.num_experts,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            top_k=self.config.top_k,
            routing=self.config.routing,
            target_modules=TARGET_MODULES,
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
            stacks = broadcast.extras.get("stacks") if broadcast.extras else None
            if stacks:
                self._load_stacks(model, stacks)
        model.train()
        optimizer = self.build_optimizer(model)
        scheduler = self.build_lr_scheduler(optimizer, self._total_steps(loader))
        device = next(model.parameters()).device
        seen = 0
        steps = 0
        last_task = 0.0
        last_reg = 0.0
        last_orth = 0.0
        for _ in range(self.config.local_epochs):
            for batch in loader:
                inputs = self._to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                task_loss = self._task_loss(model, inputs)
                reg, orth, _l2 = self._regularizer(model)
                loss = task_loss + reg
                loss.backward()
                optimizer.step()
                scheduler.step()
                seen += self._rows(inputs)
                steps += 1
                last_task = float(task_loss.detach())
                last_reg = float(reg.detach())
                last_orth = orth
        stats: Dict[str, Any] = {
            "task_loss": last_task,
            "regularizer": last_reg,
            "orthogonality": last_orth,
            "stack_rank": self._stack_rank(model),
            "steps": steps,
        }
        return ClientPayload(
            client_id=client_id,
            num_samples=max(seen, 1),
            weights=extract_trainable(model),
            stats=stats,
        )

    def aggregate(self, payloads: List[ClientPayload], task_id: int, round_id: int) -> BroadcastPackage:
        merged = fedavg(payloads)
        return BroadcastPackage(weights=merged, extras={"stacks": dict(self._stacks)})

    def on_task_end(self, task_id: int, payloads: List[ClientPayload]) -> None:
        merged = fedavg(payloads)
        reinit = self._fold_all(merged)
        weights: TensorDict = {name: reinit.get(name, value) for name, value in merged.items()}
        self.set_broadcast(BroadcastPackage(weights=weights, extras={"stacks": dict(self._stacks)}))

    def _fold_all(self, weights: TensorDict) -> TensorDict:
        reinit: TensorDict = {}
        prefixes = sorted({key[: -len(_LORANEW_A)] for key in weights if key.endswith(_LORANEW_A)})
        for prefix in prefixes:
            new_a = weights[prefix + _LORANEW_A].to(torch.float32)
            new_b = weights[prefix + _LORANEW_B].to(torch.float32)
            old_a, old_b = self._stacks.get(prefix, (None, None))
            self._stacks[prefix] = _grow_stack(old_a, old_b, new_a, new_b)
            fresh_a, fresh_b = _fresh_pair(new_a.shape[1], new_b.shape[0], self._rank, self._dtype, new_a.device)
            reinit[prefix + _LORANEW_A] = fresh_a
            reinit[prefix + _LORANEW_B] = fresh_b
        return reinit

    def _regularizer(self, model: nn.Module) -> Tuple[torch.Tensor, float, float]:
        orth_terms: List[torch.Tensor] = []
        l2_terms: List[torch.Tensor] = []
        for _, adapter in iter_olora(model):
            for expert in adapter.experts:
                a_new = expert.loranew_A.to(torch.float32)
                b_new = expert.loranew_B.to(torch.float32)
                if expert.stack_rank > 0:
                    a_stack = expert.lora_A_stack.to(torch.float32)
                    orth_terms.append(torch.mm(a_stack, a_new.t()).abs().sum())
                l2_terms.append(a_new.norm(p=2) + b_new.norm(p=2))
        device = next(model.parameters()).device
        zero = torch.zeros((), dtype=torch.float32, device=device)
        orth = torch.stack(orth_terms).sum() if orth_terms else zero
        l2 = torch.stack(l2_terms).sum() if l2_terms else zero
        reg = self.hparams.lamda_1 * orth + self.hparams.lamda_2 * l2
        return reg, float(orth.detach()), float(l2.detach())

    def _task_loss(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        outputs = model(**inputs)
        logits = self._logits(outputs)
        labels = inputs["labels"]
        shift_logits = logits[..., :-1, :].contiguous().to(torch.float32)
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def _load_stacks(self, model: nn.Module, stacks: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        for name, adapter in iter_olora(model):
            for index, expert in enumerate(adapter.experts):
                pair = stacks.get(f"{name}.experts.{index}")
                if pair is not None:
                    expert.set_stack(pair[0], pair[1])

    def _total_steps(self, loader: Iterable) -> int:
        try:
            steps_per_epoch = len(loader)
        except TypeError:
            steps_per_epoch = 1
        return max(self.config.local_epochs * steps_per_epoch, 1)

    @staticmethod
    def _stack_rank(model: nn.Module) -> int:
        for _, adapter in iter_olora(model):
            return int(adapter.experts[0].stack_rank)
        return 0

    @staticmethod
    def _logits(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, dict):
            return outputs["logits"]
        return outputs.logits

    @staticmethod
    def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return moved

    @staticmethod
    def _rows(inputs: Dict[str, Any]) -> int:
        for value in inputs.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.shape[0])
        return 1
