from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.aggregation import extract_trainable, fedavg, load_weights
from harness.interfaces import (
    BroadcastPackage,
    ClientPayload,
    FederatedMethod,
    TensorDict,
)
from models.backbone import load_backbone
from models.moe_lora import inject_moe_lora

from baselines.fed_lwf.teacher import TeacherContext, extract_logits

IGNORE_INDEX = -100
EPS = 1e-8


@dataclass(frozen=True)
class FedLwFHyperparams:
    temperature: float = 2.0
    lambda_o: float = 1.0


def _batch_size(batch: Mapping[str, Any]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            return int(value.size(0))
    return 1


class FedLwF(FederatedMethod):
    name = "fed-lwf"

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hparams = FedLwFHyperparams()
        self._teacher_weights: Optional[TensorDict] = None

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
        if task_id == 0:
            self._teacher_weights = None
            return
        broadcast = self.current_broadcast()
        if broadcast is None:
            self._teacher_weights = None
            return
        self._teacher_weights = {
            name: value.detach().clone() for name, value in broadcast.weights.items()
        }

    def _total_steps(self, loader: Iterable) -> int:
        try:
            return max(self.config.local_epochs * len(loader), 1)
        except TypeError:
            return 1

    @staticmethod
    def _to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = value.to(device) if torch.is_tensor(value) else value
        return moved

    def _forward_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch["labels"]
        inputs = {key: value for key, value in batch.items() if key != "labels"}
        logits = extract_logits(model(**inputs))
        shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    def _distillation_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch["labels"]
        inputs = {key: value for key, value in batch.items() if key != "labels"}
        with TeacherContext(model, self._teacher_weights) as teacher:
            teacher_logits = teacher.logits(inputs)
        student_logits = extract_logits(model(**inputs))
        shift_student = student_logits[:, :-1, :]
        shift_teacher = teacher_logits[:, :-1, :]
        answer_mask = labels[:, 1:].ne(IGNORE_INDEX)
        return self._temperature_scaled_ce(shift_student, shift_teacher, answer_mask)

    def _temperature_scaled_ce(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        temperature = float(self.hparams.temperature)
        inverse = 1.0 / temperature
        student = student_logits.to(torch.float32)
        teacher = teacher_logits.to(torch.float32)
        teacher_prob = F.softmax(teacher, dim=-1)
        student_prob = F.softmax(student, dim=-1)
        teacher_power = teacher_prob.clamp_min(EPS).pow(inverse)
        teacher_soft = teacher_power / teacher_power.sum(dim=-1, keepdim=True)
        student_power = student_prob.clamp_min(EPS).pow(inverse)
        student_soft = student_power / student_power.sum(dim=-1, keepdim=True)
        per_token = -(teacher_soft * torch.log(student_soft.clamp_min(EPS))).sum(dim=-1)
        selection = mask.to(torch.float32)
        denominator = selection.sum().clamp_min(1.0)
        return (per_token * selection).sum() / denominator

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
        model.train()
        device = next(model.parameters()).device
        optimizer = self.build_optimizer(model)
        scheduler = self.build_lr_scheduler(optimizer, self._total_steps(loader))
        teacher_ready = self._teacher_weights is not None
        running_loss = 0.0
        running_task = 0.0
        running_distill = 0.0
        steps = 0
        num_samples = 0
        for epoch in range(self.config.local_epochs):
            for raw_batch in loader:
                batch = self._to_device(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                if teacher_ready:
                    distillation = self._distillation_loss(model, batch)
                    task_ce = self._forward_loss(model, batch)
                    loss = task_ce + self.hparams.lambda_o * distillation
                    running_distill += float(distillation.detach())
                else:
                    task_ce = self._forward_loss(model, batch)
                    loss = task_ce
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += float(loss.detach())
                running_task += float(task_ce.detach())
                steps += 1
                if epoch == 0:
                    num_samples += _batch_size(batch)
        stats: Dict[str, Any] = {
            "loss": running_loss / max(steps, 1),
            "task_ce": running_task / max(steps, 1),
            "distillation": running_distill / max(steps, 1),
            "temperature": self.hparams.temperature,
            "lambda_o": self.hparams.lambda_o,
            "has_teacher": teacher_ready,
            "local_steps": steps,
        }
        return ClientPayload(
            client_id=client_id,
            num_samples=max(num_samples, 1),
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
        return BroadcastPackage(
            weights=merged,
            extras={"method": self.name, "task_id": task_id, "round_id": round_id},
        )
