from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.aggregation import extract_trainable, fedavg, load_weights
from harness.interfaces import BroadcastPackage, ClientPayload, FederatedMethod
from models.backbone import compute_token_type_mask, load_backbone

from baselines.fed_mode.modality_layer import (
    adapters_disabled,
    clear_token_type_mask,
    inject_modality_decoupled,
    iter_modality_layers,
    set_token_type_mask,
)


@dataclass(frozen=True)
class FedMoDEHyperparams:
    num_text_experts: int = 4
    adapter_rank: int = 8
    kd_lambda: float = 0.3
    kd_temperature: float = 2.0
    image_token_id: int = 32000


class FedMoDE(FederatedMethod):
    name = "fed-mode"

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hparams = FedMoDEHyperparams()

    def build_model(self) -> nn.Module:
        dtype = self.config.resolve_dtype()
        model = load_backbone(self.backbone_spec, dtype)
        inject_modality_decoupled(
            model,
            adapter_layers=self.backbone_spec.adapter_layers,
            num_text_experts=self.hparams.num_text_experts,
            rank=self.hparams.adapter_rank,
            alpha=self.config.lora_alpha,
        )
        return model

    def _model_device(self, model: nn.Module) -> torch.device:
        return next(model.parameters()).device

    def _loader_steps(self, loader: Iterable) -> int:
        try:
            return max(1, len(loader))
        except TypeError:
            return 1

    def _prepare_batch(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        prepared: Dict[str, Any] = {}
        for key, value in batch.items():
            prepared[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return prepared

    def _forward_logits(self, model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
        )
        return outputs.logits if hasattr(outputs, "logits") else outputs

    def _teacher_logits(self, model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            with adapters_disabled(model):
                logits = self._forward_logits(model, batch)
        if was_training:
            model.train()
        return logits

    def _language_modeling_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

    def _kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        temperature = self.hparams.kd_temperature
        vocab = student_logits.size(-1)
        student = F.log_softmax(student_logits.reshape(-1, vocab).float() / temperature, dim=-1)
        teacher = F.softmax(teacher_logits.reshape(-1, vocab).float() / temperature, dim=-1)
        return F.kl_div(student, teacher, reduction="batchmean") * (temperature * temperature)

    def _collect_routing_stats(self, model: nn.Module) -> Tuple[torch.Tensor, float]:
        masses: List[torch.Tensor] = []
        fractions: List[float] = []
        for _, layer in iter_modality_layers(model):
            if layer.last_text_mass is not None:
                masses.append(layer.last_text_mass.detach().float().cpu())
            if layer.last_visual_fraction is not None:
                fractions.append(float(layer.last_visual_fraction.detach().cpu()))
        if masses:
            mass = torch.stack(masses, dim=0).mean(dim=0)
        else:
            mass = torch.zeros(self.hparams.num_text_experts)
        fraction = sum(fractions) / len(fractions) if fractions else 0.0
        return mass, fraction

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
        device = self._model_device(model)
        model.train()
        optimizer = self.build_optimizer(model)
        total_steps = self.config.local_epochs * self._loader_steps(loader)
        scheduler = self.build_lr_scheduler(optimizer, total_steps)

        num_samples = 0
        step_count = 0
        ce_total = 0.0
        kd_total = 0.0
        loss_total = 0.0
        mass_sum = torch.zeros(self.hparams.num_text_experts)
        fraction_sum = 0.0

        for epoch in range(self.config.local_epochs):
            for batch in loader:
                batch = self._prepare_batch(batch, device)
                labels = batch["labels"]
                token_mask = compute_token_type_mask(batch["input_ids"], self.hparams.image_token_id)
                set_token_type_mask(token_mask)

                student_logits = self._forward_logits(model, batch)
                teacher_logits = self._teacher_logits(model, batch)

                ce = self._language_modeling_loss(student_logits, labels)
                kd = self._kd_loss(student_logits, teacher_logits)
                loss = ce + self.hparams.kd_lambda * kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                clear_token_type_mask()

                mass, fraction = self._collect_routing_stats(model)
                mass_sum = mass_sum + mass
                fraction_sum += fraction
                step_count += 1
                ce_total += float(ce.detach())
                kd_total += float(kd.detach())
                loss_total += float(loss.detach())
                if epoch == 0:
                    num_samples += int(labels.size(0))

        denom = max(1, step_count)
        stats: Dict[str, Any] = {
            "ce_loss": ce_total / denom,
            "kd_loss": kd_total / denom,
            "loss": loss_total / denom,
            "visual_fraction": fraction_sum / denom,
            "text_routing_mass": (mass_sum / denom).tolist(),
            "steps": step_count,
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
        total = float(sum(payload.num_samples for payload in payloads))
        visual_fraction = sum(
            payload.stats.get("visual_fraction", 0.0) * payload.num_samples for payload in payloads
        ) / max(total, 1.0)
        return BroadcastPackage(weights=merged, extras={"visual_fraction": visual_fraction})
