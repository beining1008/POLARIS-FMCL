from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.aggregation import extract_trainable, fedavg, load_weights
from harness.interfaces import BroadcastPackage, ClientPayload, FederatedMethod
from models.backbone import load_backbone

from baselines.fed_smolora.dual_routing import (
    INSTRUCTION_EMBED_DIM,
    clear_instruction_embedding,
    encode_instruction,
    inject_smolora,
    iter_smolora,
    set_instruction_embedding,
)


@dataclass(frozen=True)
class FedSMoLoRAHyperparams:
    num_vu_blocks: int = 2
    num_if_blocks: int = 2
    block_rank: int = 8
    instruction_embed_dim: int = INSTRUCTION_EMBED_DIM
    published_reference_blocks: int = 4
    published_reference_rank: int = 16


def _instruction_texts(batch: Dict[str, Any]) -> List[str]:
    if "instruction" in batch:
        return list(batch["instruction"])
    return list(batch["question"])


class FedSMoLoRA(FederatedMethod):
    name = "fed-smolora"

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hparams = FedSMoLoRAHyperparams()
        self.dtype = config.resolve_dtype()

    def build_model(self) -> nn.Module:
        assert self.hparams.num_vu_blocks == self.hparams.num_if_blocks
        assert self.hparams.num_vu_blocks + self.hparams.num_if_blocks == self.config.num_experts
        assert self.hparams.num_vu_blocks <= self.hparams.published_reference_blocks
        assert self.hparams.block_rank <= self.hparams.published_reference_rank
        model = load_backbone(self.backbone_spec, self.dtype)
        inject_smolora(
            model,
            adapter_layers=self.backbone_spec.adapter_layers,
            num_vu_blocks=self.hparams.num_vu_blocks,
            num_ins_blocks=self.hparams.num_if_blocks,
            block_rank=self.hparams.block_rank,
            alpha=self.config.lora_alpha,
            instruction_embed_dim=self.hparams.instruction_embed_dim,
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

    def _language_modeling_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

    def _collect_routing_stats(
        self, model: nn.Module
    ) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        vu_mass: List[float] = []
        ins_mass: List[float] = []
        vu_usage: List[torch.Tensor] = []
        ins_usage: List[torch.Tensor] = []
        for _, layer in iter_smolora(model):
            if layer.last_fusion is not None:
                vu_mass.append(float(layer.last_fusion[..., 0].mean()))
                ins_mass.append(float(layer.last_fusion[..., 1].mean()))
            if layer.last_vu_routing is not None:
                vu_usage.append((layer.last_vu_routing > 0).float().mean(dim=0).cpu())
            if layer.last_ins_routing is not None:
                ins_usage.append((layer.last_ins_routing > 0).float().mean(dim=0).cpu())
        vu = sum(vu_mass) / len(vu_mass) if vu_mass else 0.0
        ins = sum(ins_mass) / len(ins_mass) if ins_mass else 0.0
        vu_hist = (
            torch.stack(vu_usage, dim=0).mean(dim=0)
            if vu_usage
            else torch.zeros(self.hparams.num_vu_blocks)
        )
        ins_hist = (
            torch.stack(ins_usage, dim=0).mean(dim=0)
            if ins_usage
            else torch.zeros(self.hparams.num_if_blocks)
        )
        return vu, ins, vu_hist, ins_hist

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
        loss_total = 0.0
        vu_mass_total = 0.0
        ins_mass_total = 0.0
        vu_usage_total = torch.zeros(self.hparams.num_vu_blocks)
        ins_usage_total = torch.zeros(self.hparams.num_if_blocks)

        for epoch in range(self.config.local_epochs):
            for batch in loader:
                batch = self._prepare_batch(batch, device)
                labels = batch["labels"]
                embedding = encode_instruction(_instruction_texts(batch)).to(
                    device=device, dtype=self.dtype
                )
                set_instruction_embedding(embedding)

                logits = self._forward_logits(model, batch)
                loss = self._language_modeling_loss(logits, labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                clear_instruction_embedding()

                vu_mass, ins_mass, vu_usage, ins_usage = self._collect_routing_stats(model)
                vu_mass_total += vu_mass
                ins_mass_total += ins_mass
                vu_usage_total = vu_usage_total + vu_usage
                ins_usage_total = ins_usage_total + ins_usage
                step_count += 1
                loss_total += float(loss.detach())
                if epoch == 0:
                    num_samples += int(labels.size(0))

        denom = max(1, step_count)
        stats: Dict[str, Any] = {
            "loss": loss_total / denom,
            "vu_routing_mass": vu_mass_total / denom,
            "if_routing_mass": ins_mass_total / denom,
            "vu_block_usage": (vu_usage_total / denom).tolist(),
            "if_block_usage": (ins_usage_total / denom).tolist(),
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
        vu_mass = sum(
            payload.stats.get("vu_routing_mass", 0.0) * payload.num_samples for payload in payloads
        ) / max(total, 1.0)
        if_mass = sum(
            payload.stats.get("if_routing_mass", 0.0) * payload.num_samples for payload in payloads
        ) / max(total, 1.0)
        return BroadcastPackage(
            weights=merged,
            extras={"vu_routing_mass": vu_mass, "if_routing_mass": if_mass},
        )
