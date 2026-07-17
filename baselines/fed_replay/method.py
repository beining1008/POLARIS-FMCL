from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.aggregation import extract_trainable, fedavg, load_weights
from harness.interfaces import BroadcastPackage, ClientPayload, FederatedMethod
from models.backbone import load_backbone
from models.moe_lora import inject_moe_lora

from baselines.fed_replay.buffer import ExemplarBuffer


@dataclass(frozen=True)
class FedReplayHyperparams:
    exemplar_budget: int = 2000
    replay_ratio: float = 0.5
    herding_feature_batches: int = 32


class FedReplay(FederatedMethod):
    name = "fed-replay"

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hparams = FedReplayHyperparams()
        self._client_buffers: Dict[int, ExemplarBuffer] = {}

    def build_model(self) -> nn.Module:
        spec = self.backbone_spec
        model = load_backbone(spec, dtype=self.config.resolve_dtype())
        inject_moe_lora(
            model,
            adapter_layers=spec.adapter_layers,
            num_experts=self.config.num_experts,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            top_k=self.config.top_k,
            routing=self.config.routing,
        )
        return model

    def _buffer_for(self, client_id: int) -> ExemplarBuffer:
        if client_id not in self._client_buffers:
            self._client_buffers[client_id] = ExemplarBuffer(
                budget=self.hparams.exemplar_budget,
                feature_batches=self.hparams.herding_feature_batches,
            )
        return self._client_buffers[client_id]

    def _steps_per_epoch(self, loader: Iterable) -> int:
        try:
            return max(len(loader), 1)
        except TypeError:
            return 1

    def _batch_size(self, batch: Mapping[str, torch.Tensor]) -> int:
        for value in batch.values():
            return int(value.shape[0])
        return 0

    def _to_device(self, batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {key: value.to(device) for key, value in batch.items()}

    def _merge_batches(
        self,
        current: Mapping[str, torch.Tensor],
        replay: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {key: torch.cat([current[key], replay[key]], dim=0) for key in current.keys()}

    def _mix_replay(
        self,
        batch: Dict[str, torch.Tensor],
        buffer: ExemplarBuffer,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if buffer.is_empty():
            return batch
        replay_count = int(math.floor(self.hparams.replay_ratio * self._batch_size(batch)))
        if replay_count <= 0:
            return batch
        replay = buffer.sample(replay_count)
        if replay is None:
            return batch
        return self._merge_batches(batch, self._to_device(replay, device))

    def _token_cross_entropy(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        device = next(model.parameters()).device
        buffer = self._buffer_for(client_id)
        if round_id == 0:
            buffer.rebalance(task_id + 1)
        model.train()
        optimizer = self.build_optimizer(model)
        total_steps = max(self.config.local_epochs * self._steps_per_epoch(loader), 1)
        scheduler = self.build_lr_scheduler(optimizer, total_steps)
        observed = 0
        replay_steps = 0
        running_loss = 0.0
        steps = 0
        for epoch in range(self.config.local_epochs):
            for raw_batch in loader:
                batch = self._to_device(raw_batch, device)
                if epoch == 0:
                    observed += self._batch_size(batch)
                before = self._batch_size(batch)
                batch = self._mix_replay(batch, buffer, device)
                if self._batch_size(batch) > before:
                    replay_steps += 1
                loss = self._token_cross_entropy(model, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += float(loss.detach())
                steps += 1
        if round_id == self.config.rounds_per_task - 1:
            buffer.fill_from_shard(model, loader, task_id)
        stats = {
            "loss": running_loss / max(steps, 1),
            "exemplars": len(buffer),
            "slots_per_task": buffer.slots_per_task,
            "replay_steps": replay_steps,
            "replay_ratio": self.hparams.replay_ratio,
            "task_counts": buffer.task_counts(),
        }
        return ClientPayload(
            client_id=client_id,
            num_samples=max(observed, 1),
            weights=extract_trainable(model),
            stats=stats,
        )

    def aggregate(self, payloads: List[ClientPayload], task_id: int, round_id: int) -> BroadcastPackage:
        merged = fedavg(payloads)
        return BroadcastPackage(weights=merged, extras={"task_id": task_id, "round_id": round_id})
