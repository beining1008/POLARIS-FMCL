from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.aggregation import extract_trainable, fedavg, load_weights
from harness.interfaces import BroadcastPackage, ClientPayload, FederatedMethod
from models.backbone import load_backbone
from models.moe_lora import MoELoRALinear, inject_moe_lora, iter_adapters, layer_label
from polaris.covariance import CovarianceBank, merge_pooled_matrices
from polaris.interference import GammaAccumulator, OnlineGammaEstimator
from polaris.pefosu import PEFOSUAggregator, union_key
from polaris.projection import BilateralProjector
from polaris.scheduler import InterferenceScheduler, effective_rank


@dataclass(frozen=True)
class PolarisHyperparams:
    budget_ratio: float = 0.75
    warmup_epochs: float = 1.0
    rank_threshold: float = 1e-3
    initial_layer_budget: int = 14
    per_expert_budget: Optional[int] = None


class Polaris(FederatedMethod):
    name = "polaris"

    def __init__(self, config, backbone_spec, hyperparams: Optional[PolarisHyperparams] = None) -> None:
        super().__init__(config, backbone_spec)
        self.hyper = hyperparams or PolarisHyperparams()
        self.scheduler = InterferenceScheduler(
            budget_ratio=self.hyper.budget_ratio,
            rank_threshold=self.hyper.rank_threshold,
        )
        self.estimator = OnlineGammaEstimator()
        self.aggregator = PEFOSUAggregator()
        self.projector = BilateralProjector()
        self._client_banks: Dict[int, CovarianceBank] = {}
        self._module_layers: Dict[str, str] = {}
        self._budgets: Dict[str, int] = {}
        self._per_task_ranks: Dict[str, int] = {}
        self._active_task: int = 0

    def build_model(self) -> nn.Module:
        model = load_backbone(self.backbone_spec, self.config.resolve_dtype())
        inject_moe_lora(
            model,
            self.backbone_spec.adapter_layers,
            num_experts=self.config.num_experts,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            top_k=self.config.top_k,
            routing=self.config.routing,
        )
        return model

    def on_task_start(self, task_id: int) -> None:
        self._active_task = task_id
        self.projector.load(self.aggregator.export_bases())

    def _forward_loss(self, model: nn.Module, batch: Dict) -> torch.Tensor:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = batch["labels"][..., 1:].contiguous()
        return F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)).to(torch.float32),
            shifted_labels.view(-1),
            ignore_index=-100,
        )

    def _ensure_budgets(self, adapters: Dict[str, MoELoRALinear]) -> None:
        for module_name, adapter in adapters.items():
            label = layer_label(adapter, module_name)
            self._module_layers[module_name] = label
            if label not in self._budgets:
                self._budgets[label] = self.hyper.initial_layer_budget

    def _budget_of(self, module_name: str) -> int:
        label = self._module_layers.get(module_name, module_name)
        return self._budgets.get(label, self.hyper.initial_layer_budget)

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
            if "bases" in broadcast.extras:
                self.projector.load(broadcast.extras["bases"])
        model.train()
        adapters = dict(iter_adapters(model))
        self._ensure_budgets(adapters)
        bank = self._client_banks.setdefault(client_id, CovarianceBank(self.config.num_experts))
        steps_per_epoch = len(loader)
        total_steps = steps_per_epoch * self.config.local_epochs
        warmup_steps = int(self.hyper.warmup_epochs * steps_per_epoch)
        optimizer = self.build_optimizer(model)
        lr_scheduler = self.build_lr_scheduler(optimizer, total_steps)
        gamma = GammaAccumulator() if task_id >= 1 and self.projector.has_bases() else None
        step = 0
        for _ in range(self.config.local_epochs):
            for batch in loader:
                optimizer.zero_grad(set_to_none=False)
                loss = self._forward_loss(model, batch)
                loss.backward()
                if gamma is not None and step < warmup_steps:
                    gamma.accumulate(adapters, self.projector.bases)
                strength = self.projector.warmup_strength(step, warmup_steps)
                self.projector.apply(adapters, strength)
                bank.accumulate(adapters)
                optimizer.step()
                lr_scheduler.step()
                step += 1
        statistics: Dict = {"local_steps": step}
        if gamma is not None:
            statistics["gamma"] = gamma.export()
        return ClientPayload(
            client_id=client_id,
            num_samples=len(loader.dataset),
            weights=extract_trainable(model),
            stats=statistics,
        )

    def aggregate(self, payloads: List[ClientPayload], task_id: int, round_id: int) -> BroadcastPackage:
        merged = fedavg(payloads)
        return BroadcastPackage(
            weights=merged,
            extras={
                "bases": self.aggregator.export_bases(),
                "budgets": dict(self._budgets),
            },
        )

    def _mixing_weights(self, payloads: List[ClientPayload]) -> List[Tuple[int, float]]:
        total = float(sum(p.num_samples for p in payloads))
        return [(p.client_id, p.num_samples / total) for p in payloads]

    def _union_all(self, mixing: List[Tuple[int, float]]) -> None:
        layer_names = set()
        for _, bank in self._client_banks.items():
            layer_names.update(bank.layers())
        for layer_name in sorted(layer_names):
            rank = self._budget_of(layer_name)
            for expert_index in range(self.config.num_experts):
                factors: List[Tuple[float, torch.Tensor]] = []
                for client_id, mix_weight in mixing:
                    bank = self._client_banks.get(client_id)
                    if bank is None:
                        continue
                    factor = bank.factorize(layer_name, expert_index, rank)
                    if factor is not None:
                        factors.append((mix_weight, factor))
                if factors:
                    self.aggregator.union(union_key(layer_name, expert_index), factors, rank)

    def _refresh_per_task_ranks(self, mixing: List[Tuple[int, float]]) -> None:
        weighted_banks = [
            (mix_weight, self._client_banks[client_id])
            for client_id, mix_weight in mixing
            if client_id in self._client_banks
        ]
        module_names = set()
        for _, bank in weighted_banks:
            module_names.update(bank.layers())
        grouped: Dict[str, torch.Tensor] = {}
        for module_name in module_names:
            pooled = merge_pooled_matrices(weighted_banks, module_name)
            if pooled is None:
                continue
            label = self._module_layers.get(module_name, module_name)
            grouped[label] = pooled if label not in grouped else grouped[label] + pooled
        for label, pooled in grouped.items():
            eigenvalues = torch.linalg.eigvalsh(pooled).clamp_min(0.0)
            self._per_task_ranks[label] = effective_rank(
                eigenvalues, self.scheduler.rank_threshold
            )

    def on_task_end(self, task_id: int, payloads: List[ClientPayload]) -> None:
        mixing = self._mixing_weights(payloads)
        self._refresh_per_task_ranks(mixing)
        self._union_all(mixing)
        if task_id >= 1:
            client_statistics = [
                (mix_weight, payload.stats["gamma"])
                for payload, (_, mix_weight) in zip(payloads, mixing)
                if "gamma" in payload.stats
            ]
            merged = self.estimator.merge_clients(client_statistics)
            self.estimator.update_task(merged)
        if self.estimator.tasks_observed() >= 1:
            caps = self.scheduler.capacity_caps(self._per_task_ranks)
            landscape = self.estimator.landscape()
            self._budgets = self.scheduler.allocate(
                landscape, caps, self.hyper.per_expert_budget
            )
        for bank in self._client_banks.values():
            bank.reset()
        current = self.current_broadcast()
        if current is not None:
            self.set_broadcast(
                BroadcastPackage(
                    weights=current.weights,
                    extras={
                        "bases": self.aggregator.export_bases(),
                        "budgets": dict(self._budgets),
                        "union_diagnostics": {
                            key: {
                                "spectral_cut": result.spectral_cut,
                                "retained_variance_ratio": result.retained_variance_ratio,
                                "carry_over_residual": result.carry_over_residual,
                            }
                            for key, result in self.aggregator.last_diagnostics.items()
                        },
                    },
                )
            )
