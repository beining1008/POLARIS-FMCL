from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.aggregation import extract_trainable, fedavg, load_weights
from harness.interfaces import BroadcastPackage, ClientPayload, FederatedMethod
from models.backbone import load_backbone
from models.moe_lora import inject_moe_lora, iter_adapters

from baselines.fed_splitlora.split_state import SubspaceSplitState, move_batch


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


@dataclass(frozen=True)
class FedSplitLoRAHyperparams:
    split_alpha: float = 20.0
    split_temperature: float = 28.0
    activation_batches: int = 32


class FedSplitLoRA(FederatedMethod):
    name = "fed-splitlora"

    def __init__(self, config, backbone_spec) -> None:
        super().__init__(config, backbone_spec)
        self.hp = FedSplitLoRAHyperparams()
        self._client_split: Dict[int, SubspaceSplitState] = {}
        self._task_id = 0
        self._reinit_pending: set = set()

    def build_model(self) -> nn.Module:
        dtype = self.config.resolve_dtype()
        model = load_backbone(self.backbone_spec, dtype)
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

    def on_task_start(self, task_id: int) -> None:
        self._task_id = task_id
        self._reinit_pending = set(self._client_split.keys()) if task_id > 0 else set()

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
        state = self._state(client_id)
        if round_id == 0 and client_id in self._reinit_pending and state.has_split():
            self._reinitialize_experts(model, state, device)
            self._reinit_pending.discard(client_id)
        if round_id == 0:
            state.update_grad_before_train(model, loader, device)
        stats = self._train_local(model, loader, state, device)
        if round_id == self.config.rounds_per_task - 1:
            state.update_grad_after_train(model, loader, device)
        payload = ClientPayload(
            client_id=client_id,
            num_samples=max(int(stats.get("seen", 0)), 1),
            weights=extract_trainable(model),
            stats=stats,
        )
        return payload

    def aggregate(self, payloads: List[ClientPayload], task_id: int, round_id: int) -> BroadcastPackage:
        merged = fedavg(payloads)
        return BroadcastPackage(weights=merged, extras={"task_id": task_id, "round_id": round_id})

    def on_task_end(self, task_id: int, payloads: List[ClientPayload]) -> None:
        broadcast = self.current_broadcast()
        if broadcast is not None:
            routing = self._aggregate_routing(payloads)
            deltas = self._task_deltas(broadcast.weights, routing)
            for state in self._client_split.values():
                for name, delta in deltas.items():
                    state.fold_old_weight(name, delta)
        for state in self._client_split.values():
            state.split(task_id)

    def _state(self, client_id: int) -> SubspaceSplitState:
        state = self._client_split.get(client_id)
        if state is None:
            state = SubspaceSplitState(
                self.hp.split_alpha,
                self.hp.split_temperature,
                self.hp.activation_batches,
            )
            self._client_split[client_id] = state
        return state

    def _reinitialize_experts(self, model: nn.Module, state: SubspaceSplitState, device: torch.device) -> None:
        with torch.no_grad():
            for name, adapter in iter_adapters(model):
                if not state.has_split_for(name):
                    continue
                for expert in adapter.experts:
                    rows = state.minor_subspace_init(name, expert.lora_A.shape[0], expert.lora_A.dtype, device)
                    expert.lora_A.copy_(rows)
                    nn.init.zeros_(expert.lora_B)

    def _train_local(
        self,
        model: nn.Module,
        loader: Iterable,
        state: SubspaceSplitState,
        device: torch.device,
    ) -> Dict[str, Any]:
        model.train()
        optimizer = self.build_optimizer(model)
        total_steps = max(self.config.local_epochs * self._loader_length(loader), 1)
        scheduler = self.build_lr_scheduler(optimizer, total_steps)
        handles = self._register_old_weight(model, state)
        route_mass: Dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros(self.config.num_experts, dtype=torch.float32)
        )
        seen = 0
        running = 0.0
        for _ in range(self.config.local_epochs):
            for batch in loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = compute_language_modeling_loss(model, batch)
                loss.backward()
                self._project_gradients(model, state)
                optimizer.step()
                scheduler.step()
                seen += self._infer_batch_size(batch)
                running += float(loss.detach())
                self._collect_routing(model, route_mass)
        for handle in handles:
            handle.remove()
        return {
            "seen": seen,
            "loss": running / max(seen, 1),
            "route_mass": {name: mass.tolist() for name, mass in route_mass.items()},
        }

    def _project_gradients(self, model: nn.Module, state: SubspaceSplitState) -> None:
        if not state.has_split():
            return
        with torch.no_grad():
            for name, adapter in iter_adapters(model):
                if not state.has_split_for(name):
                    continue
                for expert in adapter.experts:
                    if expert.lora_A.grad is not None:
                        expert.lora_A.grad.copy_(state.project_to_minor(name, expert.lora_A.grad))

    def _register_old_weight(self, model: nn.Module, state: SubspaceSplitState) -> List[Any]:
        handles: List[Any] = []
        for name, adapter in iter_adapters(model):
            if state.old_weight_for(name) is None:
                continue
            handles.append(adapter.register_forward_hook(self._make_old_weight_hook(name, state)))
        return handles

    def _make_old_weight_hook(self, name: str, state: SubspaceSplitState):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            delta = state.old_weight_for(name)
            if delta is None:
                return output
            return output + F.linear(inputs[0], delta.to(dtype=inputs[0].dtype))

        return hook

    def _collect_routing(self, model: nn.Module, route_mass: Dict[str, torch.Tensor]) -> None:
        for name, adapter in iter_adapters(model):
            routing = adapter.last_routing
            if routing is None:
                continue
            flat = routing.detach().to(torch.float32).reshape(-1, adapter.num_experts)
            route_mass[name] = route_mass[name] + flat.sum(dim=0).cpu()

    def _aggregate_routing(self, payloads: List[ClientPayload]) -> Dict[str, torch.Tensor]:
        total = float(sum(payload.num_samples for payload in payloads)) or 1.0
        aggregate: Dict[str, torch.Tensor] = {}
        for payload in payloads:
            weight = payload.num_samples / total
            masses = payload.stats.get("route_mass", {})
            for name, values in masses.items():
                tensor = torch.tensor(values, dtype=torch.float32) * weight
                aggregate[name] = aggregate[name] + tensor if name in aggregate else tensor
        return aggregate

    def _task_deltas(
        self,
        weights: Dict[str, torch.Tensor],
        routing: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        scaling = self.config.lora_alpha / self.config.lora_rank
        deltas: Dict[str, torch.Tensor] = {}
        for name, experts in self._group_expert_params(weights).items():
            shares = self._normalized_shares(routing.get(name), len(experts))
            accumulator: Optional[torch.Tensor] = None
            for index, (lora_a, lora_b) in enumerate(experts):
                contribution = shares[index] * scaling * lora_b.to(torch.float32).mm(lora_a.to(torch.float32))
                accumulator = contribution if accumulator is None else accumulator + contribution
            if accumulator is not None:
                deltas[name] = accumulator
        return deltas

    @staticmethod
    def _group_expert_params(
        weights: Dict[str, torch.Tensor],
    ) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
        groups: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
        for key, value in weights.items():
            if ".experts." not in key:
                continue
            head, tail = key.split(".experts.", 1)
            parts = tail.split(".")
            leaf = parts[-1]
            if leaf in ("lora_A", "lora_B"):
                groups[head][int(parts[0])][leaf] = value
        ordered: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        for name, experts in groups.items():
            pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for index in sorted(experts.keys()):
                entry = experts[index]
                if "lora_A" in entry and "lora_B" in entry:
                    pairs.append((entry["lora_A"], entry["lora_B"]))
            ordered[name] = pairs
        return ordered

    @staticmethod
    def _normalized_shares(mass: Optional[torch.Tensor], num_experts: int) -> List[float]:
        if mass is None:
            return [1.0 / num_experts] * num_experts
        total = float(mass.sum())
        if total <= 0.0:
            return [1.0 / num_experts] * num_experts
        return [float(value) / total for value in mass]

    @staticmethod
    def _loader_length(loader: Iterable) -> int:
        try:
            return max(len(loader), 1)
        except TypeError:
            return 1

    @staticmethod
    def _infer_batch_size(batch: Dict[str, Any]) -> int:
        for key in ("input_ids", "labels", "pixel_values"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        return 1
