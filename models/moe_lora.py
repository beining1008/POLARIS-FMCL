from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_MODULES = ("q_proj", "o_proj")


class LoRAExpert(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class TopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
        top_weights = F.softmax(top_values.to(torch.float32), dim=-1).to(logits.dtype)
        weights = torch.zeros_like(logits).scatter_(-1, top_indices, top_weights)
        return weights, top_indices


class DenseRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        weights = F.softmax(logits.to(torch.float32), dim=-1).to(logits.dtype)
        indices = torch.argsort(weights, dim=-1, descending=True)
        return weights, indices


class MoELoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        num_experts: int,
        rank: int,
        alpha: float,
        top_k: int = 1,
        routing: str = "topk",
        layer_index: int = -1,
    ) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer_index = layer_index
        dtype = base.weight.dtype
        self.experts = nn.ModuleList(
            LoRAExpert(base.in_features, base.out_features, rank, alpha, dtype)
            for _ in range(num_experts)
        )
        if routing == "dense":
            self.router: nn.Module = DenseRouter(base.in_features, num_experts, dtype)
        else:
            self.router = TopKRouter(base.in_features, num_experts, top_k, dtype)
        self.last_routing: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, _ = self.router(x)
        self.last_routing = weights.detach()
        output = self.base(x)
        for index, expert in enumerate(self.experts):
            gate = weights[..., index : index + 1]
            if bool(torch.any(gate > 0)):
                output = output + gate * expert(x)
        return output

    def expert_parameters(self, index: int) -> Tuple[nn.Parameter, nn.Parameter]:
        expert = self.experts[index]
        return expert.lora_A, expert.lora_B


def _parent_and_leaf(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _layer_index_of(qualified_name: str) -> int:
    parts = qualified_name.split(".")
    for left, right in zip(parts, parts[1:]):
        if left in ("layers", "h", "blocks") and right.isdigit():
            return int(right)
    return -1


def inject_moe_lora(
    model: nn.Module,
    adapter_layers: Tuple[int, ...],
    num_experts: int,
    rank: int,
    alpha: float,
    top_k: int = 1,
    routing: str = "topk",
    target_modules: Tuple[str, ...] = TARGET_MODULES,
) -> Dict[str, MoELoRALinear]:
    wrapped: Dict[str, MoELoRALinear] = {}
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        layer_index = _layer_index_of(name)
        if leaf in target_modules and layer_index in adapter_layers:
            replacements.append((name, module, layer_index))
    for name, module, layer_index in replacements:
        adapter = MoELoRALinear(
            module,
            num_experts=num_experts,
            rank=rank,
            alpha=alpha,
            top_k=top_k,
            routing=routing,
            layer_index=layer_index,
        )
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, adapter)
        wrapped[name] = adapter
    return wrapped


def iter_adapters(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, MoELoRALinear):
            yield name, module


def layer_label(adapter: MoELoRALinear, fallback: str) -> str:
    if adapter.layer_index >= 0:
        return f"layer{adapter.layer_index}"
    return fallback
