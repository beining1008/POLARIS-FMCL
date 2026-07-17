from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moe_lora import (
    DenseRouter,
    TARGET_MODULES,
    TopKRouter,
    _layer_index_of,
    _parent_and_leaf,
)


def _fresh_pair(
    in_features: int,
    out_features: int,
    rank: int,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = torch.empty(rank, in_features, dtype=torch.float32, device=device)
    nn.init.kaiming_uniform_(a, a=math.sqrt(5))
    b = torch.zeros(out_features, rank, dtype=dtype, device=device)
    return a.to(dtype), b


def _grow_stack(
    old_a: Optional[torch.Tensor],
    old_b: Optional[torch.Tensor],
    new_a: torch.Tensor,
    new_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if old_a is None or old_a.shape[0] == 0:
        return new_a.clone(), new_b.clone()
    grown_a = torch.cat([old_a, new_a.to(old_a.dtype)], dim=0)
    grown_b = torch.cat([old_b, new_b.to(old_b.dtype)], dim=1)
    return grown_a, grown_b


class OLoRAExpert(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        fresh_a, fresh_b = _fresh_pair(in_features, out_features, rank, dtype)
        self.loranew_A = nn.Parameter(fresh_a)
        self.loranew_B = nn.Parameter(fresh_b)
        self.register_buffer("lora_A_stack", torch.zeros(0, in_features, dtype=dtype))
        self.register_buffer("lora_B_stack", torch.zeros(out_features, 0, dtype=dtype))

    @property
    def stack_rank(self) -> int:
        return int(self.lora_A_stack.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(F.linear(x, self.loranew_A), self.loranew_B) * self.scaling
        if self.lora_A_stack.shape[0] > 0:
            frozen = F.linear(F.linear(x, self.lora_A_stack), self.lora_B_stack)
            out = out + frozen * self.scaling
        return out

    def set_stack(self, a_stack: torch.Tensor, b_stack: torch.Tensor) -> None:
        device = self.loranew_A.device
        self.lora_A_stack = a_stack.detach().to(dtype=self.lora_A_stack.dtype, device=device)
        self.lora_B_stack = b_stack.detach().to(dtype=self.lora_B_stack.dtype, device=device)

    def fold_loranew(self) -> None:
        grown_a, grown_b = _grow_stack(
            self.lora_A_stack,
            self.lora_B_stack,
            self.loranew_A.detach(),
            self.loranew_B.detach(),
        )
        self.lora_A_stack = grown_a.to(self.lora_A_stack.dtype)
        self.lora_B_stack = grown_b.to(self.lora_B_stack.dtype)
        fresh_a, fresh_b = _fresh_pair(
            self.in_features,
            self.out_features,
            self.rank,
            self.loranew_A.dtype,
            self.loranew_A.device,
        )
        with torch.no_grad():
            self.loranew_A.copy_(fresh_a)
            self.loranew_B.copy_(fresh_b)


class OLoRAMoELinear(nn.Module):
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
            OLoRAExpert(base.in_features, base.out_features, rank, alpha, dtype)
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
        return expert.loranew_A, expert.loranew_B


def inject_olora_moe(
    model: nn.Module,
    adapter_layers: Tuple[int, ...],
    num_experts: int,
    rank: int,
    alpha: float,
    top_k: int = 1,
    routing: str = "topk",
    target_modules: Tuple[str, ...] = TARGET_MODULES,
) -> Dict[str, OLoRAMoELinear]:
    wrapped: Dict[str, OLoRAMoELinear] = {}
    replacements: List[Tuple[str, nn.Linear, int]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        layer_index = _layer_index_of(name)
        if leaf in target_modules and layer_index in adapter_layers:
            replacements.append((name, module, layer_index))
    for name, module, layer_index in replacements:
        adapter = OLoRAMoELinear(
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


def iter_olora(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, OLoRAMoELinear):
            yield name, module
