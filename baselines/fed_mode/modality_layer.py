from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moe_lora import (
    TARGET_MODULES,
    LoRAExpert,
    _layer_index_of,
    _parent_and_leaf,
)

_CURRENT_TOKEN_MASK: Optional[torch.Tensor] = None


def set_token_type_mask(mask: Optional[torch.Tensor]) -> None:
    global _CURRENT_TOKEN_MASK
    _CURRENT_TOKEN_MASK = mask


def clear_token_type_mask() -> None:
    global _CURRENT_TOKEN_MASK
    _CURRENT_TOKEN_MASK = None


def current_token_type_mask() -> Optional[torch.Tensor]:
    return _CURRENT_TOKEN_MASK


class ModalityDecoupledLinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        num_text_experts: int,
        rank: int,
        alpha: float,
        layer_index: int = -1,
    ) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.num_text_experts = num_text_experts
        self.layer_index = layer_index
        dtype = base.weight.dtype
        self.v_adapter = LoRAExpert(base.in_features, base.out_features, rank, alpha, dtype)
        self.text_experts = nn.ModuleList(
            LoRAExpert(base.in_features, base.out_features, rank, alpha, dtype)
            for _ in range(num_text_experts)
        )
        self.text_router = nn.Linear(base.in_features, num_text_experts, bias=False, dtype=dtype)
        self.adapters_enabled: bool = True
        self.last_text_routing: Optional[torch.Tensor] = None
        self.last_text_mass: Optional[torch.Tensor] = None
        self.last_visual_fraction: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if not self.adapters_enabled:
            return base_out
        visual = current_token_type_mask()
        if visual is None:
            visual_gate = x.new_zeros(x.shape[:-1] + (1,))
        else:
            visual_gate = visual.to(x.dtype).unsqueeze(-1)
        text_gate = 1.0 - visual_gate
        visual_out = self.v_adapter(x) * visual_gate
        logits = self.text_router(x)
        probs = F.softmax(logits.float(), dim=-1).to(x.dtype)
        text_out = torch.zeros_like(base_out)
        for index, expert in enumerate(self.text_experts):
            text_out = text_out + probs[..., index : index + 1] * expert(x)
        text_out = text_out * text_gate
        detached = probs.detach()
        text_weight = text_gate.detach()
        denom = text_weight.sum().clamp_min(1.0)
        reduce_dims = tuple(range(detached.dim() - 1))
        self.last_text_routing = detached
        self.last_text_mass = (detached * text_weight).sum(dim=reduce_dims) / denom
        self.last_visual_fraction = visual_gate.detach().mean()
        return base_out + visual_out + text_out


def iter_modality_layers(model: nn.Module) -> Iterator[Tuple[str, ModalityDecoupledLinear]]:
    for name, module in model.named_modules():
        if isinstance(module, ModalityDecoupledLinear):
            yield name, module


class adapters_disabled:
    def __init__(self, model: nn.Module) -> None:
        self.layers: List[ModalityDecoupledLinear] = [
            module for _, module in iter_modality_layers(model)
        ]
        self.previous: List[bool] = []

    def __enter__(self) -> "adapters_disabled":
        self.previous = [layer.adapters_enabled for layer in self.layers]
        for layer in self.layers:
            layer.adapters_enabled = False
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        for layer, previous in zip(self.layers, self.previous):
            layer.adapters_enabled = previous
        return False


def inject_modality_decoupled(
    model: nn.Module,
    adapter_layers: Tuple[int, ...],
    num_text_experts: int,
    rank: int,
    alpha: float,
    target_modules: Tuple[str, ...] = TARGET_MODULES,
) -> Dict[str, ModalityDecoupledLinear]:
    wrapped: Dict[str, ModalityDecoupledLinear] = {}
    replacements: List[Tuple[str, nn.Linear, int]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        layer_index = _layer_index_of(name)
        if leaf in target_modules and layer_index in adapter_layers:
            replacements.append((name, module, layer_index))
    for name, module, layer_index in replacements:
        adapter = ModalityDecoupledLinear(
            module,
            num_text_experts=num_text_experts,
            rank=rank,
            alpha=alpha,
            layer_index=layer_index,
        )
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, adapter)
        wrapped[name] = adapter
    return wrapped
