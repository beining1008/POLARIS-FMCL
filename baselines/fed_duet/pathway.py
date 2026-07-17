from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moe_lora import LoRAExpert


class CrossAttentionGate(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, num_experts: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.scale = 1.0 / math.sqrt(hidden_size)
        self.query = nn.Linear(in_features, hidden_size, bias=False, dtype=dtype)
        self.to_logits = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
        query = self.query(x)
        scores = torch.matmul(query, prompts.transpose(0, 1)) * self.scale
        attention = F.softmax(scores.to(torch.float32), dim=-1).to(query.dtype)
        context = torch.matmul(attention, prompts)
        logits = self.to_logits(context)
        mixture = F.softmax(logits.to(torch.float32), dim=-1).to(query.dtype)
        return mixture


class SemanticPathway(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_shared: int,
        num_local: int,
        num_prompts: int,
        rank: int,
        alpha: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_shared = num_shared
        self.num_local = num_local
        self.num_experts = num_shared + num_local
        self.shared_experts = nn.ModuleList(
            LoRAExpert(in_features, out_features, rank, alpha, dtype) for _ in range(num_shared)
        )
        self.local_experts = nn.ModuleList(
            LoRAExpert(in_features, out_features, rank, alpha, dtype) for _ in range(num_local)
        )
        self.prompts = nn.Parameter(torch.empty(num_prompts, hidden_size, dtype=dtype))
        nn.init.normal_(self.prompts, std=0.02)
        self.gate = CrossAttentionGate(in_features, hidden_size, self.num_experts, dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mixture = self.gate(x, self.prompts)
        experts: List[LoRAExpert] = list(self.shared_experts) + list(self.local_experts)
        output = mixture[..., 0:1] * experts[0](x)
        for index in range(1, len(experts)):
            output = output + mixture[..., index : index + 1] * experts[index](x)
        return output, mixture


class ParametricPathway(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.adapter = LoRAExpert(in_features, out_features, rank, alpha, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class PathwayFusion(nn.Module):
    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, query: torch.Tensor, semantic: torch.Tensor, parametric: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack((semantic, parametric), dim=-2)
        scale = 1.0 / math.sqrt(query.size(-1))
        scores = (query.unsqueeze(-2) * stacked).sum(dim=-1) * scale / self.temperature
        weights = F.softmax(scores.to(torch.float32), dim=-1).to(query.dtype)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
        return fused, weights


class DuetPathwayLinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        hidden_size: int,
        num_shared: int,
        num_local: int,
        num_prompts: int,
        rank: int,
        alpha: float,
        fusion_temperature: float,
        layer_index: int = -1,
    ) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        in_features = base.in_features
        out_features = base.out_features
        dtype = base.weight.dtype
        self.layer_index = layer_index
        self.num_semantic = num_shared + num_local
        self.semantic = SemanticPathway(
            in_features,
            out_features,
            hidden_size,
            num_shared,
            num_local,
            num_prompts,
            rank,
            alpha,
            dtype,
        )
        self.parametric = ParametricPathway(in_features, out_features, rank, alpha, dtype)
        self.fusion = PathwayFusion(fusion_temperature)
        self.last_mixture: Optional[torch.Tensor] = None
        self.last_fusion: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        semantic_out, mixture = self.semantic(x)
        parametric_out = self.parametric(x)
        fused, fusion_weights = self.fusion(base_out, semantic_out, parametric_out)
        self.last_mixture = mixture.detach().to(torch.float32).reshape(-1, self.num_semantic).mean(dim=0)
        self.last_fusion = fusion_weights.detach().to(torch.float32).reshape(-1, 2).mean(dim=0)
        return base_out + fused
