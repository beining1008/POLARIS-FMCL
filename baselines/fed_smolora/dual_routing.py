from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moe_lora import (
    TARGET_MODULES,
    LoRAExpert,
    _layer_index_of,
    _parent_and_leaf,
)

INSTRUCTION_EMBED_DIM = 384
SENTENCE_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

_INSTRUCTION_EMBEDDING: Optional[torch.Tensor] = None


def set_instruction_embedding(embedding: Optional[torch.Tensor]) -> None:
    global _INSTRUCTION_EMBEDDING
    _INSTRUCTION_EMBEDDING = embedding


def clear_instruction_embedding() -> None:
    global _INSTRUCTION_EMBEDDING
    _INSTRUCTION_EMBEDDING = None


def current_instruction_embedding() -> Optional[torch.Tensor]:
    return _INSTRUCTION_EMBEDDING


def encode_instruction(texts: Sequence[str]) -> torch.Tensor:
    raise NotImplementedError(
        f"{SENTENCE_ENCODER} returns a frozen [{len(texts)}, {INSTRUCTION_EMBED_DIM}] instruction embedding"
    )


def top1_gate(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits.to(torch.float32), dim=-1)
    top_value, top_index = probs.max(dim=-1, keepdim=True)
    weights = torch.zeros_like(probs).scatter_(-1, top_index, top_value)
    return weights.to(logits.dtype), top_index.squeeze(-1)


class SMoExpert(LoRAExpert):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        branch: str,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__(in_features, out_features, rank, alpha, dtype)
        self.branch = branch


class VisualUnderstandingGate(nn.Module):
    def __init__(self, hidden_size: int, num_blocks: int, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gate = nn.Linear(hidden_size, num_blocks, bias=False, dtype=dtype)

    def forward(self, pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return top1_gate(self.gate(pooled))


class InstructionFollowingGate(nn.Module):
    def __init__(self, embed_dim: int, num_blocks: int, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.gate = nn.Linear(embed_dim, num_blocks, bias=False, dtype=dtype)

    def forward(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return top1_gate(self.gate(embedding))


class AdaptiveFusion(nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.head = nn.Linear(hidden_size, 2, dtype=dtype)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        logits = self.head(pooled)
        return F.softmax(logits.to(torch.float32), dim=-1).to(logits.dtype)


class SMoLoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        num_vu_blocks: int,
        num_ins_blocks: int,
        rank: int,
        alpha: float,
        instruction_embed_dim: int = INSTRUCTION_EMBED_DIM,
        layer_index: int = -1,
    ) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.num_vu_blocks = num_vu_blocks
        self.num_ins_blocks = num_ins_blocks
        self.layer_index = layer_index
        in_features = base.in_features
        out_features = base.out_features
        dtype = base.weight.dtype
        self.lora_vu_blocks = nn.ModuleList(
            SMoExpert(in_features, out_features, rank, alpha, "vu", dtype)
            for _ in range(num_vu_blocks)
        )
        self.lora_ins_blocks = nn.ModuleList(
            SMoExpert(in_features, out_features, rank, alpha, "ins", dtype)
            for _ in range(num_ins_blocks)
        )
        self.lora_vu_gate = VisualUnderstandingGate(in_features, num_vu_blocks, dtype)
        self.lora_ins_gate = InstructionFollowingGate(instruction_embed_dim, num_ins_blocks, dtype)
        self.lora_fc = AdaptiveFusion(in_features, dtype)
        self.last_vu_routing: Optional[torch.Tensor] = None
        self.last_ins_routing: Optional[torch.Tensor] = None
        self.last_fusion: Optional[torch.Tensor] = None

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1) if x.dim() >= 3 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        broadcast_shape = (-1,) + (1,) * (base_out.dim() - 1)
        pooled = self._pool(x)
        vu_weights, _ = self.lora_vu_gate(pooled)
        fusion = self.lora_fc(pooled)
        vu_out = torch.zeros_like(base_out)
        for index, block in enumerate(self.lora_vu_blocks):
            gate = vu_weights[..., index].view(*broadcast_shape)
            vu_out = vu_out + gate * block(x)
        ins_out = torch.zeros_like(base_out)
        embedding = current_instruction_embedding()
        if embedding is not None:
            ins_weights, _ = self.lora_ins_gate(embedding.to(base_out.dtype))
            for index, block in enumerate(self.lora_ins_blocks):
                gate = ins_weights[..., index].view(*broadcast_shape)
                ins_out = ins_out + gate * block(x)
            self.last_ins_routing = ins_weights.detach()
        else:
            self.last_ins_routing = None
        alpha_fuse = fusion[..., 0].view(*broadcast_shape)
        beta_fuse = fusion[..., 1].view(*broadcast_shape)
        self.last_vu_routing = vu_weights.detach()
        self.last_fusion = fusion.detach()
        return base_out + alpha_fuse * vu_out + beta_fuse * ins_out


def iter_smolora(model: nn.Module) -> Iterator[Tuple[str, SMoLoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, SMoLoRALinear):
            yield name, module


def inject_smolora(
    model: nn.Module,
    adapter_layers: Tuple[int, ...],
    num_vu_blocks: int,
    num_ins_blocks: int,
    block_rank: int,
    alpha: float,
    instruction_embed_dim: int = INSTRUCTION_EMBED_DIM,
    target_modules: Tuple[str, ...] = TARGET_MODULES,
) -> Dict[str, SMoLoRALinear]:
    wrapped: Dict[str, SMoLoRALinear] = {}
    replacements: List[Tuple[str, nn.Linear, int]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        layer_index = _layer_index_of(name)
        if leaf in target_modules and layer_index in adapter_layers:
            replacements.append((name, module, layer_index))
    for name, module, layer_index in replacements:
        adapter = SMoLoRALinear(
            module,
            num_vu_blocks=num_vu_blocks,
            num_ins_blocks=num_ins_blocks,
            rank=block_rank,
            alpha=alpha,
            instruction_embed_dim=instruction_embed_dim,
            layer_index=layer_index,
        )
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, adapter)
        wrapped[name] = adapter
    return wrapped
