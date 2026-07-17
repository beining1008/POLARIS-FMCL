from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hidden_size: int
    num_layers: int
    adapter_layers: Tuple[int, ...]
    vision_tower: str
    projector: str


BACKBONES = {
    "llava-1.5-7b": BackboneSpec(
        name="llava-1.5-7b",
        hidden_size=4096,
        num_layers=32,
        adapter_layers=tuple(range(24, 32)),
        vision_tower="clip-vit-large-patch14-336",
        projector="mlp2x_gelu",
    ),
    "llava-1.5-13b": BackboneSpec(
        name="llava-1.5-13b",
        hidden_size=5120,
        num_layers=40,
        adapter_layers=tuple(range(32, 40)),
        vision_tower="clip-vit-large-patch14-336",
        projector="mlp2x_gelu",
    ),
    "qwen2.5-vl-7b": BackboneSpec(
        name="qwen2.5-vl-7b",
        hidden_size=3584,
        num_layers=28,
        adapter_layers=tuple(range(20, 28)),
        vision_tower="native-vit",
        projector="merger",
    ),
}


def load_backbone(spec: BackboneSpec, dtype: torch.dtype = torch.float16) -> nn.Module:
    raise NotImplementedError


def load_tokenizer(spec: BackboneSpec):
    raise NotImplementedError


def generate_answer(model: nn.Module, tokenizer, image, question: str, max_new_tokens: int = 32) -> str:
    raise NotImplementedError


def compute_token_type_mask(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    raise NotImplementedError
