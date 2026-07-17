from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

VISION_HIDDEN: Dict[str, int] = {
    "llava-1.5-7b": 1024,
    "llava-1.5-13b": 1024,
    "qwen2.5-vl-7b": 1280,
}


class ProjectorMLP(nn.Module):
    def __init__(
        self,
        vision_hidden: int,
        hidden_size: int,
        multiplier: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        inner = hidden_size * multiplier
        self.net = nn.Sequential(
            nn.Linear(vision_hidden, inner, dtype=dtype),
            nn.GELU(),
            nn.Linear(inner, hidden_size, dtype=dtype),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ProjectorLibrary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict()
        self._active: Optional[str] = None

    def allocate(
        self,
        task: str,
        vision_hidden: int,
        hidden_size: int,
        multiplier: int,
        dtype: torch.dtype,
    ) -> None:
        if task in self.projectors:
            return
        self.projectors[task] = ProjectorMLP(vision_hidden, hidden_size, multiplier, dtype)

    def module(self, task: str) -> ProjectorMLP:
        return self.projectors[task]

    def set_active(self, task: str) -> None:
        self._active = task

    def active_module(self) -> ProjectorMLP:
        if self._active is None:
            raise RuntimeError("projector library has no active task")
        return self.projectors[self._active]

    def set_trainable(self, task: str, flag: bool) -> None:
        for param in self.projectors[task].parameters():
            param.requires_grad_(flag)

    def freeze_task(self, task: str) -> None:
        self.set_trainable(task, False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.active_module()(features)


class TaskKeyStore(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.keys = nn.ParameterDict()
        self.centroids = nn.ParameterDict()

    def allocate(self, task: str) -> None:
        if task in self.keys:
            return
        self.keys[task] = nn.Parameter(torch.zeros(self.embed_dim, dtype=torch.float32))
        self.centroids[task] = nn.Parameter(
            torch.zeros(self.embed_dim, dtype=torch.float32), requires_grad=False
        )

    def key(self, task: str) -> nn.Parameter:
        return self.keys[task]

    def centroid(self, task: str) -> torch.Tensor:
        return self.centroids[task].data

    def set_centroid(self, task: str, value: torch.Tensor) -> None:
        with torch.no_grad():
            self.centroids[task].copy_(value.to(self.centroids[task].dtype))

    def set_trainable(self, task: str, flag: bool) -> None:
        self.keys[task].requires_grad_(flag)

    def freeze_key(self, task: str) -> None:
        self.set_trainable(task, False)

    def cosine_retrieve(self, embedding: torch.Tensor) -> int:
        query = embedding.reshape(-1).to(torch.float32)
        best_task = -1
        best_score = float("-inf")
        for task, key in self.keys.items():
            score = float(F.cosine_similarity(query, key.reshape(-1).to(torch.float32), dim=0))
            if score > best_score:
                best_score = score
                best_task = int(task)
        return best_task
