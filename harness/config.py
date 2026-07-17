from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    backbone: str = "llava-1.5-7b"
    benchmark: str = "coin6"
    method: str = "polaris"
    num_experts: int = 4
    lora_rank: int = 8
    lora_alpha: float = 16.0
    top_k: int = 1
    routing: str = "topk"
    num_clients: int = 5
    dirichlet_beta: float = 0.3
    rounds_per_task: int = 1
    local_epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-4
    lr_schedule: str = "cosine"
    weight_decay: float = 0.0
    max_seq_len: int = 2048
    dtype: str = "float16"
    seed: int = 0

    def resolve_dtype(self):
        import torch

        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[self.dtype]
