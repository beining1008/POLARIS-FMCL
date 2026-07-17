from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from harness.interfaces import ClientPayload, TensorDict


def client_weights(payloads: List[ClientPayload]) -> List[float]:
    total = float(sum(p.num_samples for p in payloads))
    return [p.num_samples / total for p in payloads]


def fedavg(payloads: List[ClientPayload], keys: Optional[Iterable[str]] = None) -> TensorDict:
    mix = client_weights(payloads)
    reference = payloads[0].weights
    names = list(keys) if keys is not None else list(reference.keys())
    merged: TensorDict = {}
    for name in names:
        accumulator = torch.zeros_like(reference[name], dtype=torch.float32)
        for w, payload in zip(mix, payloads):
            accumulator.add_(payload.weights[name].to(torch.float32), alpha=w)
        merged[name] = accumulator.to(reference[name].dtype)
    return merged


def extract_trainable(model: nn.Module) -> TensorDict:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_weights(model: nn.Module, weights: TensorDict, strict: bool = False) -> None:
    with torch.no_grad():
        named = dict(model.named_parameters())
        for name, value in weights.items():
            if name in named:
                named[name].copy_(value.to(named[name].dtype))
            elif strict:
                raise KeyError(name)
