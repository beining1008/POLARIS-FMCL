from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn as nn

from harness.interfaces import TensorDict


def trainable_named_parameters(model: nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


def zeros_like_trainable(model: nn.Module) -> TensorDict:
    return {
        name: torch.zeros_like(param, dtype=torch.float32)
        for name, param in trainable_named_parameters(model)
    }


def diagonal_fisher(
    model: nn.Module,
    loader: Iterable,
    forward_loss: Callable[[nn.Module, Any], torch.Tensor],
    max_batches: int,
) -> TensorDict:
    accumulator = zeros_like_trainable(model)
    prior_mode = model.training
    model.eval()
    batches = 0
    for batch in loader:
        if batches >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        loss = forward_loss(model, batch)
        loss.backward()
        for name, param in trainable_named_parameters(model):
            if param.grad is None:
                continue
            squared = param.grad.detach().to(torch.float32).pow(2)
            accumulator[name].add_(squared)
        batches += 1
    model.zero_grad(set_to_none=True)
    model.train(prior_mode)
    denominator = float(max(batches, 1))
    for name in accumulator:
        accumulator[name].div_(denominator)
    return accumulator


def accumulate_fisher(
    running: Optional[TensorDict],
    estimate: TensorDict,
    enabled: bool,
) -> TensorDict:
    if running is None or not enabled:
        return {name: value.clone() for name, value in estimate.items()}
    merged: TensorDict = dict(running)
    for name, value in estimate.items():
        if name in merged:
            merged[name] = merged[name] + value
        else:
            merged[name] = value.clone()
    return merged


def consolidation_penalty(
    model: nn.Module,
    anchor: Optional[TensorDict],
    fisher: Optional[TensorDict],
    ewc_lambda: float,
    device: torch.device,
) -> torch.Tensor:
    zero = torch.zeros((), dtype=torch.float32, device=device)
    if anchor is None or not fisher:
        return zero
    total = zero
    for name, param in trainable_named_parameters(model):
        if name not in anchor or name not in fisher:
            continue
        theta_star = anchor[name].to(device=param.device, dtype=torch.float32)
        importance = fisher[name].to(device=param.device, dtype=torch.float32)
        delta = param.to(torch.float32) - theta_star
        term = (importance * delta.pow(2)).sum()
        total = total + term.to(device)
    return 0.5 * ewc_lambda * total


def fisher_trace(fisher: Optional[TensorDict]) -> float:
    if not fisher:
        return 0.0
    total = 0.0
    for value in fisher.values():
        total += float(value.sum())
    return total
