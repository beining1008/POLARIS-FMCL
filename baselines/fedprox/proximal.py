from __future__ import annotations

from typing import Dict, Iterable

import torch


class ProximalAdamW(torch.optim.AdamW):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        weight_decay: float,
        proximal_mu: float,
    ) -> None:
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.proximal_mu = float(proximal_mu)
        self._anchors: Dict[int, torch.Tensor] = {}

    def set_anchors(self, anchors: Dict[int, torch.Tensor]) -> None:
        self._anchors = anchors

    def perturbed_gradient_step(self) -> None:
        if self.proximal_mu == 0.0 or not self._anchors:
            return
        with torch.no_grad():
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    anchor = self._anchors.get(id(param))
                    if anchor is None:
                        continue
                    drift = param.detach().to(torch.float32) - anchor
                    param.grad.add_(drift.to(param.grad.dtype), alpha=self.proximal_mu)

    def step(self, closure=None):
        self.perturbed_gradient_step()
        return super().step(closure)
