from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from harness.aggregation import extract_trainable, load_weights
from harness.interfaces import TensorDict


def extract_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs


class TeacherContext:
    def __init__(self, model: nn.Module, teacher_weights: TensorDict) -> None:
        self.model = model
        self.teacher_weights = teacher_weights
        self._student_weights: Optional[TensorDict] = None
        self._was_training = False

    def __enter__(self) -> "TeacherContext":
        self._student_weights = extract_trainable(self.model)
        self._was_training = self.model.training
        load_weights(self.model, self.teacher_weights)
        self.model.eval()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._student_weights is not None:
            load_weights(self.model, self._student_weights)
        self._student_weights = None
        self.model.train(self._was_training)
        return False

    @torch.no_grad()
    def logits(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**inputs)
        return extract_logits(outputs).detach()
