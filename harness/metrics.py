from __future__ import annotations

from typing import List, Optional


class AccuracyMatrix:
    def __init__(self, num_tasks: int) -> None:
        self.num_tasks = num_tasks
        self.values: List[List[Optional[float]]] = [
            [None for _ in range(num_tasks)] for _ in range(num_tasks)
        ]
        self.zero_shot: Optional[List[float]] = None

    def record(self, after_task: int, eval_task: int, accuracy: float) -> None:
        self.values[after_task][eval_task] = accuracy

    def entry(self, after_task: int, eval_task: int) -> float:
        value = self.values[after_task][eval_task]
        if value is None:
            raise ValueError(f"missing entry ({after_task}, {eval_task})")
        return value

    def average_accuracy(self) -> float:
        final = self.num_tasks - 1
        return sum(self.entry(final, j) for j in range(self.num_tasks)) / self.num_tasks

    def backward_transfer(self) -> float:
        final = self.num_tasks - 1
        deltas = [
            self.entry(final, j) - self.entry(j, j) for j in range(self.num_tasks - 1)
        ]
        return sum(deltas) / (self.num_tasks - 1)

    def forward_transfer(self) -> float:
        baseline = self.zero_shot if self.zero_shot is not None else [0.0] * self.num_tasks
        gains = [self.entry(j - 1, j) - baseline[j] for j in range(1, self.num_tasks)]
        return sum(gains) / (self.num_tasks - 1)

    def summary(self) -> dict:
        return {
            "AA": self.average_accuracy(),
            "BWT": self.backward_transfer(),
            "FWT": self.forward_transfer(),
        }
