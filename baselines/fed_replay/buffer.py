from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn

SampleRecord = Dict[str, torch.Tensor]


def extract_sample_features(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    raise NotImplementedError


def split_batch(batch: Mapping[str, torch.Tensor]) -> List[SampleRecord]:
    keys = list(batch.keys())
    if not keys:
        return []
    size = int(batch[keys[0]].shape[0])
    records: List[SampleRecord] = []
    for index in range(size):
        record = {key: batch[key][index].detach().to("cpu").clone() for key in keys}
        records.append(record)
    return records


def collate_records(records: List[SampleRecord]) -> Dict[str, torch.Tensor]:
    keys = list(records[0].keys())
    return {key: torch.stack([record[key] for record in records], dim=0) for key in keys}


class ExemplarBuffer:
    def __init__(self, budget: int, feature_batches: int) -> None:
        self.budget = int(budget)
        self.feature_batches = int(feature_batches)
        self.slots_per_task = int(budget)
        self.exemplars: Dict[int, List[SampleRecord]] = {}

    def __len__(self) -> int:
        return sum(len(records) for records in self.exemplars.values())

    def is_empty(self) -> bool:
        return len(self) == 0

    def task_counts(self) -> Dict[int, int]:
        return {task_id: len(records) for task_id, records in self.exemplars.items()}

    def rebalance(self, tasks_seen: int) -> None:
        self.slots_per_task = self.budget // max(tasks_seen, 1)
        for task_id in list(self.exemplars.keys()):
            prefix = self.exemplars[task_id][: self.slots_per_task]
            self.exemplars[task_id] = prefix

    def herding_select(
        self,
        features: torch.Tensor,
        task_mean: torch.Tensor,
        slots: int,
    ) -> List[int]:
        features = features.to(torch.float32)
        task_mean = task_mean.to(torch.float32)
        total = int(features.shape[0])
        slots = min(int(slots), total)
        selected: List[int] = []
        selected_sum = torch.zeros_like(task_mean)
        available = torch.ones(total, dtype=torch.bool)
        for _ in range(slots):
            count = len(selected)
            candidate_means = (selected_sum.unsqueeze(0) + features) / float(count + 1)
            distances = torch.linalg.norm(task_mean.unsqueeze(0) - candidate_means, dim=1)
            distances = distances.masked_fill(~available, float("inf"))
            best = int(torch.argmin(distances))
            selected.append(best)
            available[best] = False
            selected_sum = selected_sum + features[best]
        return selected

    def fill_from_shard(self, model: nn.Module, loader: Iterable, task_id: int) -> None:
        feature_chunks: List[torch.Tensor] = []
        records: List[SampleRecord] = []
        for index, batch in enumerate(loader):
            if index >= self.feature_batches:
                break
            features = extract_sample_features(model, batch).detach().to(torch.float32)
            feature_chunks.append(features)
            records.extend(split_batch(batch))
        if not feature_chunks:
            self.exemplars[task_id] = []
            return
        stacked = torch.cat(feature_chunks, dim=0)
        task_mean = stacked.mean(dim=0)
        slots = min(self.slots_per_task, int(stacked.shape[0]))
        chosen = self.herding_select(stacked, task_mean, slots)
        self.exemplars[task_id] = [records[index] for index in chosen]

    def sample(self, k: int) -> Optional[Dict[str, torch.Tensor]]:
        if k <= 0:
            return None
        pool: List[SampleRecord] = [
            record for records in self.exemplars.values() for record in records
        ]
        if not pool:
            return None
        k = min(k, len(pool))
        chosen = random.sample(pool, k)
        return collate_records(chosen)
