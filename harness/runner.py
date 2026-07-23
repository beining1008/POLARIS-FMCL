from __future__ import annotations

import random
from typing import List

import torch

from data.benchmarks import BENCHMARKS, build_client_loaders, build_eval_loader
from harness.config import ExperimentConfig
from harness.interfaces import ClientPayload
from harness.metrics import AccuracyMatrix
from harness.registry import build_method
from models.backbone import BACKBONES, generate_answer, load_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FederatedContinualRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.backbone_spec = BACKBONES[config.backbone]
        self.tasks = BENCHMARKS[config.benchmark]
        self.method = build_method(config.method, config, self.backbone_spec)
        self.matrix = AccuracyMatrix(len(self.tasks))

    def run(self) -> AccuracyMatrix:
        cfg = self.config
        set_seed(cfg.seed)
        tokenizer = load_tokenizer(self.backbone_spec)
        models = [self.method.build_model() for _ in range(cfg.num_clients)]
        server_model = self.method.build_model()
        self.matrix.zero_shot = [
            self.evaluate(server_model, tokenizer, task) for task in self.tasks
        ]

        for task_id, task in enumerate(self.tasks):
            loaders = build_client_loaders(
                task,
                num_clients=cfg.num_clients,
                beta=cfg.dirichlet_beta,
                batch_size=cfg.batch_size,
                seed=cfg.seed,
                tokenizer=tokenizer,
            )
            self.method.on_task_start(task_id)
            payloads: List[ClientPayload] = []
            for round_id in range(cfg.rounds_per_task):
                broadcast = self.method.current_broadcast()
                payloads = [
                    self.method.local_update(
                        client_id, task_id, round_id, models[client_id], loaders[client_id], broadcast
                    )
                    for client_id in range(cfg.num_clients)
                ]
                package = self.method.aggregate(payloads, task_id, round_id)
                self.method.set_broadcast(package)
            self.method.on_task_end(task_id, payloads)
            self._sync_server_model(server_model)
            for eval_task_id in range(len(self.tasks)):
                accuracy = self.evaluate(server_model, tokenizer, self.tasks[eval_task_id])
                self.matrix.record(task_id, eval_task_id, accuracy)
        return self.matrix

    def _sync_server_model(self, server_model: torch.nn.Module) -> None:
        broadcast = self.method.current_broadcast()
        if broadcast is None:
            return
        from harness.aggregation import load_weights

        load_weights(server_model, broadcast.weights)

    def evaluate(self, model: torch.nn.Module, tokenizer, task: str) -> float:
        loader = build_eval_loader(task, batch_size=self.config.batch_size, tokenizer=tokenizer)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for sample in loader:
                prediction = generate_answer(model, tokenizer, sample["image"], sample["question"])
                reference = sample["answer"]
                if self._match(prediction, reference, task):
                    correct += 1
                total += 1
        return 100.0 * correct / max(total, 1)

    @staticmethod
    def _match(prediction: str, reference: str, task: str) -> bool:
        prediction = prediction.strip().lower()
        reference = reference.strip().lower()
        if task in ("scienceqa", "imagenet", "iconqa", "ai2d"):
            return prediction == reference
        return reference in prediction


def run_experiment(config: ExperimentConfig) -> dict:
    runner = FederatedContinualRunner(config)
    matrix = runner.run()
    return matrix.summary()
