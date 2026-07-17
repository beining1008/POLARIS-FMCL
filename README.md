# POLARIS

Code for the paper *Exposing and Resolving Spurious Isolation in Federated Multimodal Continual Learning* (under review at IEEE Transactions on Multimedia).

POLARIS (Per-expert Orthogonal Landscape-Aware Routing-Informed Subspace protection) maintains a per-expert protection basis in the gradient subspace from routing-weighted covariance, merges client bases through an orthogonality-preserving federated union (PE-FOSU), reads per-layer protection budgets from an online interference landscape through capped water-filling, and anneals projection strength during each task's warmup window.

This is a method-level release. Backbone loading, tokenizers, and dataset loaders are interface stubs (`NotImplementedError`); every method-level component is implemented in full. There are no run scripts.

## Layout

```
polaris/     POLARIS: routing-weighted covariance, PE-FOSU union, bilateral projection,
             online interference estimation, capped water-filling scheduler, training lifecycle
harness/     shared federated continual-learning harness: method interface, FedAvg,
             task-sequence runner, AA/BWT/FWT metrics, method registry
models/      backbone specs and stubs; shared MoE-LoRA adapter (experts, routers, injection)
data/        CoIN-6 and CoIN-Long-10 task streams (stub loaders); Dirichlet partitioner
baselines/   the fourteen comparison methods, one directory each
```

Every method implements the same interface (`harness/interfaces.py`) and runs under the same protocol: C = 5 clients, Dirichlet partition, one FedAvg round per task with one local epoch per client, AdamW at 2e-4 under cosine decay. POLARIS trains E = 4 experts at rank r = 8 with top-1 routing; every baseline matches this adapter budget, and a baseline whose paper publishes its own routing or adapter form (dense gating in Fed-MoELoRA, the rank pool of Fed-PCLR, the dual pathways of Fed-MoDE and Fed-Duet) keeps that form at the same budget.

## Baselines

Non-federated methods are federated by applying FedAvg to their local update rule with the authors' default hyperparameters; every method uses the same adapter budget as POLARIS.

| Baseline | Venue | Code | Federated adaptation |
|---|---|---|---|
| FedProx | MLSys 2020 | adapted | native federated |
| Fed-EWC | PNAS 2017 | re-impl. | FedAvg on local EWC update |
| Fed-LwF | ECCV 2016 | re-impl. | FedAvg on local LwF update |
| Fed-Replay | CVPR 2017 | re-impl. | FedAvg on local replay update |
| Fed-GPM | ICLR 2021 | adapted | FedAvg on local GPM update |
| Fed-O-LoRA | EMNLP 2023 | adapted | FedAvg on local O-LoRA update |
| Fed-KeepLoRA | ICLR 2026 | adapted | FedAvg on local KeepLoRA update |
| Fed-SplitLoRA | ICLR 2026 | adapted | FedAvg on local SplitLoRA update |
| Fed-MoELoRA | NeurIPS 2024 | adapted | FedAvg on local MoELoRA update |
| Fed-SMoLoRA | ICCV 2025 | adapted | FedAvg on local SMoLoRA update |
| Fed-MoDE | NeurIPS 2025 | re-impl. | FedAvg on local MoDE update |
| Fed-PCLR | ICLR 2026 | adapted | FedAvg on local PCLR update |
| Fed-EProj | TIP 2026 | re-impl. | FedAvg on local EProj update |
| Fed-Duet | ICLR 2026 | adapted | native federated (protocol source) |

### Provenance and adaptation notes

- **FedProx** (MLSys 2020): adapted from the official release (`github.com/litian96/FedProx`, MIT). Native federated: the proximal term enters each client's local objective under the shared round schedule.
- **Fed-EWC** (PNAS 2017): re-implemented; the authors released no code. Fisher information is estimated client-locally at each task boundary and the quadratic penalty enters the local objective; aggregation remains plain FedAvg on the adapter weights.
- **Fed-LwF** (ECCV 2016): re-implemented; the official release is MATLAB/MatConvNet. The distillation teacher is the frozen task-boundary global snapshot and stays client-local.
- **Fed-Replay** (CVPR 2017): re-implemented; the official iCaRL release is marked non-runnable by its authors. Each client keeps an exemplar buffer drawn from its own partition; no exemplar leaves a client.
- **Fed-GPM** (ICLR 2021): adapted from the official release (`github.com/sahagobinda/GPM`, MIT). The projection basis is maintained client-locally as published, and FedAvg aggregates adapter weights only.
- **Fed-O-LoRA** (EMNLP 2023): adapted from the official release (`github.com/cmnfriend/O-LoRA`, MIT). The orthogonality regularizer is applied client-locally; the frozen subspace stack is folded server-side and broadcast at task boundaries.
- **Fed-KeepLoRA** (ICLR 2026): adapted from the official release (`github.com/MaolinLuo/KeepLoRA`, MIT). The residual gradient adaptation runs client-locally, and no basis crosses clients.
- **Fed-SplitLoRA** (ICLR 2026): adapted from the official release (`github.com/iLearn-Lab/ICLR26-SplitLoRA`, Apache-2.0). The gradient-space split is computed client-locally.
- **Fed-MoELoRA** (NeurIPS 2024): adapted from the CoIN codebase (`github.com/zackschen/CoIN`, Apache-2.0). The MoE-LoRA mechanics are used as published at the shared budget, with the router included in FedAvg.
- **Fed-SMoLoRA** (ICCV 2025): adapted from the official release (`github.com/Minato-Zackie/SMoLoRA`, Apache-2.0). Same shared budget; FedAvg over all trainable parameters.
- **Fed-MoDE** (NeurIPS 2025): re-implemented; the official repository contains no implementation code at the time of writing. The paper instantiates on unified generative models; this harness exercises the understanding tasks.
- **Fed-PCLR** (ICLR 2026): adapted from the official release (`github.com/SII-HITclearlove777/PCLR`; the release declares Apache-2.0 in its README and ships no license file). The release is single-node continual instruction tuning, so FedAvg applies to the local PCLR update, and the rank-pool compression runs server-side at each task boundary.
- **Fed-EProj** (TIP 2026): re-implemented; the authors released no code. FedAvg applies to the local EProj update; historical projectors stay frozen on all clients.
- **Fed-Duet** (ICLR 2026): the official release (`github.com/cocogt96/Fed-Duet`) is natively federated and is the protocol source for this comparison. That repository carries no license, so no code from it is redistributed here; `baselines/fed_duet/` is an independent implementation of the published mechanism written for this harness.

The code in `baselines/` follows the division stated in each note: what a note declares client-local stays client-local in the code, and what crosses clients is exactly what FedAvg aggregates.

## License

MIT (see `LICENSE`). The licenses of adapted upstream releases are noted above and apply to the respective original repositories.
