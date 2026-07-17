from __future__ import annotations

import importlib
from typing import Dict, Tuple

_METHODS: Dict[str, Tuple[str, str]] = {
    "polaris": ("polaris.method", "Polaris"),
    "fedprox": ("baselines.fedprox.method", "FedProx"),
    "fed-ewc": ("baselines.fed_ewc.method", "FedEWC"),
    "fed-lwf": ("baselines.fed_lwf.method", "FedLwF"),
    "fed-replay": ("baselines.fed_replay.method", "FedReplay"),
    "fed-gpm": ("baselines.fed_gpm.method", "FedGPM"),
    "fed-olora": ("baselines.fed_olora.method", "FedOLoRA"),
    "fed-keeplora": ("baselines.fed_keeplora.method", "FedKeepLoRA"),
    "fed-splitlora": ("baselines.fed_splitlora.method", "FedSplitLoRA"),
    "fed-moelora": ("baselines.fed_moelora.method", "FedMoELoRA"),
    "fed-smolora": ("baselines.fed_smolora.method", "FedSMoLoRA"),
    "fed-mode": ("baselines.fed_mode.method", "FedMoDE"),
    "fed-pclr": ("baselines.fed_pclr.method", "FedPCLR"),
    "fed-eproj": ("baselines.fed_eproj.method", "FedEProj"),
    "fed-duet": ("baselines.fed_duet.method", "FedDuet"),
}


def available_methods() -> Tuple[str, ...]:
    return tuple(sorted(_METHODS.keys()))


def build_method(name: str, config, backbone_spec):
    key = name.lower()
    if key not in _METHODS:
        raise KeyError(f"unknown method '{name}', available: {available_methods()}")
    module_path, class_name = _METHODS[key]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(config, backbone_spec)
