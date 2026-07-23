#!/usr/bin/env bash
set -euo pipefail

backbone="${1:-llava-1.5-7b}"
benchmark="${2:-coin6}"
beta="${3:-0.3}"

methods="polaris fed-duet fed-pclr fed-eproj fed-mode fed-smolora fed-moelora fed-splitlora fed-keeplora fed-olora fed-gpm fed-replay fed-lwf fed-ewc fedprox"

for seed in 0 1 2; do
  for method in ${methods}; do
    python train.py --method "${method}" --backbone "${backbone}" --benchmark "${benchmark}" --beta "${beta}" --seed "${seed}"
  done
done
