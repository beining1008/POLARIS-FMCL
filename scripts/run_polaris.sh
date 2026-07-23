#!/usr/bin/env bash
set -euo pipefail

backbone="${1:-llava-1.5-7b}"
benchmark="${2:-coin6}"
beta="${3:-0.3}"

for seed in 0 1 2; do
  python train.py --method polaris --backbone "${backbone}" --benchmark "${benchmark}" --beta "${beta}" --seed "${seed}"
done
