#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/flux2_klein_high_quality_car_reference/zimage_i2l_anchor_probe

exec > outputs/flux2_klein_high_quality_car_reference/zimage_i2l_anchor_probe/run.log 2>&1
date -Is
export PATH="/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader || true

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

.venv/bin/python3 scripts/probe_zimage_i2l_lora.py \
  --output outputs/flux2_klein_high_quality_car_reference/zimage_i2l_anchor_probe \
  --generate \
  --samples 1 \
  --steps 32 \
  --width 768 \
  --height 768 \
  --seed 20260614 \
  --offload-device cpu

date -Is
