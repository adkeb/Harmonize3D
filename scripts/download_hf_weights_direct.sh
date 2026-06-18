#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${1:-stabilityai/sd-turbo}"
LOCAL_DIR="${2:-models/sd-turbo}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  shift
fi
DOWNLOAD_ARGS=("$@")

if [[ ${#DOWNLOAD_ARGS[@]} -eq 0 && "$MODEL_ID" == "stabilityai/sd-turbo" ]]; then
  DOWNLOAD_ARGS=(
    --tool wget
    --include
    "model_index.json"
    "scheduler/*"
    "tokenizer/*"
    "text_encoder/config.json"
    "text_encoder/model.fp16.safetensors"
    "unet/config.json"
    "unet/diffusion_pytorch_model.fp16.safetensors"
    "vae/config.json"
    "vae/diffusion_pytorch_model.fp16.safetensors"
  )
fi

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="*"
export no_proxy="*"
export HF_ENDPOINT
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

source .venv/bin/activate
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python3}"

"$PYTHON_BIN" - <<'PY'
import os
for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    if os.environ.get(key):
        raise SystemExit(f"Proxy variable is still set: {key}")
print("HF_ENDPOINT=", os.environ.get("HF_ENDPOINT"))
print("NO_PROXY=", os.environ.get("NO_PROXY"))
PY

echo "Direct endpoint preflight (no proxy): $HF_ENDPOINT"
curl -I --noproxy '*' --connect-timeout 10 "$HF_ENDPOINT" | sed -n '1,12p'
echo "Direct model API preflight (no proxy): $MODEL_ID"
curl -I --noproxy '*' --connect-timeout 10 "$HF_ENDPOINT/api/models/$MODEL_ID" | sed -n '1,12p'

if [[ -x tools/hfd.sh ]]; then
  tools/hfd.sh "$MODEL_ID" \
    --local-dir "$LOCAL_DIR" \
    "${DOWNLOAD_ARGS[@]}"
else
  hf download "$MODEL_ID" \
    --local-dir "$LOCAL_DIR"
fi

echo "Downloaded $MODEL_ID to $LOCAL_DIR"
