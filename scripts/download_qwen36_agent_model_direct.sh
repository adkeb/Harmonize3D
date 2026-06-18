#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
LOCAL_DIR="${2:-/root/sakura/models/Qwen3.6-35B-A3B-NVFP4}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="*"
export no_proxy="*"
export HF_ENDPOINT
export HF_HUB_ENABLE_HF_TRANSFER=0

mkdir -p "$LOCAL_DIR"

echo "Direct model API preflight: ${HF_ENDPOINT}/api/models/${MODEL_ID}"
if command -v curl >/dev/null 2>&1; then
  curl -I --noproxy '*' --connect-timeout 10 "${HF_ENDPOINT}/api/models/${MODEL_ID}" | sed -n '1,12p'
fi

if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL_ID" --local-dir "$LOCAL_DIR" --local-dir-use-symlinks False --resume-download
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL_DIR" --local-dir-use-symlinks False --resume-download
elif [[ -x tools/hfd.sh ]]; then
  tools/hfd.sh "$MODEL_ID" --local-dir "$LOCAL_DIR" --tool aria2c -x 8
else
  echo "ERROR: install huggingface_hub CLI or keep tools/hfd.sh executable." >&2
  exit 1
fi

echo "Model files ready at: ${LOCAL_DIR}"
echo "Start with: LOCAL_MODEL_DIR=\"${LOCAL_DIR}\" scripts/start_qwen36_agent_service.sh"
