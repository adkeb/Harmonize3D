#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="*"
export no_proxy="*"

download_file() {
  local url="$1"
  local output_dir="$2"
  local output_name="$3"
  local expected_min_bytes="$4"

  mkdir -p "$output_dir"
  if [[ -f "$output_dir/$output_name" ]]; then
    local size
    size=$(stat -c '%s' "$output_dir/$output_name")
    if [[ "$size" -ge "$expected_min_bytes" ]]; then
      echo "Already downloaded: $output_dir/$output_name ($size bytes)"
      return
    fi
  fi

  echo "Downloading: $url"
  aria2c \
    --continue=true \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --max-connection-per-server=4 \
    --split=4 \
    --min-split-size=64M \
    --summary-interval=30 \
    --console-log-level=notice \
    --download-result=hide \
    --dir "$output_dir" \
    --out "$output_name" \
    "$url"
}

download_file \
  "https://hf-mirror.com/DiffSynth-Studio/Z-Image-i2L/resolve/main/model.safetensors" \
  "models/Z-Image-i2L" \
  "model.safetensors" \
  3226000000

download_file \
  "https://hf-mirror.com/DiffSynth-Studio/General-Image-Encoders/resolve/main/SigLIP2-G384/model.safetensors" \
  "models/General-Image-Encoders/SigLIP2-G384" \
  "model.safetensors" \
  4654000000

download_file \
  "https://hf-mirror.com/DiffSynth-Studio/General-Image-Encoders/resolve/main/DINOv3-7B/model.safetensors" \
  "models/General-Image-Encoders/DINOv3-7B" \
  "model.safetensors" \
  13400000000
