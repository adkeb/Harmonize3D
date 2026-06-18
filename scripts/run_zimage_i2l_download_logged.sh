#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/download_logs

exec > outputs/download_logs/zimage_i2l_download.log 2>&1
date -Is
scripts/download_zimage_i2l_assets.sh
date -Is
