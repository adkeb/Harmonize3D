#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p outputs/download_logs
log="outputs/download_logs/zimage_i2l_download.log"
pidfile="outputs/download_logs/zimage_i2l_download.pid"

rm -f "$log" "$pidfile"
nohup scripts/download_zimage_i2l_assets.sh > "$log" 2>&1 &
echo "$!" > "$pidfile"
echo "$!"
