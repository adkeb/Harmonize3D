#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BLENDER_VERSION="${BLENDER_VERSION:-5.1.2}"
BLENDER_SERIES="${BLENDER_SERIES:-5.1}"
ARCHIVE="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
URL="https://download.blender.org/release/Blender${BLENDER_SERIES}/${ARCHIVE}"
TARGET_DIR="tools/blender-${BLENDER_VERSION}-linux-x64"
CONFIG_PATH="${CONFIG_PATH:-configs/local.json}"

mkdir -p tools
if [ ! -f "tools/${ARCHIVE}" ]; then
  wget -c "$URL" -O "tools/${ARCHIVE}"
fi

if [ ! -x "${TARGET_DIR}/blender" ]; then
  tar -xf "tools/${ARCHIVE}" -C tools
fi

python3 - <<PY
import json
from pathlib import Path

config_path = Path("${CONFIG_PATH}")
config = json.loads(config_path.read_text(encoding="utf-8"))
config.setdefault("system", {})["blender_path"] = str((Path.cwd() / "${TARGET_DIR}" / "blender").resolve())
config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Configured Blender:", config["system"]["blender_path"])
PY

"${TARGET_DIR}/blender" --version | sed -n '1,4p'
