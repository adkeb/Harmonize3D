from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return manifest_path
