from __future__ import annotations

import shutil
import subprocess
import json
from pathlib import Path
from typing import Any

from PIL import Image


REQUIRED_CHANNELS = ("rgb", "depth", "edge", "normal", "mask")


def _image_dynamic_range(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        extrema = image.convert("L").getextrema()
    return int(extrema[0]), int(extrema[1])


def validate_render_manifest(manifest: Path) -> None:
    with manifest.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    for view in data.get("views", []):
        files = view.get("files", {})
        for channel in REQUIRED_CHANNELS:
            path = Path(files.get(channel, ""))
            if not path.exists():
                raise RuntimeError(f"Missing Blender render channel {channel!r}: {path}")
        for channel in ("rgb", "mask", "normal"):
            low, high = _image_dynamic_range(Path(files[channel]))
            if high - low < 8 and high <= 8:
                raise RuntimeError(
                    f"Blender channel {channel!r} appears empty or affected by broken color management: {files[channel]}"
                )


def render_model_with_blender(
    *,
    model_path: str | Path,
    output_dir: str | Path,
    blender_script: str | Path,
    blender_path: str | None = None,
    views: int = 8,
    resolution: int = 1024,
    resolution_x: int | None = None,
    resolution_y: int | None = None,
    engine: str = "CYCLES",
    samples: int = 64,
    camera_distance: float = 3.2,
    camera: dict[str, Any] | None = None,
) -> Path:
    executable = blender_path or shutil.which("blender")
    if not executable:
        raise RuntimeError("Blender was not found in PATH. Install Blender or set system.blender_path in configs/local.json.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "--background",
        "--python",
        str(blender_script),
        "--",
        "--model",
        str(model_path),
        "--output",
        str(output),
        "--views",
        str(views),
        "--resolution",
        str(resolution),
        "--engine",
        engine,
        "--samples",
        str(samples),
        "--camera-distance",
        str(camera_distance),
    ]
    if resolution_x and resolution_y:
        command.extend(["--resolution-x", str(resolution_x), "--resolution-y", str(resolution_y)])
    if camera is not None:
        command.extend(["--camera-json", json.dumps(camera)])
    subprocess.run(command, check=True)
    manifest = output / "manifest.json"
    if not manifest.exists():
        raise RuntimeError(f"Blender finished without writing the expected manifest: {manifest}")
    validate_render_manifest(manifest)
    return manifest
