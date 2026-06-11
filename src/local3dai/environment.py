from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PYTHON_MODULES = [
    "numpy",
    "PIL",
    "cv2",
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
]


def _run(command: list[str], timeout: int = 8) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return completed.returncode, completed.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def detect_gpu() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False}

    query = [
        nvidia_smi,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    code, output = _run(query)
    info: dict[str, Any] = {"available": code == 0, "nvidia_smi": nvidia_smi}
    if code == 0 and output:
        first = output.splitlines()[0]
        parts = [part.strip() for part in first.split(",")]
        if len(parts) >= 3:
            info.update(
                {
                    "name": parts[0],
                    "memory_total_mb": int(float(parts[1])),
                    "driver_version": parts[2],
                }
            )

    code, full_output = _run([nvidia_smi])
    match = re.search(r"CUDA UMD Version:\s*([0-9.]+)", full_output)
    if match:
        info["cuda_runtime_reported_by_driver"] = match.group(1)
    return info


def detect_modules() -> dict[str, dict[str, Any]]:
    modules: dict[str, dict[str, Any]] = {}
    for name in PYTHON_MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None:
            modules[name] = {"available": False}
            continue
        try:
            module = __import__(name)
            modules[name] = {
                "available": True,
                "version": getattr(module, "__version__", ""),
            }
        except Exception as exc:  # pragma: no cover - defensive import reporting
            modules[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return modules


def find_local_model_candidates(search_roots: list[Path] | None = None, limit: int = 40, max_depth: int = 5) -> list[str]:
    env_roots = os.environ.get("LOCAL3DAI_MODEL_SCAN_ROOTS")
    if env_roots:
        roots = [Path(item) for item in env_roots.split(os.pathsep) if item]
    else:
        roots = search_roots or [Path.cwd(), Path("/root/sakura/work"), Path("/mnt/c/Users"), Path("/mnt/d"), Path("/mnt/e")]
    patterns = ("*.safetensors", "*.ckpt", "*.bin", "*.gguf")
    skip_dirs = {".cache", ".git", "node_modules", "__pycache__", "AppData", "Windows", "Program Files", "Program Files (x86)"}
    candidates: list[str] = []

    def walk(root: Path, depth: int) -> None:
        nonlocal candidates
        if len(candidates) >= limit or depth > max_depth:
            return
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    if len(candidates) >= limit:
                        return
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name in skip_dirs:
                            continue
                        walk(Path(entry.path), depth + 1)
                    elif entry.is_file(follow_symlinks=False):
                        path = Path(entry.path)
                        if not any(path.match(pattern) for pattern in patterns):
                            continue
                        name = path.name.lower()
                        parent = str(path.parent).lower()
                        if any(token in name or token in parent for token in ("flux", "sd3", "stable", "diffusion")):
                            candidates.append(str(path))
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            return

    for root in roots:
        walk(root, 0)
        if len(candidates) >= limit:
            break
    return candidates


def detect_environment(*, include_model_scan: bool = False) -> dict[str, Any]:
    blender = shutil.which("blender")
    uv = shutil.which("uv")
    report: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "cwd": str(Path.cwd()),
        },
        "tools": {
            "uv": uv or "",
            "blender": blender or "",
        },
        "gpu": detect_gpu(),
        "python_modules": detect_modules(),
    }
    if include_model_scan:
        report["local_model_candidates"] = find_local_model_candidates()
    return report


def print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))
