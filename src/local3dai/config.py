from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "system": {
        "workspace": "",
        "os": "",
        "python": "",
        "gpu_name": "",
        "gpu_vram_mb": 0,
        "nvidia_driver": "",
        "cuda_runtime_reported_by_driver": "",
        "blender_path": "",
    },
    "paths": {
        "models_dir": "models",
        "outputs_dir": "outputs",
        "blender_script": "blender_scripts/batch_render.py",
    },
    "render": {
        "resolution": 1024,
        "views": 8,
        "engine": "CYCLES",
        "samples": 64,
        "camera_distance": 3.2,
    },
    "model_generation": {
        "default_backend": "sample",
        "external_command": "",
        "output_dir": "outputs/generated_models",
    },
    "ai": {
        "default_backend": "mock",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "candidates_per_view": 2,
        "steps": 24,
        "guidance_scale": 3.5,
        "strength": 0.62,
        "seed": 20260610,
    },
    "models": {},
    "score": {
        "edge_weight": 0.55,
        "mask_weight": 0.25,
        "prompt_weight": 0.2,
        "copy_top_k": 1,
    },
    "agent": {
        "enabled": True,
        "max_generations": 10,
        "target_view": "view_05",
        "expand_view_ids": ["view_01", "view_05", "view_06"],
        "default_reference_channels": ["rgb", "edge"],
        "experimental_reference_channels": [],
        "roughness_weight": 0.25,
        "edge_weight": 0.35,
        "mask_weight": 0.25,
        "background_weight": 0.15,
        "pass_threshold": 0.62,
    },
}


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        user_config = json.load(fh)
    return deep_merge(DEFAULT_CONFIG, user_config)


def write_config(path: str | Path, config: dict[str, Any], *, force: bool = False) -> Path:
    config_path = Path(path)
    if config_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing config: {config_path}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return config_path


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    workspace = config.get("system", {}).get("workspace") or Path.cwd()
    return Path(workspace) / path
