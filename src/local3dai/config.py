from __future__ import annotations

import copy
import json
import os
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
        "default_model_key": "flux2_klein_4b",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "candidates_per_view": 2,
        "steps": 4,
        "guidance_scale": 1.0,
        "strength": 0.62,
        "seed": 20260610,
    },
    "reference_generation": {
        "default_model_key": "flux2_klein_4b",
        "provider": "external_imagegen",
        "dashscope_model": "wan2.6-t2i",
        "dashscope_image_edit_model": "wan2.6-image",
        "concept_size": "1472*1104",
        "module_reference_size": "1280*1280",
        "width": 1280,
        "height": 1280,
        "concurrent_workers": 2,
        "dashscope_retries": 4,
        "dashscope_retry_delay_seconds": 8,
        "max_review_attempts": 3,
    },
    "image2_executor": {
        "provider": "local_model",
        "model_key": "flux2_klein_4b",
        "reference_model_key": "flux2_klein_4b",
        "final_model_key": "flux2_klein_4b",
        "seed": 20260610,
        "concept_width": 1024,
        "concept_height": 1024,
        "module_width": 1024,
        "module_height": 1024,
        "final_width": 1024,
        "final_height": 1024,
        "timeout_seconds": 900,
        "command": "",
    },
    "models": {},
    "score": {
        "version": "structure_v2",
        "edge_weight": 0.55,
        "mask_weight": 0.25,
        "prompt_weight": 0.2,
        "copy_top_k": 1,
        "structure_v2": {
            "silhouette_weight": 0.3,
            "edge_chamfer_weight": 0.25,
            "added_part_weight": 0.2,
            "roughness_weight": 0.15,
            "background_weight": 0.1,
        },
    },
    "agent_llm": {
        "enabled": True,
        "provider": "openai-compatible",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.7-plus",
        "hf_model_id": "",
        "hf_endpoint": "",
        "local_model_dir": "",
        "container_model_dir": "",
        "service_script": "",
        "timeout_seconds": 90,
        "temperature": 0.2,
        "max_tokens": 1600,
        "api_key": "",
        "api_key_env": "DASHSCOPE_API_KEY",
        "no_proxy": False,
        "use_tool_schema": True,
        "fallback_to_rules": False,
    },
    "auto_agent": {
        "default_source_mode": "procedural",
        "text_to_3d_fallback": "procedural",
        "model_backend": "sample",
        "default_model_key": "flux2_klein_4b",
        "default_output_views": 3,
        "quality_presets": {
            "fast": {"render_resolution": 768, "ai_resolution": 768, "render_samples": 32},
            "balanced": {"render_resolution": 1024, "ai_resolution": 1024, "render_samples": 64},
            "high": {"render_resolution": 1536, "ai_resolution": 1536, "render_samples": 96},
        },
    },
    "agent": {
        "enabled": True,
        "default_model_key": "flux2_klein_4b",
        "max_generations": 10,
        "target_view": "view_locked",
        "expand_view_ids": ["view_locked", "view_left_30", "view_right_30"],
        "candidates_per_view": 3,
        "default_reference_channels": ["rgb", "edge", "depth", "normal", "mask", "skeleton"],
        "experimental_reference_channels": [],
        "roughness_weight": 0.25,
        "edge_weight": 0.35,
        "mask_weight": 0.25,
        "background_weight": 0.15,
        "pass_threshold": 0.62,
    },
}


def _parse_dotenv_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _load_dotenv_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _parse_dotenv_value(raw_value)


def _load_project_dotenv(config_path: Path | None = None) -> None:
    candidates: list[Path] = [Path.cwd() / ".env"]
    if config_path is not None:
        candidates.append(config_path.resolve().parent.parent / ".env")
    candidates.append(Path(__file__).resolve().parents[2] / ".env")
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _load_dotenv_file(resolved)


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    llm = merged.setdefault("agent_llm", {})
    string_overrides = {
        "H3D_AGENT_LLM_PROVIDER": "provider",
        "H3D_AGENT_LLM_BASE_URL": "base_url",
        "H3D_AGENT_LLM_MODEL": "model",
        "H3D_AGENT_LLM_API_KEY": "api_key",
        "H3D_AGENT_LLM_API_KEY_ENV": "api_key_env",
    }
    for env_name, key in string_overrides.items():
        value = os.environ.get(env_name)
        if value:
            llm[key] = value

    int_overrides = {
        "H3D_AGENT_LLM_MAX_TOKENS": "max_tokens",
    }
    for env_name, key in int_overrides.items():
        value = os.environ.get(env_name)
        if value:
            llm[key] = int(value)

    float_overrides = {
        "H3D_AGENT_LLM_TIMEOUT_SECONDS": "timeout_seconds",
        "H3D_AGENT_LLM_TEMPERATURE": "temperature",
    }
    for env_name, key in float_overrides.items():
        value = os.environ.get(env_name)
        if value:
            llm[key] = float(value)

    bool_overrides = {
        "H3D_AGENT_LLM_ENABLED": "enabled",
        "H3D_AGENT_LLM_NO_PROXY": "no_proxy",
        "H3D_AGENT_LLM_USE_TOOL_SCHEMA": "use_tool_schema",
        "H3D_AGENT_LLM_FALLBACK_TO_RULES": "fallback_to_rules",
    }
    for env_name, key in bool_overrides.items():
        value = os.environ.get(env_name)
        if value:
            llm[key] = _env_bool(value)
    return merged


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
        _load_project_dotenv(None)
        return _apply_env_overrides(DEFAULT_CONFIG)
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    _load_project_dotenv(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        user_config = json.load(fh)
    return _apply_env_overrides(deep_merge(DEFAULT_CONFIG, user_config))


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
