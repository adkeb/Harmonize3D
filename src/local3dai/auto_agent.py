from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageStat

from .agent import AgentRunOptions, run_agent_render, summarize_agent_report
from .camera import CameraState
from .config import load_config
from .manifest import read_manifest, write_manifest
from .modelgen import generate_3d_model
from .sample import create_sample_renders
from .workflow import WorkflowOptions, run_workflow


Progress = Callable[[str, str, float], None]
QWEN_AGENT_MODEL_ID = ""
QWEN_AGENT_SERVED_MODEL = "qwen3.7-plus"
QWEN_AGENT_HF_ENDPOINT = ""
QWEN_AGENT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_AGENT_API_KEY_ENV = "DASHSCOPE_API_KEY"

AUTO_STAGE_IDS = [
    "understand",
    "expand",
    "plan",
    "source",
    "mesh_check",
    "camera",
    "render",
    "agent",
    "score",
    "retry",
    "package",
    "complete",
]


@dataclass
class AutoRunOptions:
    request: str
    output_dir: Path
    config_path: Path = Path("configs/local.json")
    source_mode: str = "auto"
    model_path: Path | None = None
    reference_image: Path | None = None
    output_views: int = 3
    quality_mode: str = "balanced"
    geometry_mode: str = "strict"
    style_preset: str = "product"
    backend_model_key: str | None = None
    backend: str | None = None
    num_candidates_per_view: int = 3
    max_retries: int = 2
    seed: int = 20260610
    dry_run: bool = False
    use_llm: bool = True


def _direct_opener(no_proxy: bool) -> urllib.request.OpenerDirector:
    if no_proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _resolve_llm_api_key(llm_cfg: dict[str, Any]) -> tuple[str, str]:
    direct = str(llm_cfg.get("api_key", "") or "").strip()
    if direct and direct.upper() not in {"EMPTY", "NONE", "NULL"}:
        return direct, "api_key"
    env_name = str(llm_cfg.get("api_key_env", QWEN_AGENT_API_KEY_ENV) or "").strip()
    if env_name and os.environ.get(env_name):
        return str(os.environ[env_name]), env_name
    for fallback_env in ("H3D_AGENT_LLM_API_KEY", "DASHSCOPE_API_KEY"):
        if os.environ.get(fallback_env):
            return str(os.environ[fallback_env]), fallback_env
    if direct.upper() == "EMPTY":
        return "EMPTY", "api_key"
    return "", env_name


def _llm_headers(llm_cfg: dict[str, Any], *, accept_json: bool = True) -> dict[str, str]:
    api_key, _source = _resolve_llm_api_key(llm_cfg)
    headers = {"User-Agent": "Harmonize3D-AutoAgent/1.0"}
    if accept_json:
        headers["Accept"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _clamp_views(value: int) -> int:
    if value <= 1:
        return 1
    if value <= 3:
        return 3
    if value <= 5:
        return 5
    return 8


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return cleaned[:48] or "auto-task"


def _extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _call_qwen_planner(config: dict[str, Any], options: AutoRunOptions) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    llm_cfg = config.get("agent_llm", {})
    if not options.use_llm or not llm_cfg.get("enabled", True):
        return None, {"backend": "rules", "reason": "llm_disabled"}

    base_url = str(llm_cfg.get("base_url", QWEN_AGENT_BASE_URL)).rstrip("/")
    model = str(llm_cfg.get("model", QWEN_AGENT_SERVED_MODEL))
    timeout = float(llm_cfg.get("timeout_seconds", 3.0))
    api_key, api_key_source = _resolve_llm_api_key(llm_cfg)
    if not api_key and str(llm_cfg.get("api_key_env", "") or "").strip():
        reason = f"llm_api_key_missing: set {llm_cfg.get('api_key_env')}"
        if llm_cfg.get("fallback_to_rules", True):
            return None, {"backend": "rules", "reason": reason, "model": model, "base_url": base_url}
        raise RuntimeError(reason)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the Harmonize3D local 3D rendering planning agent. "
                    "Use the provided controlled tools when useful, but never invent files or bypass tool outputs. "
                    "Return strict JSON with keys auto_task, prompt_plan, camera_plan, tool_plan when responding in text. "
                    "The final AI render must only consume Blender render_manifest channels unless a user explicitly enables reference-image style mode."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "request": options.request,
                        "source_mode": options.source_mode,
                        "output_views": options.output_views,
                        "quality_mode": options.quality_mode,
                        "geometry_mode": options.geometry_mode,
                        "style_preset": options.style_preset,
                        "available_tools": LOCAL_TOOL_SPECS,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": float(llm_cfg.get("temperature", 0.2)),
        "max_tokens": int(llm_cfg.get("max_tokens", 1600)),
    }
    if llm_cfg.get("use_tool_schema", True):
        payload["tools"] = _openai_tool_specs()
        payload["tool_choice"] = "auto"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = _llm_headers(llm_cfg, accept_json=False)
    headers.update({"Content-Type": "application/json"})
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers=headers,
        method="POST",
    )
    opener = _direct_opener(bool(llm_cfg.get("no_proxy", True)))
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        if llm_cfg.get("fallback_to_rules", True):
            return None, {"backend": "rules", "reason": f"llm_unavailable: {exc}", "model": model, "base_url": base_url}
        raise RuntimeError(f"Qwen planner is unavailable at {base_url}: {exc}") from exc

    parsed = json.loads(raw)
    message = parsed.get("choices", [{}])[0].get("message", {})
    content = message.get("content") or ""
    tool_plan = _tool_plan_from_openai_message(message)
    plan = _extract_json(content)
    if plan is None and tool_plan:
        plan = {"tool_plan": tool_plan}
    if plan is None:
        if llm_cfg.get("fallback_to_rules", True):
            return None, {"backend": "rules", "reason": "llm_returned_non_json", "model": model, "base_url": base_url}
        raise RuntimeError("Qwen planner returned non-JSON content.")
    if tool_plan:
        plan.setdefault("tool_plan", tool_plan)
    return plan, {
        "backend": "qwen_openai_compatible",
        "model": model,
        "base_url": base_url,
        "api_key_source": api_key_source if api_key and api_key != "EMPTY" else "",
        "tool_call_count": len(tool_plan),
    }


def qwen_runtime_status(config: dict[str, Any], *, check_hf_mirror: bool = False, timeout: float | None = None) -> dict[str, Any]:
    llm_cfg = config.get("agent_llm", {})
    provider = str(llm_cfg.get("provider", "openai-compatible"))
    base_url = str(llm_cfg.get("base_url", QWEN_AGENT_BASE_URL)).rstrip("/")
    model = str(llm_cfg.get("model", QWEN_AGENT_SERVED_MODEL))
    hf_model_id = str(llm_cfg.get("hf_model_id", QWEN_AGENT_MODEL_ID))
    hf_endpoint = str(llm_cfg.get("hf_endpoint", QWEN_AGENT_HF_ENDPOINT)).rstrip("/")
    local_model_dir_raw = str(llm_cfg.get("local_model_dir", "") or "")
    local_model_dir = Path(local_model_dir_raw) if local_model_dir_raw else None
    container_model_dir = str(llm_cfg.get("container_model_dir", "") or "")
    service_script = str(llm_cfg.get("service_script", "") or "")
    no_proxy = bool(llm_cfg.get("no_proxy", True))
    probe_timeout = float(timeout if timeout is not None else min(float(llm_cfg.get("timeout_seconds", 3.0)), 2.0))
    api_key, api_key_source = _resolve_llm_api_key(llm_cfg)
    opener = _direct_opener(no_proxy)
    service: dict[str, Any] = {
        "base_url": base_url,
        "provider": provider,
        "reachable": False,
        "served": False,
        "models": [],
        "api_key_configured": bool(api_key and api_key != "EMPTY"),
        "api_key_source": api_key_source if api_key and api_key != "EMPTY" else "",
    }
    if not api_key and str(llm_cfg.get("api_key_env", "") or "").strip():
        service["error"] = f"missing API key env {llm_cfg.get('api_key_env')}"
    else:
        try:
            request = urllib.request.Request(
                f"{base_url}/models",
                headers=_llm_headers(llm_cfg),
                method="GET",
            )
            with opener.open(request, timeout=probe_timeout) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            models = [str(item.get("id", "")) for item in data.get("data", []) if isinstance(item, dict)]
            service.update(
                {
                    "reachable": True,
                    "served": model in models or any(hf_model_id and hf_model_id in item for item in models) or not models,
                    "models": models[:32],
                }
            )
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            service["error"] = str(exc)

    mirror: dict[str, Any] = {
        "checked": bool(check_hf_mirror),
        "endpoint": hf_endpoint,
        "model_id": hf_model_id,
        "reachable": None,
    }
    if check_hf_mirror and hf_endpoint and hf_model_id:
        try:
            request = urllib.request.Request(
                f"{hf_endpoint}/api/models/{hf_model_id}",
                headers={"User-Agent": "Harmonize3D-AutoAgent/1.0", "Accept": "application/json"},
                method="HEAD",
            )
            with opener.open(request, timeout=probe_timeout) as response:
                mirror.update({"reachable": 200 <= int(response.status) < 400, "status_code": int(response.status)})
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            mirror.update({"reachable": False, "error": str(exc)})
    elif check_hf_mirror:
        mirror.update({"reachable": False, "error": "hf_endpoint or hf_model_id is not configured"})

    local_model = {
        "path": str(local_model_dir) if local_model_dir else "",
        "container_path": container_model_dir,
        "exists": bool(local_model_dir and local_model_dir.exists()),
        "has_config": bool(local_model_dir and (local_model_dir / "config.json").exists()),
        "has_tokenizer": bool(
            local_model_dir
            and ((local_model_dir / "tokenizer_config.json").exists() or (local_model_dir / "tokenizer.json").exists())
        ),
        "service_script": service_script,
    }
    if local_model_dir and local_model_dir.exists():
        try:
            local_model["size_bytes"] = sum(path.stat().st_size for path in local_model_dir.rglob("*") if path.is_file())
        except OSError:
            local_model["size_bytes"] = 0

    return {
        "type": "qwen_runtime_status",
        "provider": provider,
        "model": model,
        "hf_model_id": hf_model_id,
        "hf_endpoint": hf_endpoint,
        "local_model": local_model,
        "no_proxy": no_proxy,
        "service": service,
        "hf_mirror": mirror,
        "ready": bool(service.get("reachable") and service.get("served")),
    }


def mesh_sanity_report(model_path: str | Path, *, source_mode: str, expected_views: int) -> dict[str, Any]:
    path = Path(model_path)
    suffix = path.suffix.lower()
    exists = path.exists() and path.is_file()
    supported = suffix in {".obj", ".glb", ".gltf", ".fbx", ".ply", ".stl"}
    size_bytes = path.stat().st_size if exists else 0
    checks = {
        "exists": exists,
        "supported_extension": supported,
        "nonempty_file": size_bytes > 0,
        "expected_views_valid": expected_views in {1, 3, 5, 8},
    }
    failures = [key for key, passed in checks.items() if not passed]
    status = "pass" if not failures else "failed"
    return {
        "type": "mesh_sanity",
        "status": status,
        "model_path": str(path),
        "source_mode": source_mode,
        "suffix": suffix,
        "size_bytes": size_bytes,
        "checks": checks,
        "failure_reasons": failures,
    }


def _detect_object_type(text: str) -> str:
    lowered = text.lower()
    if any(token in text for token in ("车", "跑车", "汽车", "电动")) or any(token in lowered for token in ("car", "vehicle", "hypercar", "sports car")):
        return "car"
    if any(token in text for token in ("建筑", "房子", "展馆")) or "architecture" in lowered:
        return "architecture"
    if any(token in text for token in ("角色", "人物", "人像")) or "character" in lowered:
        return "character"
    if any(token in text for token in ("椅", "桌", "沙发", "家具")) or "furniture" in lowered:
        return "furniture"
    if any(token in text for token in ("产品", "商品")) or "product" in lowered:
        return "product"
    return "unknown"


def _material_phrase(request: str, object_type: str) -> str:
    lowered = request.lower()
    if "white" in lowered or "白" in request:
        if object_type == "car":
            return "pearl white automotive paint, smoke-gray glass, satin black tires, dark metallic wheels"
        return "smooth pearl white material, subtle satin highlights"
    if "red" in lowered or "红" in request:
        return "deep red glossy finish, clean reflective highlights"
    if "black" in lowered or "黑" in request:
        return "glossy black ceramic finish, controlled studio reflections"
    return "premium clean material, smooth surface finish, controlled studio reflections"


def _scene_phrase(style_preset: str) -> str:
    if style_preset == "cinematic":
        return "controlled cinematic studio lighting, clean dark-to-neutral background"
    if style_preset == "ecommerce":
        return "neutral ecommerce studio background, even softbox lighting, clean floor"
    if style_preset == "concept":
        return "concept design studio, neutral background, crisp presentation lighting"
    return "neutral gray product photography studio, softbox lighting, clean floor"


def _source_mode(options: AutoRunOptions, config: dict[str, Any]) -> str:
    requested = options.source_mode.strip().lower().replace("-", "_")
    if requested in {"model_path", "existing_model"} or options.model_path:
        return "existing_model"
    if requested in {"image_3d", "image_to_3d", "hunyuan_reference"} or options.reference_image:
        return "image_to_3d"
    if requested in {"procedural", "sample"}:
        return "procedural"
    if requested in {"text_3d", "text_to_3d", "prompt_3d"}:
        return "text_to_3d"
    return str(config.get("auto_agent", {}).get("default_source_mode", "procedural"))


def build_rule_plan(config: dict[str, Any], options: AutoRunOptions) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    object_type = _detect_object_type(options.request)
    source_mode = _source_mode(options, config)
    output_views = _clamp_views(int(options.output_views or config.get("auto_agent", {}).get("default_output_views", 3)))
    task_id = f"auto-{time.strftime('%Y%m%d-%H%M%S')}-{_slug(options.request)}"
    material = _material_phrase(options.request, object_type)
    scene = _scene_phrase(options.style_preset)
    geometry_phrase = (
        "Preserve the exact silhouette, proportions, holes, wheel placement, visible edges, and all mesh-derived geometry."
        if options.geometry_mode == "strict"
        else "Preserve the source mesh silhouette and major geometry while improving material quality."
    )
    base_prompt = options.request.strip()
    render_prompt = (
        f"Render the same 3D mesh as a factory-new high-end CGI product image. {geometry_phrase} "
        f"Use {material}. Place it in a {scene}. Keep the design unbranded, pristine, smooth, and production-ready."
    )
    negative_constraints = [
        "no text",
        "no logos",
        "no changed silhouette",
        "no added structural parts",
        "no noisy background",
    ]
    auto_task = {
        "task_id": task_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "user_request": options.request,
        "expanded_request": render_prompt,
        "source_mode": source_mode,
        "object_type": object_type,
        "style_preset": options.style_preset,
        "quality_mode": options.quality_mode,
        "geometry_mode": options.geometry_mode,
        "output_views": output_views,
        "num_candidates_per_view": max(1, int(options.num_candidates_per_view)),
        "max_retries": max(0, int(options.max_retries)),
    }
    prompt_plan = {
        "base_prompt": base_prompt,
        "render_prompt": render_prompt,
        "negative_prompt": "geometry drift, changed silhouette, extra parts, missing parts, text, watermark, logo, noisy texture",
        "prompt_variants": [
            {
                "name": "smooth_product",
                "prompt": f"{render_prompt} Use broad clean highlights and smooth premium material finish.",
                "intended_effect": "clean product render with high geometry preservation",
            },
            {
                "name": "strict_geometry",
                "prompt": f"{render_prompt} Treat the Blender white render as the exact geometry authority.",
                "intended_effect": "reduce silhouette drift and added parts",
            },
        ],
        "forbidden_changes": negative_constraints,
    }
    camera_plan = plan_cameras(auto_task)
    return auto_task, prompt_plan, camera_plan


def _merge_plan(rule_value: dict[str, Any], llm_value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(llm_value, dict):
        return rule_value
    merged = dict(rule_value)
    for key, value in llm_value.items():
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged


def plan_cameras(auto_task: dict[str, Any]) -> dict[str, Any]:
    object_type = str(auto_task.get("object_type", "unknown"))
    style = str(auto_task.get("style_preset", "product"))
    output_views = _clamp_views(int(auto_task.get("output_views", 3)))
    if object_type in {"car", "product", "furniture"}:
        hero_azimuth, elevation, ortho_scale = 330.0, 22.0, 2.55
    elif object_type == "architecture":
        hero_azimuth, elevation, ortho_scale = 315.0, 24.0, 3.0
    elif object_type == "character":
        hero_azimuth, elevation, ortho_scale = 330.0, 10.0, 2.35
    else:
        hero_azimuth, elevation, ortho_scale = 330.0, 20.0, 2.7
    if style == "cinematic":
        elevation += 3.0
    base = CameraState(
        azimuth_deg=hero_azimuth,
        elevation_deg=elevation,
        distance_scale=1.0,
        ortho_scale=ortho_scale,
        target=(0.0, 0.0, 0.05),
        viewport_aspect=1.0,
        coordinate_space="blender_z_up",
    )
    views = [
        {
            "view_id": "view_locked",
            "role": "hero product render",
            "azimuth_deg": hero_azimuth,
            "elevation_deg": elevation,
            "distance_scale": 1.0,
            "ortho_scale": ortho_scale,
            "camera_type": "orthographic",
        }
    ]
    if output_views >= 3:
        views.extend(
            [
                {
                    "view_id": "view_left_30",
                    "role": "left consistency view",
                    "azimuth_deg": (hero_azimuth - 30.0) % 360.0,
                    "elevation_deg": elevation,
                    "distance_scale": 1.0,
                    "ortho_scale": ortho_scale,
                    "camera_type": "orthographic",
                },
                {
                    "view_id": "view_right_30",
                    "role": "right consistency view",
                    "azimuth_deg": (hero_azimuth + 30.0) % 360.0,
                    "elevation_deg": elevation,
                    "distance_scale": 1.0,
                    "ortho_scale": ortho_scale,
                    "camera_type": "orthographic",
                },
            ]
        )
    if output_views >= 5:
        views.extend(
            [
                {"view_id": "view_front", "role": "front verification view", "azimuth_deg": 0.0, "elevation_deg": elevation, "camera_type": "orthographic"},
                {"view_id": "view_side", "role": "side verification view", "azimuth_deg": 90.0, "elevation_deg": elevation, "camera_type": "orthographic"},
            ]
        )
    if output_views >= 8:
        views.extend(
            [
                {"view_id": "view_rear", "role": "rear verification view", "azimuth_deg": 180.0, "elevation_deg": elevation, "camera_type": "orthographic"},
                {"view_id": "view_top_oblique", "role": "top oblique verification view", "azimuth_deg": 45.0, "elevation_deg": 45.0, "camera_type": "orthographic"},
                {"view_id": "view_detail", "role": "detail material view", "azimuth_deg": hero_azimuth, "elevation_deg": elevation + 6.0, "camera_type": "orthographic"},
            ]
        )
    return {
        "camera_state": base.to_dict(),
        "views": views[:output_views],
        "composition": {
            "subject_frame_ratio": [0.65, 0.85],
            "default_camera_type": "orthographic",
            "notes": "Blender fixed-camera renderer currently emits view_locked, view_left_30, and view_right_30 from the hero camera state.",
        },
    }


def retry_policy_decisions(scores: dict[str, Any]) -> list[dict[str, str]]:
    decisions: list[dict[str, str]] = []
    if float(scores.get("silhouette_iou", 1.0)) < 0.75:
        decisions.append({"reason": "silhouette_iou below threshold", "action": "use strict_geometry prompt and reduce generation freedom"})
    if float(scores.get("added_part_penalty", 0.0)) > 0.15:
        decisions.append({"reason": "added_part_penalty too high", "action": "reinforce unbranded minimal design and no added structural parts"})
    if float(scores.get("roughness", 1.0)) < 0.55:
        decisions.append({"reason": "roughness below threshold", "action": "switch to smooth_product prompt and avoid depth/normal references"})
    if float(scores.get("background_cleanliness", 1.0)) < 0.85:
        decisions.append({"reason": "background_cleanliness below threshold", "action": "use neutral gray studio background preset"})
    return decisions


LOCAL_TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "qwen_planner", "purpose": "use the configured Qwen OpenAI-compatible service as the Auto Agent brain and supervisor"},
    {"name": "requirement_expander", "purpose": "parse and expand one natural-language request into auto_task and prompt_plan"},
    {"name": "auto_camera_planner", "purpose": "produce deterministic Blender-compatible camera plans"},
    {"name": "generate_or_load_3d", "purpose": "generate, load, or select a 3D mesh source"},
    {"name": "mesh_quality_check", "purpose": "verify the selected mesh exists, has a supported format, and is safe to render"},
    {"name": "render_white_channels", "purpose": "render Blender rgb/edge/mask/depth/normal channels from the mesh"},
    {"name": "ai_candidate_search", "purpose": "generate per-view AI candidates only from render_manifest channels"},
    {"name": "structure_scoring", "purpose": "score candidates against mesh-derived render channels"},
    {"name": "execute_workflow", "purpose": "run the controlled Harmonize3D workflow adapter with selected tools and manifests"},
    {"name": "visual_judgement", "purpose": "judge final images using image statistics, structure scores, and multiview scores"},
    {"name": "package_outputs", "purpose": "write stable reports, logs, contact sheets, and final artifacts"},
]


def _openai_tool_specs() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for spec in LOCAL_TOOL_SPECS:
        name = str(spec["name"])
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(spec.get("purpose", "")),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "Why this tool should run now."},
                            "inputs": {"type": "object", "description": "Tool-specific input summary."},
                        },
                        "required": ["reason"],
                        "additionalProperties": True,
                    },
                },
            }
        )
    return tools


def _tool_plan_from_openai_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        raw_args = function.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except (TypeError, ValueError, json.JSONDecodeError):
            args = {"raw_arguments": raw_args}
        plan.append({"tool": name, "arguments": args, "id": call.get("id", "")})
    return plan


class AutoToolExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, name: str, args: dict[str, Any], fn: Callable[[], Any]) -> Any:
        started = time.time()
        call: dict[str, Any] = {
            "tool": name,
            "args": _summarize_tool_args(args),
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self.calls.append(call)
        try:
            result = fn()
            call["status"] = "complete"
            call["elapsed_seconds"] = round(time.time() - started, 3)
            call["result"] = _summarize_tool_result(result)
            return result
        except Exception as exc:
            call["status"] = "failed"
            call["elapsed_seconds"] = round(time.time() - started, 3)
            call["error"] = str(exc)
            raise


def _summarize_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, Path):
            summary[key] = str(value)
        elif isinstance(value, str):
            summary[key] = value if len(value) <= 240 else value[:237] + "..."
        elif isinstance(value, (int, float, bool)) or value is None:
            summary[key] = value
        elif isinstance(value, (list, tuple)):
            summary[key] = list(value[:12]) if len(value) <= 12 else list(value[:12]) + ["..."]
        elif isinstance(value, dict):
            summary[key] = {str(k): _summarize_tool_args({"v": v})["v"] for k, v in list(value.items())[:16]}
        else:
            summary[key] = type(value).__name__
    return summary


def _summarize_tool_result(result: Any) -> Any:
    if isinstance(result, Path):
        return str(result)
    if isinstance(result, dict):
        keys = [
            "backend",
            "reason",
            "model",
            "base_url",
            "hf_model_id",
            "hf_endpoint",
            "no_proxy",
            "tool_call_count",
            "auto_task",
            "prompt_plan",
            "camera_plan",
            "tool_plan",
            "checks",
            "failure_reasons",
            "mesh_sanity",
            "mesh_metadata",
            "status",
            "task_id",
            "workdir",
            "model_path",
            "render_manifest",
            "agent_report",
            "visual_judgement",
            "tool_calls",
            "final_image",
            "comparison_image",
            "contact_sheet",
        ]
        return {key: result[key] for key in keys if key in result}
    if isinstance(result, (list, tuple)):
        return [_summarize_tool_result(item) for item in result[:8]]
    if isinstance(result, (str, int, float, bool)) or result is None:
        return result
    return type(result).__name__


def _image_judgement(path: str | Path) -> dict[str, Any]:
    image_path = Path(path)
    if not image_path.exists() or not image_path.is_file():
        return {"exists": False, "nonblank": False, "path": str(image_path)}
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        extrema = rgb.getextrema()
    channel_ranges = [hi - lo for lo, hi in extrema]
    mean = [round(float(value), 3) for value in stat.mean]
    stddev = [round(float(value), 3) for value in stat.stddev]
    return {
        "exists": True,
        "path": str(image_path),
        "size": list(rgb.size),
        "mean_rgb": mean,
        "stddev_rgb": stddev,
        "nonblank": max(channel_ranges) > 8 and max(stat.stddev) > 2,
        "dynamic_range": max(channel_ranges),
    }


def visual_judgement_report(
    *,
    final_image: str,
    comparison_image: str,
    contact_sheet: str,
    agent_summary: dict[str, Any],
) -> dict[str, Any]:
    final_stats = _image_judgement(final_image)
    comparison_stats = _image_judgement(comparison_image)
    contact_stats = _image_judgement(contact_sheet) if contact_sheet else {"exists": False, "nonblank": False, "path": ""}
    structure_scores = dict(agent_summary.get("structure_scores", {}))
    multiview_scores = dict(agent_summary.get("multiview_scores", {}))
    view_totals = [float(score.get("total", 0.0)) for score in structure_scores.values() if isinstance(score, dict)]
    min_structure = min(view_totals) if view_totals else float(agent_summary.get("selected_trial", {}).get("scores", {}).get("total", 0.0))
    multiview_total = float(multiview_scores.get("total", 1.0 if len(view_totals) <= 1 else 0.0))
    failures: list[str] = []
    if not final_stats.get("nonblank"):
        failures.append("final_image_blank_or_missing")
    if comparison_image and not comparison_stats.get("nonblank"):
        failures.append("comparison_image_blank_or_missing")
    if contact_sheet and not contact_stats.get("nonblank"):
        failures.append("contact_sheet_blank_or_missing")
    if min_structure < 0.45:
        failures.append("structure_score_too_low")
    if multiview_total < 0.45:
        failures.append("multiview_consistency_too_low")
    checks = {
        "final_image_nonblank": bool(final_stats.get("nonblank")),
        "comparison_image_nonblank": bool(comparison_stats.get("nonblank")) if comparison_image else True,
        "contact_sheet_nonblank": bool(contact_stats.get("nonblank")) if contact_sheet else True,
        "structure_score_gate": min_structure >= 0.45,
        "multiview_consistency_gate": multiview_total >= 0.45,
        "structure_review_gate": min_structure >= 0.62,
        "multiview_review_gate": multiview_total >= 0.55,
    }
    status = "pass"
    if failures:
        status = "failed" if any(item.endswith("blank_or_missing") for item in failures) else "needs_review"
    elif min_structure < 0.62 or multiview_total < 0.55:
        status = "needs_review"
    return {
        "type": "visual_judgement",
        "backend": "cv_structure_v2",
        "has_visual_judgement": True,
        "status": status,
        "checks": checks,
        "human_review_recommended": status != "pass",
        "final_image": final_stats,
        "comparison_image": comparison_stats,
        "contact_sheet": contact_stats,
        "structure_min_total": round(min_structure, 6),
        "multiview_total": round(multiview_total, 6),
        "failure_reasons": failures,
        "notes": [
            "Visual judgement is computed from final pixels plus mesh-derived structure and multiview scores.",
            "It is independent of the text planner and can run in mock mode.",
        ],
    }


def _workflow_source_mode(auto_task: dict[str, Any], options: AutoRunOptions, config: dict[str, Any]) -> tuple[str, str]:
    source_mode = str(auto_task.get("source_mode", "procedural"))
    if source_mode == "existing_model":
        return "model_path", source_mode
    if source_mode == "image_to_3d":
        return "hunyuan_reference", source_mode
    if source_mode == "text_to_3d":
        fallback = str(config.get("auto_agent", {}).get("text_to_3d_fallback", "procedural"))
        return fallback, source_mode
    return "procedural", source_mode


def _quality_resolution(config: dict[str, Any], quality_mode: str, *, dry_run: bool) -> tuple[int, int, int]:
    if dry_run:
        return 128, 128, 128
    presets = config.get("auto_agent", {}).get("quality_presets", {})
    preset = presets.get(quality_mode, presets.get("balanced", {}))
    render_resolution = int(preset.get("render_resolution", config.get("render", {}).get("resolution", 1024)))
    ai_resolution = int(preset.get("ai_resolution", config.get("ai", {}).get("width", 1024)))
    samples = int(preset.get("render_samples", config.get("render", {}).get("samples", 64)))
    return render_resolution, ai_resolution, samples


def _write_log(path: Path, stage: str, message: str, progress: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": time.strftime("%H:%M:%S"), "stage": stage, "message": message, "progress": progress}, ensure_ascii=False) + "\n")


def _file_url(path: str | Path) -> str:
    return f"/api/file?path={str(path)}"


def _copy_if_exists(source: str | Path | None, target: Path) -> str:
    if not source:
        return ""
    source_path = Path(source)
    if not source_path.exists() or not source_path.is_file():
        return str(source_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target)
    return str(target)


def _package_auto_outputs(
    *,
    workdir: Path,
    auto_task: dict[str, Any],
    prompt_plan: dict[str, Any],
    camera_plan: dict[str, Any],
    workflow_summary: dict[str, Any],
    planning_info: dict[str, Any],
    log_path: Path,
    started: float,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    final_dir = workdir / "final"
    reports_dir = workdir / "reports"
    final_image = _copy_if_exists(workflow_summary.get("final_image"), final_dir / "final.png")
    comparison_image = _copy_if_exists(workflow_summary.get("comparison_image"), final_dir / "white_vs_final.png")
    contact_sheet = _copy_if_exists(
        workflow_summary.get("multiview_contact_sheet") or workflow_summary.get("three_view_contact"),
        final_dir / "contact_sheet.png",
    )
    agent_summary = workflow_summary.get("agent_summary") or workflow_summary.get("summary") or {}
    final_view_images: dict[str, str] = {}
    for view_id, source in dict(agent_summary.get("final_view_images", {})).items():
        final_view_images[view_id] = _copy_if_exists(source, final_dir / f"final_{view_id}.png")
    agent_report = _copy_if_exists(workflow_summary.get("agent_report") or agent_summary.get("agent_report"), reports_dir / "agent_report.json")
    scores = {
        "structure": agent_summary.get("structure_scores", {}),
        "multiview": agent_summary.get("multiview_scores", {}),
        "decisions": agent_summary.get("retry_decisions", []),
    }
    scores_path = write_manifest(reports_dir / "scores.json", scores)
    multiview_path = write_manifest(reports_dir / "multiview_score.json", agent_summary.get("multiview_scores", {}))
    visual_judgement = visual_judgement_report(
        final_image=final_image,
        comparison_image=comparison_image,
        contact_sheet=contact_sheet,
        agent_summary=agent_summary,
    )
    visual_path = write_manifest(reports_dir / "visual_judgement.json", visual_judgement)
    tool_calls_path = write_manifest(reports_dir / "tool_calls.json", {"tools": LOCAL_TOOL_SPECS, "calls": tool_calls or []})
    status = str(workflow_summary.get("status", "needs_review"))
    mesh_sanity_status = "unknown"
    if workflow_summary.get("mesh_sanity"):
        try:
            mesh_sanity_status = str(read_manifest(workflow_summary["mesh_sanity"]).get("status", "unknown"))
        except (OSError, ValueError, json.JSONDecodeError):
            mesh_sanity_status = "failed"
    if mesh_sanity_status == "failed":
        status = "failed"
    if visual_judgement["status"] == "failed":
        status = "failed"
    elif visual_judgement["status"] == "needs_review" and status == "complete":
        status = "needs_review"
    summary = {
        "type": "auto_agent_summary",
        "status": status,
        "task_id": auto_task["task_id"],
        "workdir": str(workdir),
        "request": auto_task.get("user_request", ""),
        "planning": planning_info,
        "source_mode_requested": auto_task.get("source_mode", ""),
        "source_mode_resolved": workflow_summary.get("source_mode", ""),
        "model_path": workflow_summary.get("model_path", ""),
        "mesh_metadata": workflow_summary.get("mesh_metadata", ""),
        "mesh_sanity": workflow_summary.get("mesh_sanity", ""),
        "render_manifest": workflow_summary.get("render_manifest", ""),
        "auto_task": str(workdir / "auto_task.json"),
        "prompt_plan": str(workdir / "prompt_plan.json"),
        "camera_plan": str(workdir / "camera_plan.json"),
        "scores": str(scores_path),
        "multiview_score": str(multiview_path),
        "run_log": str(log_path),
        "tool_calls": str(tool_calls_path),
        "visual_judgement": str(visual_path),
        "agent_report": agent_report,
        "final_image": final_image,
        "comparison_image": comparison_image,
        "contact_sheet": contact_sheet,
        "final_view_images": final_view_images,
        "elapsed_seconds": round(time.time() - started, 3),
        "workflow_summary": workflow_summary,
        "capabilities": {
            "visual_judgement": {
                "enabled": True,
                "backend": visual_judgement["backend"],
                "status": visual_judgement["status"],
                "checked_artifacts": ["final_image", "comparison_image", "contact_sheet"],
            },
            "mesh_sanity": {"enabled": True, "status": mesh_sanity_status},
            "tool_execution": {
                "enabled": True,
                "execution_mode": "controlled_local_tools",
                "tool_count": len(LOCAL_TOOL_SPECS),
                "tool_call_count": len(tool_calls or []),
                "allowed_tool_names": [tool["name"] for tool in LOCAL_TOOL_SPECS],
                "executed_tool_names": [call["tool"] for call in tool_calls or []],
            },
        },
    }
    summary["artifacts"] = {
        key: value
        for key, value in summary.items()
        if key
        in {
            "auto_task",
            "prompt_plan",
            "camera_plan",
            "scores",
            "multiview_score",
            "run_log",
            "agent_report",
            "visual_judgement",
            "tool_calls",
            "final_image",
            "comparison_image",
            "contact_sheet",
            "render_manifest",
            "model_path",
            "mesh_metadata",
            "mesh_sanity",
        }
        and value
    }
    summary["artifact_urls"] = {key: _file_url(value) for key, value in summary["artifacts"].items() if Path(str(value)).suffix}
    write_manifest(workdir / "auto_summary.json", summary)
    return summary


def _run_auto_dry_workflow(
    *,
    config: dict[str, Any],
    options: AutoRunOptions,
    workdir: Path,
    auto_task: dict[str, Any],
    prompt_plan: dict[str, Any],
    progress: Progress | None,
    tool_executor: AutoToolExecutor | None = None,
) -> dict[str, Any]:
    def emit(stage: str, message: str, fraction: float) -> None:
        if progress:
            progress(stage, message, fraction)

    emit("source", "Creating deterministic sample 3D source for auto dry-run", 0.3)
    generate_args = {"prompt": prompt_plan["base_prompt"], "output": workdir / "mesh" / "white_mesh.obj", "backend": "sample"}
    if tool_executor:
        model_path = tool_executor.run("generate_or_load_3d", generate_args, lambda: generate_3d_model(**generate_args))
    else:
        model_path = generate_3d_model(**generate_args)
    metadata_path = write_manifest(
        workdir / "mesh" / "metadata.json",
        {
            "type": "mesh_metadata",
            "source_mode": "dry_run_sample",
            "requested_source_mode": auto_task.get("source_mode", ""),
            "model_path": str(model_path),
            "prompt": prompt_plan["base_prompt"],
            "created_by": "generate_or_load_3d",
        },
    )
    emit("mesh_check", "Checking generated mesh source", 0.36)
    sanity_args = {"model_path": model_path, "source_mode": "dry_run_sample", "expected_views": int(auto_task.get("output_views", 3))}
    if tool_executor:
        sanity = tool_executor.run("mesh_quality_check", sanity_args, lambda: mesh_sanity_report(**sanity_args))
    else:
        sanity = mesh_sanity_report(**sanity_args)
    sanity_path = write_manifest(workdir / "mesh" / "sanity.json", sanity)
    emit("camera", "Using auto camera plan for dry-run sample views", 0.4)
    emit("render", "Creating synthetic Blender channel manifest for auto dry-run", 0.45)
    render_args = {"output_dir": workdir / "renders", "views": max(3, int(auto_task.get("output_views", 3))), "resolution": 128}
    if tool_executor:
        render_manifest = tool_executor.run("render_white_channels", render_args, lambda: create_sample_renders(**render_args))
    else:
        render_manifest = create_sample_renders(**render_args)
    emit("agent", "Running MeshLock Agent with mock backend", 0.65)
    max_generations = max(1, int(auto_task["num_candidates_per_view"]) * min(3, int(auto_task["output_views"])) + int(auto_task.get("max_retries", 0)))
    agent_options = AgentRunOptions(
            input_renders=render_manifest,
            output_dir=workdir / "agent",
            prompt=prompt_plan["render_prompt"],
            config_path=options.config_path,
            model_key=options.backend_model_key or config.get("auto_agent", {}).get("default_model_key") or config.get("agent", {}).get("default_model_key", "flux2_klein_4b"),
            backend=options.backend or "mock",
            target_view="view_locked",
            max_generations=max_generations,
            seed=options.seed,
            expand_views=int(auto_task.get("output_views", 3)) > 1,
            expand_view_ids=("view_locked", "view_left_30", "view_right_30"),
            default_reference_channels=("rgb", "edge"),
            negative_prompt=prompt_plan.get("negative_prompt", ""),
            steps=1,
            width=128,
            height=128,
        )
    agent_args = {"input_renders": render_manifest, "output_dir": workdir / "agent", "backend": options.backend or "mock", "max_generations": max_generations}
    if tool_executor:
        agent_summary = tool_executor.run(
            "ai_candidate_search",
            agent_args,
            lambda: run_agent_render(agent_options, progress=lambda stage, message, fraction: emit(stage, message, 0.65 + fraction * 0.3)),
        )
        tool_executor.run(
            "structure_scoring",
            {"agent_report": agent_summary.get("agent_report", "")},
            lambda: {
                "structure_scores": agent_summary.get("structure_scores", {}),
                "multiview_scores": agent_summary.get("multiview_scores", {}),
                "retry_decisions": agent_summary.get("retry_decisions", []),
            },
        )
    else:
        agent_summary = run_agent_render(agent_options, progress=lambda stage, message, fraction: emit(stage, message, 0.65 + fraction * 0.3))
    with Path(render_manifest).open("r", encoding="utf-8") as fh:
        render_data = json.load(fh)
    white_image = render_data["views"][0]["files"]["rgb"]
    return {
        "type": "workflow_summary",
        "status": agent_summary["status"],
        "workdir": str(workdir),
        "prompt": prompt_plan["render_prompt"],
        "source_mode": "dry_run_sample",
        "model_path": str(model_path),
        "mesh_metadata": str(metadata_path),
        "mesh_sanity": str(sanity_path),
        "render_manifest": str(render_manifest),
        "agent_report": agent_summary["agent_report"],
        "score_summary": summarize_agent_report(agent_summary["agent_report"]),
        "white_image": white_image,
        "final_image": agent_summary["final_image"],
        "comparison_image": agent_summary["comparison_image"],
        "three_view_contact": agent_summary.get("three_view_contact", ""),
        "multiview_contact_sheet": agent_summary.get("multiview_contact_sheet", ""),
        "agent_summary": agent_summary,
    }


def run_auto_agent(options: AutoRunOptions, progress: Progress | None = None) -> dict[str, Any]:
    started = time.time()
    config = load_config(options.config_path)
    workdir = Path(options.output_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "reports" / "run.log"

    def emit(stage: str, message: str, fraction: float) -> None:
        _write_log(log_path, stage, message, fraction)
        if progress:
            progress(stage, message, fraction)

    tool_executor = AutoToolExecutor()
    emit("understand", "Understanding natural language request", 0.04)
    llm_plan, planning_info = tool_executor.run(
        "qwen_planner",
        {"request": options.request, "model": config.get("agent_llm", {}).get("model", QWEN_AGENT_SERVED_MODEL)},
        lambda: _call_qwen_planner(config, options),
    )
    rule_task, rule_prompt, rule_camera = tool_executor.run(
        "requirement_expander",
        {"request": options.request, "style_preset": options.style_preset, "geometry_mode": options.geometry_mode},
        lambda: build_rule_plan(config, options),
    )
    auto_task = _merge_plan(rule_task, llm_plan.get("auto_task") if llm_plan else None)
    prompt_plan = _merge_plan(rule_prompt, llm_plan.get("prompt_plan") if llm_plan else None)
    camera_plan = tool_executor.run(
        "auto_camera_planner",
        {"object_type": auto_task.get("object_type"), "output_views": auto_task.get("output_views", options.output_views)},
        lambda: _merge_plan(rule_camera, llm_plan.get("camera_plan") if llm_plan else None),
    )
    auto_task["task_id"] = str(auto_task.get("task_id") or rule_task["task_id"])
    auto_task["user_request"] = options.request
    auto_task["output_views"] = _clamp_views(int(auto_task.get("output_views", options.output_views)))
    auto_task["num_candidates_per_view"] = max(1, int(auto_task.get("num_candidates_per_view", options.num_candidates_per_view)))
    auto_task["max_retries"] = max(0, int(auto_task.get("max_retries", options.max_retries)))
    planning_info.setdefault("hf_model_id", config.get("agent_llm", {}).get("hf_model_id", QWEN_AGENT_MODEL_ID))
    planning_info.setdefault("hf_endpoint", config.get("agent_llm", {}).get("hf_endpoint", QWEN_AGENT_HF_ENDPOINT))
    planning_info.setdefault("no_proxy", bool(config.get("agent_llm", {}).get("no_proxy", True)))
    planning_info.setdefault("available_tools", LOCAL_TOOL_SPECS)

    emit("expand", "Writing auto_task and prompt plan", 0.12)
    write_manifest(workdir / "auto_task.json", auto_task)
    write_manifest(workdir / "prompt_plan.json", prompt_plan)
    emit("plan", "Writing auto camera plan", 0.18)
    write_manifest(workdir / "camera_plan.json", camera_plan)

    if options.dry_run:
        workflow_summary = _run_auto_dry_workflow(
            config=config,
            options=options,
            workdir=workdir,
            auto_task=auto_task,
            prompt_plan=prompt_plan,
            progress=emit,
            tool_executor=tool_executor,
        )
    else:
        render_resolution, ai_resolution, samples = _quality_resolution(config, str(auto_task.get("quality_mode", options.quality_mode)), dry_run=False)
        workflow_mode, requested_source_mode = _workflow_source_mode(auto_task, options, config)
        emit("camera", "Using auto camera plan for Blender fixed-view render", 0.24)
        camera_state = CameraState.from_payload(camera_plan.get("camera_state")).to_dict()
        max_generations = max(1, int(auto_task["num_candidates_per_view"]) * min(3, int(auto_task["output_views"])) + int(auto_task.get("max_retries", 0)))
        workflow_options = WorkflowOptions(
            prompt=prompt_plan["render_prompt"],
            workdir=workdir,
            source_mode=workflow_mode,
            model_path=options.model_path,
            reference_image=options.reference_image,
            model_backend=str(config.get("auto_agent", {}).get("model_backend", "sample")),
            config_path=options.config_path,
            views=max(3, int(auto_task.get("output_views", 3))),
            render_resolution=render_resolution,
            render_samples=samples,
            ai_width=ai_resolution,
            ai_height=ai_resolution,
            candidates=int(auto_task["num_candidates_per_view"]),
            seed=options.seed,
            steps=int(config.get("models", {}).get(options.backend_model_key or config.get("agent", {}).get("default_model_key", "flux2_klein_4b"), {}).get("steps", config.get("ai", {}).get("steps", 4))),
            negative_prompt=prompt_plan.get("negative_prompt", ""),
            model_key=options.backend_model_key or config.get("auto_agent", {}).get("default_model_key") or config.get("agent", {}).get("default_model_key", "flux2_klein_4b"),
            agent_render=True,
            agent_max_generations=max_generations,
            agent_target_view="view_locked",
            agent_expand_views=int(auto_task.get("output_views", 3)) > 1,
            camera=camera_state,
        )
        workflow_summary = tool_executor.run(
            "execute_workflow",
            {"source_mode": workflow_mode, "views": workflow_options.views, "model_key": workflow_options.model_key, "workdir": workdir},
            lambda: run_workflow(workflow_options, progress=lambda stage, message, fraction: emit(stage, message, 0.24 + fraction * 0.72)),
        )
        workflow_summary["source_mode_requested"] = requested_source_mode

    if workflow_summary.get("model_path") and not workflow_summary.get("mesh_sanity"):
        metadata_path = write_manifest(
            workdir / "mesh" / "metadata.json",
            {
                "type": "mesh_metadata",
                "source_mode": workflow_summary.get("source_mode", ""),
                "requested_source_mode": workflow_summary.get("source_mode_requested", auto_task.get("source_mode", "")),
                "model_path": workflow_summary.get("model_path", ""),
                "prompt": prompt_plan["base_prompt"],
                "created_by": "execute_workflow" if not options.dry_run else "generate_or_load_3d",
            },
        )
        sanity_args = {
            "model_path": workflow_summary["model_path"],
            "source_mode": str(workflow_summary.get("source_mode", "")),
            "expected_views": int(auto_task.get("output_views", 3)),
        }
        sanity = tool_executor.run("mesh_quality_check", sanity_args, lambda: mesh_sanity_report(**sanity_args))
        sanity_path = write_manifest(workdir / "mesh" / "sanity.json", sanity)
        workflow_summary["mesh_metadata"] = str(metadata_path)
        workflow_summary["mesh_sanity"] = str(sanity_path)

    emit("score", "Collecting structure and multiview scores", 0.94)
    agent_summary = workflow_summary.get("agent_summary") or workflow_summary.get("summary") or {}
    if not options.dry_run:
        tool_executor.run(
            "structure_scoring",
            {"agent_report": workflow_summary.get("agent_report", "")},
            lambda: {
                "structure_scores": agent_summary.get("structure_scores", {}),
                "multiview_scores": agent_summary.get("multiview_scores", {}),
                "retry_decisions": agent_summary.get("retry_decisions", []),
            },
        )
    retry_decisions = list(agent_summary.get("retry_decisions", [])) if isinstance(agent_summary, dict) else []
    retry_message = f"Retry policy recorded {len(retry_decisions)} decision(s)" if retry_decisions else "Retry policy found no blocking score failures"
    emit("retry", retry_message, 0.96)

    emit("package", "Packaging auto agent outputs", 0.97)
    package_started = time.time()
    summary = _package_auto_outputs(
        workdir=workdir,
        auto_task=auto_task,
        prompt_plan=prompt_plan,
        camera_plan=camera_plan,
        workflow_summary=workflow_summary,
        planning_info=planning_info,
        log_path=log_path,
        started=started,
        tool_calls=tool_executor.calls,
    )
    tool_executor.calls.append({
        "tool": "package_outputs",
        "args": {"workdir": str(workdir)},
        "status": "complete",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(package_started)),
        "elapsed_seconds": round(time.time() - package_started, 3),
        "result": _summarize_tool_result(summary),
    })
    tool_executor.run(
        "visual_judgement",
        {"visual_judgement": summary.get("visual_judgement", "")},
        lambda: read_manifest(summary["visual_judgement"]),
    )
    write_manifest(summary["tool_calls"], {"tools": LOCAL_TOOL_SPECS, "calls": tool_executor.calls})
    summary["capabilities"]["tool_execution"]["tool_call_count"] = len(tool_executor.calls)
    summary["capabilities"]["tool_execution"]["executed_tool_names"] = [call["tool"] for call in tool_executor.calls]
    write_manifest(workdir / "auto_summary.json", summary)
    emit("complete", f"Auto Agent finished with status={summary['status']}", 1.0)
    return summary
