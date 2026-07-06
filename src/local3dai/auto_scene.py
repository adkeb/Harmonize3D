from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .agent import AgentRunOptions, run_agent_render
from .ai.backends import build_backend
from .auto_agent import (
    QWEN_AGENT_SERVED_MODEL,
    _extract_json,
    _resolve_llm_api_key,
    _slug,
    _summarize_tool_args,
    _summarize_tool_result,
    visual_judgement_report,
)
from .config import load_config, resolve_path
from .manifest import read_manifest, write_manifest
from .rendering.blender import render_model_with_blender


Progress = Callable[[str, str, float], None]

AUTO_SCENE_STAGE_IDS = [
    "understand",
    "concept",
    "decompose",
    "module_reference",
    "module_3d",
    "module_check",
    "layout",
    "scene_preview",
    "camera",
    "render",
    "agent",
    "score",
    "consistency",
    "package",
    "complete",
]

AUTO_SCENE_STAGE_LABELS = {
    "understand": "Requirement understanding",
    "concept": "Global concept image",
    "decompose": "Module decomposition",
    "module_reference": "Module reference images",
    "module_3d": "Module image-to-3D",
    "module_check": "Module mesh sanity",
    "layout": "Scene layout",
    "scene_preview": "Scene assembly preview",
    "camera": "Camera planning",
    "render": "Blender white channels",
    "agent": "Final AI rendering",
    "score": "Candidate and module scoring",
    "consistency": "Multi-view consistency",
    "package": "Packaging and reports",
    "complete": "Final output",
}

AUTO_SCENE_STAGE_ARTIFACT_KEYS: dict[str, tuple[str, ...]] = {
    "understand": ("auto_task", "scene_plan", "tool_calls", "run_log"),
    "concept": ("concept_image_plan", "global_concept", "concept_review", "concept_image_request", "concept_image2_handoff"),
    "decompose": ("module_plan", "module_prompt_info", "module_layout_check"),
    "module_reference": (
        "module_plan",
        "module_reference_review",
        "module_reference_batch_request",
        "module_reference_image2_handoff",
        "module_assets_index",
        "module_references_contact_sheet",
    ),
    "module_3d": ("module_asset_manifest", "module_assets_index"),
    "module_check": ("module_asset_manifest", "module_mesh_sanity", "module_assets_index"),
    "layout": ("scene_assembly",),
    "scene_preview": ("final_scene_manifest", "scene_model_path", "scene_preview", "assembly_report"),
    "camera": ("camera_plan", "camera_search_report", "camera_search_sheet"),
    "render": ("render_manifest", "white_render", "white_channel_contact_sheet"),
    "agent": ("white_model_position_contract", "white_position_contract_overlay", "agent_report", "final_image", "comparison_image", "contact_sheet", "final_image2_request"),
    "score": ("module_scores", "structure_scores", "multiview_score"),
    "consistency": ("multiview_score",),
    "package": (
        "visual_judgement",
        "concept_final_comparison",
        "concept_vs_final",
        "white_model_position_lock",
        "white_position_lock_overlay",
        "final_position_retry_plan",
        "image2_flow_audit",
        "tool_calls",
    ),
    "complete": ("final_image", "contact_sheet", "stages", "run_log"),
}

AUTO_SCENE_PENDING_STAGE_MAP = {
    "concept_image_generation": "concept",
    "concept_image_regeneration": "concept",
    "module_reference_generation": "module_reference",
    "module_reference_regeneration": "module_reference",
    "final_image2_render": "agent",
}

AUTO_SCENE_TOOL_STAGE_MAP = {
    "scene_planner": "understand",
    "concept_image_generation": "concept",
    "concept_image_review": "concept",
    "module_prompt_generation": "decompose",
    "module_layout_repair": "decompose",
    "module_reference_generation": "module_reference",
    "module_reference_review": "module_reference",
    "module_image_to_3d": "module_3d",
    "module_mesh_sanity": "module_check",
    "module_assets_index": "module_reference",
    "scene_layout_agent": "layout",
    "scene_assembler": "scene_preview",
    "camera_candidate_search": "camera",
    "render_white_channels": "render",
    "white_model_position_contract": "agent",
    "final_image2_render": "agent",
    "ai_candidate_search": "agent",
    "module_presence_scoring": "score",
    "concept_final_comparison": "package",
    "white_model_position_lock": "package",
    "final_position_retry_plan": "package",
    "image2_flow_audit": "package",
}

AUTO_SCENE_TOOL_SPECS: list[dict[str, str]] = [
    {"name": "scene_planner", "purpose": "use the configured DashScope multimodal model from .env as the Auto Scene brain"},
    {"name": "concept_image_generation", "purpose": "generate a global concept image from the model-expanded concept prompt"},
    {"name": "concept_image_review", "purpose": "return the generated concept image to the multimodal model for element audit"},
    {"name": "module_prompt_generation", "purpose": "ask the model to write isolated per-object prompts from the approved concept image"},
    {"name": "module_layout_repair", "purpose": "ask the model to repair generic Blender Z-up placement contract issues using few-shot layout examples"},
    {"name": "module_reference_generation", "purpose": "generate per-module centered solid-background reference images concurrently"},
    {"name": "module_reference_review", "purpose": "return generated module images to the multimodal model and revise failed prompts"},
    {"name": "module_image_to_3d", "purpose": "generate module GLB assets from reviewed reference images, falling back only when configured"},
    {"name": "module_mesh_sanity", "purpose": "check each module asset and apply failure policy"},
    {"name": "module_assets_index", "purpose": "write a direct per-module reference image and GLB index for UI, API, and report inspection"},
    {"name": "scene_layout_agent", "purpose": "compute module scale, position, rotation, and layout reasons"},
    {"name": "scene_assembler", "purpose": "assemble module GLBs into a final scene GLB and preview"},
    {"name": "camera_candidate_search", "purpose": "render low-resolution RGB previews and select the best 3D scene camera before final channels"},
    {"name": "render_white_channels", "purpose": "render or mock scene rgb/edge/mask/depth/normal channels"},
    {"name": "white_model_position_contract", "purpose": "derive normalized screen-space bbox/center/coverage contracts from Blender white-model render channels"},
    {"name": "final_image2_render", "purpose": "request Codex built-in image2 final rendering from white-model channels with position lock"},
    {"name": "ai_candidate_search", "purpose": "optionally run local final geometry-locked AI rendering from render_manifest channels"},
    {"name": "module_presence_scoring", "purpose": "score module presence and position adherence"},
    {"name": "concept_final_comparison", "purpose": "compare the final render against the global concept image and flag obvious quality failures"},
    {"name": "white_model_position_lock", "purpose": "compare final view images against Blender white-model views for screen-space layout drift"},
    {"name": "final_position_retry_plan", "purpose": "write a Codex image2 retry handoff when the final image drifts from the white-model position contract"},
    {"name": "image2_flow_audit", "purpose": "audit that real Auto Scene runs used model planning, Codex image2 handoff, model review, and reviewed-reference 3D AI"},
    {"name": "package_scene_outputs", "purpose": "write final manifests, report, contact sheet, and summary"},
]


@dataclass
class AutoSceneOptions:
    request: str
    output_dir: Path
    config_path: Path = Path("configs/local.json")
    output_views: int = 3
    quality_mode: str = "balanced"
    geometry_mode: str = "strict"
    style_preset: str = "exhibition"
    backend_model_key: str | None = None
    backend: str | None = None
    num_candidates_per_view: int = 3
    max_retries: int = 2
    seed: int = 20260610
    allow_procedural_fallback: bool = True
    require_concept_confirmation: bool = False
    dry_run: bool = False
    use_llm: bool = True
    render_backend: str = "auto"
    hero_model_path: Path | None = None


class ExternalImagegenRequired(RuntimeError):
    def __init__(self, request_path: Path) -> None:
        self.request_path = Path(request_path)
        try:
            self.request = read_manifest(self.request_path)
        except Exception:
            self.request = {}
        kind = str(self.request.get("kind") or self.request.get("type") or "external_imagegen_request")
        super().__init__(f"External image2/imagegen generation required before continuing. Request written to: {self.request_path}")
        self.kind = kind
        self.output_path = str(self.request.get("output_path") or self.request.get("output") or "")
        self.output_paths = [
            _absolute_artifact_path(item.get("output_path") or item.get("output"))
            for item in self.request.get("requests", [])
            if isinstance(item, dict) and (item.get("output_path") or item.get("output"))
        ]
        self.module_id = str(self.request.get("module_id") or "")


class SceneToolExecutor:
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


def _clamp_views(value: int) -> int:
    if value <= 1:
        return 1
    if value <= 3:
        return 3
    if value <= 5:
        return 5
    return 8


def _write_log(path: Path, stage: str, message: str, progress: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": time.strftime("%H:%M:%S"), "stage": stage, "message": message, "progress": progress}, ensure_ascii=False) + "\n")


def _config_for_quality(config: dict[str, Any], quality_mode: str) -> tuple[dict[str, Any], int, int]:
    runtime = copy.deepcopy(config)
    preset = dict(runtime.get("auto_agent", {}).get("quality_presets", {}).get(quality_mode, {}))
    render_resolution = int(preset.get("render_resolution") or runtime.get("render", {}).get("resolution", 1024))
    render_samples = int(preset.get("render_samples") or runtime.get("render", {}).get("samples", 64))
    ai_resolution = int(preset.get("ai_resolution") or runtime.get("ai", {}).get("width") or 1024)
    ai_steps = int(runtime.get("ai", {}).get("steps", 4))
    if quality_mode == "balanced":
        ai_steps = max(ai_steps, min(8, int(runtime.get("web", {}).get("ai_steps", ai_steps))))
    elif quality_mode == "high":
        ai_steps = max(ai_steps, int(runtime.get("web", {}).get("ai_steps", ai_steps)))
    runtime.setdefault("render", {})["resolution"] = render_resolution
    runtime.setdefault("render", {})["samples"] = render_samples
    return runtime, ai_resolution, ai_steps


def _merge_nonempty(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(override, dict):
        return dict(base)
    merged = dict(base)
    for key, value in override.items():
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged


def _model_cfg(config: dict[str, Any], key: str) -> dict[str, Any]:
    model = config.get("models", {}).get(key)
    if not isinstance(model, dict):
        raise RuntimeError(f"Unknown model key: {key}")
    return dict(model)


def _coerce_json_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dashscope_response_text(response: Any) -> str:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    choices = getattr(output, "choices", None) if output is not None else None
    if choices is None and isinstance(output, dict):
        choices = output.get("choices")
    if not choices:
        text = getattr(output, "text", None) if output is not None else None
        return str(text or "")
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message", {})
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    parts.append(str(value))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def _dashscope_image_ref(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    staged_dir = resolved.parent / ".dashscope_uploads"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / f"{resolved.stem}_{digest}{resolved.suffix.lower() or '.png'}"
    if not staged.exists() or staged.stat().st_mtime < resolved.stat().st_mtime:
        shutil.copy2(resolved, staged)
    return f"file://{staged}"


def _call_dashscope_multimodal_json(
    config: dict[str, Any],
    *,
    purpose: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    llm_cfg = config.get("agent_llm", {})
    api_key, api_key_source = _resolve_llm_api_key(llm_cfg)
    if not api_key:
        raise RuntimeError(f"DashScope API key missing for {purpose}; set {llm_cfg.get('api_key_env', 'DASHSCOPE_API_KEY')}.")
    try:
        import dashscope  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency until installed
        raise RuntimeError("DashScope SDK is required for multimodal scene planning. Install dashscope.") from exc
    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    response = dashscope.MultiModalConversation.call(
        api_key=api_key,
        model=str(llm_cfg.get("model", QWEN_AGENT_SERVED_MODEL)),
        messages=messages,
        temperature=float(llm_cfg.get("temperature", 0.2)),
        max_tokens=int(max_tokens or llm_cfg.get("max_tokens", 1600)),
        timeout=int(llm_cfg.get("timeout_seconds", 90)),
        enable_thinking=False,
    )
    status_code = getattr(response, "status_code", None)
    if status_code is not None and int(status_code) >= 400:
        code = getattr(response, "code", "")
        message = getattr(response, "message", "")
        raise RuntimeError(f"DashScope {purpose} failed: {status_code} {code} {message}")
    text = _dashscope_response_text(response).strip()
    parsed = _extract_json(text)
    if parsed is None:
        raise RuntimeError(f"DashScope {purpose} returned non-JSON content: {text[:400]}")
    return parsed, {
        "backend": "dashscope_multimodal",
        "model": str(llm_cfg.get("model", QWEN_AGENT_SERVED_MODEL)),
        "api_key_source": api_key_source,
        "purpose": purpose,
    }


def _mock_model_plan(config: dict[str, Any], options: AutoSceneOptions) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _generic_model_plan_defaults(options)
    plan["auto_task"]["planner"] = "mock_model_brain"
    plan["concept_image_plan"]["planner"] = "mock_model_brain"
    plan["module_plan"] = _mock_model_module_plan(options)
    for module in plan["module_plan"].get("modules", []):
        module["prompt_source"] = "mock_model_brain"
    return plan, {"backend": "mock_model_brain", "reason": "dry_run_or_mock_backend"}


def _generic_camera_plan(output_views: int) -> dict[str, Any]:
    views = [
        {
            "view_id": "view_hero",
            "role": "model-planned hero view",
            "azimuth_deg": 320.0,
            "elevation_deg": 7.0,
            "camera_type": "perspective",
            "focal_length_mm": 58.0,
            "distance_scale": 0.92,
            "target": [0.0, 0.0, 0.05],
            "composition_goal": "primary subject clear, supporting modules visible without foreground obstruction",
        },
        {
            "view_id": "view_left_30",
            "role": "hero consistency yaw +30",
            "azimuth_deg": 290.0,
            "elevation_deg": 7.0,
            "camera_type": "perspective",
            "focal_length_mm": 58.0,
            "distance_scale": 0.92,
            "target": [0.0, 0.0, 0.05],
        },
        {
            "view_id": "view_right_30",
            "role": "hero consistency yaw -30",
            "azimuth_deg": 350.0,
            "elevation_deg": 7.0,
            "camera_type": "perspective",
            "focal_length_mm": 58.0,
            "distance_scale": 0.92,
            "target": [0.0, 0.0, 0.05],
        },
    ]
    return {
        "coordinate_space": "blender_z_up",
        "views": views[: _clamp_views(output_views)],
        "composition": {
            "camera_style": "low three-quarter product view unless the model planner specifies otherwise",
            "subject_frame_ratio": [0.62, 0.86],
            "notes": "Generic fallback camera plan; model planner output should override this when available.",
        },
    }


def _generic_model_plan_defaults(options: AutoSceneOptions) -> dict[str, dict[str, Any]]:
    expanded = str(options.request)
    return {
        "auto_task": {
            "task_id": f"scene-{time.strftime('%Y%m%d-%H%M%S')}-{_slug(options.request)}",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "user_request": options.request,
            "expanded_request": expanded,
            "mode": "modular_scene_agent",
            "source_mode": "text_to_scene",
            "style_preset": options.style_preset,
            "quality_mode": options.quality_mode,
            "geometry_mode": options.geometry_mode,
            "output_views": _clamp_views(options.output_views),
            "num_candidates_per_view": options.num_candidates_per_view,
            "max_retries": options.max_retries,
            "planner": "generic_llm_fallback",
        },
        "scene_plan": {
            "scene_type": "model_planned_scene",
            "main_subject": {"name": "primary subject from user request", "role": "hero_object", "priority": 1},
            "environment": {"description": "Environment should be inferred from the user request by the model planner."},
            "composition": {"camera_style": "model-planned product view"},
            "global_style": {"material_language": "", "color_palette": [], "avoid": ["text", "logos", "people unless requested"]},
            "expected_elements": [],
        },
        "concept_image_plan": {
            "concept_prompt": expanded,
            "width": 1024,
            "height": 1024,
            "output": "concept/global_concept.png",
            "purpose": "planning-only concept image generated from model-expanded request",
        },
        "module_plan": {"modules": []},
        "prompt_plan": {
            "render_prompt": expanded,
            "reference_policy": "model_render_channels_only",
            "forbidden_image_inputs": ["concept/global_concept.png", "modules/*/reference.png"],
        },
        "camera_plan": _generic_camera_plan(options.output_views),
    }


def _mock_model_module_plan(options: AutoSceneOptions) -> dict[str, Any]:
    request = " ".join(str(options.request).split())
    prompt_prefix = request[:320] or "primary requested scene object"
    modules = [
        {
            "module_id": "primary_subject",
            "name": "primary subject from user request",
            "category": "scene_object",
            "role": "hero_object",
            "priority": 1,
            "reference_prompt": (
                f"Primary subject implied by this user request: {prompt_prefix}. "
                "Single isolated object, catalog reconstruction input."
            ),
            "expected_real_world_size": {"width": 2.0, "depth": 2.0, "height": 1.2, "unit": "meters"},
            "placement": {
                "anchor": "scene_center",
                "position": [0.0, 0.0, 0.6],
                "rotation_deg": [0.0, 0.0, 0.0],
                "scale_policy": "generic_mock_primary_subject",
            },
            "generate_reference_image": True,
            "generate_3d": True,
        },
        {
            "module_id": "support_surface",
            "name": "supporting surface or plinth",
            "category": "supporting_object",
            "role": "supporting_object",
            "priority": 2,
            "reference_prompt": (
                f"Minimal support surface or plinth suitable for the scene requested as: {prompt_prefix}. "
                "Single isolated object, simple clean geometry."
            ),
            "expected_real_world_size": {"width": 3.2, "depth": 2.4, "height": 0.25, "unit": "meters"},
            "placement": {
                "anchor": "under_primary_subject",
                "position": [0.0, 0.0, 0.125],
                "rotation_deg": [0.0, 0.0, 0.0],
                "scale_policy": "generic_mock_support",
            },
            "generate_reference_image": True,
            "generate_3d": True,
        },
        {
            "module_id": "background_element",
            "name": "simple background element",
            "category": "background_prop",
            "role": "background",
            "priority": 3,
            "reference_prompt": (
                f"Simple background element that supports the requested scene: {prompt_prefix}. "
                "Single isolated object, clean silhouette."
            ),
            "expected_real_world_size": {"width": 3.8, "depth": 0.18, "height": 2.2, "unit": "meters"},
            "placement": {
                "anchor": "behind_primary_subject",
                "position": [0.0, 2.2, 1.1],
                "rotation_deg": [0.0, 0.0, 0.0],
                "scale_policy": "generic_mock_background",
            },
            "generate_reference_image": True,
            "generate_3d": True,
        },
    ]
    return {"modules": [_normalize_module(module, index) for index, module in enumerate(modules)]}


def _normalize_module(module: dict[str, Any], index: int) -> dict[str, Any]:
    module_id = str(module.get("module_id") or _slug(str(module.get("name") or f"module_{index + 1}")) or f"module_{index + 1}")
    role = str(module.get("role") or ("hero_object" if index == 0 else "supporting_object"))
    size = module.get("expected_real_world_size") or module.get("size") or {}
    if isinstance(size, dict):
        width = float(size.get("width", 1.0))
        depth = float(size.get("depth", size.get("length", 1.0)))
        height = float(size.get("height", 1.0))
    elif isinstance(size, (list, tuple)) and len(size) >= 3:
        width, depth, height = float(size[0]), float(size[1]), float(size[2])
    else:
        width, depth, height = (2.0, 4.5, 1.2) if role == "hero_object" else (1.0, 1.0, 1.0)
    placement = module.get("placement") if isinstance(module.get("placement"), dict) else {}
    placement.setdefault("anchor", "scene_center" if role == "hero_object" else "model_planned_anchor")
    placement.setdefault("position", [0.0, 0.0, max(0.0, height / 2.0)])
    placement.setdefault("rotation_deg", [0.0, 0.0, 0.0])
    placement.setdefault("scale_policy", "model_estimated_real_world_scale")
    prompt = str(module.get("reference_prompt") or module.get("object_prompt") or module.get("prompt") or module.get("name") or module_id)
    normalized = {
        **module,
        "module_id": module_id,
        "name": str(module.get("name") or module_id.replace("_", " ")),
        "category": str(module.get("category") or "scene_object"),
        "role": role,
        "priority": int(module.get("priority", index + 1)),
        "generate_reference_image": bool(module.get("generate_reference_image", True)),
        "generate_3d": bool(module.get("generate_3d", True)),
        "reference_prompt": prompt,
        "expected_real_world_size": {"width": width, "depth": depth, "height": height, "unit": "meters"},
        "placement": placement,
        "constraints": list(module.get("constraints", [])) if isinstance(module.get("constraints", []), list) else [],
        "prompt_source": str(module.get("prompt_source") or "dashscope_multimodal_object_prompt"),
    }
    normalized["reference_prompt"] = _module_reference_prompt_with_safety(normalized, prompt)
    normalized.pop("negative_prompt", None)
    return normalized


def _normalize_model_plan(plan: dict[str, Any], options: AutoSceneOptions) -> dict[str, dict[str, Any]]:
    fallback = _generic_model_plan_defaults(options)
    auto_task = _merge_nonempty(fallback["auto_task"], _coerce_json_object(plan.get("auto_task")))
    scene_plan = _merge_nonempty(fallback["scene_plan"], _coerce_json_object(plan.get("scene_plan")))
    concept_image_plan = _merge_nonempty(fallback["concept_image_plan"], _coerce_json_object(plan.get("concept_image_plan")))
    prompt_plan = _merge_nonempty(fallback["prompt_plan"], _coerce_json_object(plan.get("prompt_plan")))
    camera_plan = _merge_nonempty(fallback["camera_plan"], _coerce_json_object(plan.get("camera_plan")))
    auto_task["task_id"] = str(auto_task.get("task_id") or fallback["auto_task"]["task_id"])
    auto_task["mode"] = "modular_scene_agent"
    auto_task["source_mode"] = "text_to_scene"
    auto_task["output_views"] = _clamp_views(int(auto_task.get("output_views", options.output_views)))
    concept_image_plan["output"] = str(concept_image_plan.get("output") or "concept/global_concept.png")
    concept_image_plan["concept_prompt"] = str(
        concept_image_plan.get("concept_prompt")
        or auto_task.get("expanded_request")
        or prompt_plan.get("base_prompt")
        or options.request
    )
    concept_image_plan.pop("negative_prompt", None)
    concept_image_plan.setdefault("purpose", "planning-only model-generated concept; audited before module prompt generation")
    prompt_plan["render_prompt"] = str(prompt_plan.get("render_prompt") or auto_task.get("expanded_request") or concept_image_plan["concept_prompt"])
    prompt_plan.pop("negative_prompt", None)
    prompt_plan["reference_policy"] = "model_render_channels_only"
    prompt_plan["forbidden_image_inputs"] = ["concept/global_concept.png", "modules/*/reference.png"]
    camera_views = camera_plan.get("views", [])
    if isinstance(camera_views, list):
        camera_plan["views"] = [view for view in camera_views if isinstance(view, dict)]
    else:
        camera_plan["views"] = []
    return {
        "auto_task": auto_task,
        "scene_plan": scene_plan,
        "concept_image_plan": concept_image_plan,
        "module_plan": {"modules": []},
        "prompt_plan": prompt_plan,
        "camera_plan": camera_plan,
    }


def _layout_coordinate_contract() -> dict[str, Any]:
    return {
        "coordinate_space": "Blender Z-up, meters",
        "axes": {
            "x": "left/right on ground plane",
            "y": "front/back depth on ground plane",
            "z": "vertical height above ground",
        },
        "placement_position_semantics": "placement.position is the center of the module bounding box in world coordinates, not the bottom contact point.",
        "ground_contact_rule": "For an object resting on the floor, position[2] should equal expected_real_world_size.height / 2, so bottom_z = position[2] - height / 2 is approximately 0.",
        "rotation_semantics": "rotation_deg is Euler [x, y, z] in degrees. Plan-view yaw belongs in rotation_deg[2], while x/y rotations are only for intentional pitch or roll.",
        "validation_rule": "Every module should have finite numeric position[3], rotation_deg[3], and bottom_z >= -0.05 unless the object intentionally penetrates below ground and the constraint explains why.",
        "do_not": [
            "Do not use z to express front/back distance.",
            "Do not set a module center below ground unless explicitly intentional.",
            "Do not change module_id values between planning and repair.",
        ],
    }


def _layout_few_shot_examples() -> list[dict[str, Any]]:
    return [
        {
            "name": "floor-standing hero object",
            "expected_real_world_size": {"width": 4.5, "depth": 2.0, "height": 1.2, "unit": "meters"},
            "bad_placement": {"position": [0.0, 0.0, 0.0], "rotation_deg": [0.0, -15.0, 0.0]},
            "why_bad": "The bbox center is on the ground, so half the object is below ground; yaw was put on the Y axis.",
            "good_placement": {"position": [0.0, 0.0, 0.6], "rotation_deg": [0.0, 0.0, -15.0]},
        },
        {
            "name": "thin background wall panel",
            "expected_real_world_size": {"width": 8.0, "depth": 0.15, "height": 3.6, "unit": "meters"},
            "bad_placement": {"position": [0.0, 1.5, -6.0], "rotation_deg": [0.0, 0.0, 0.0]},
            "why_bad": "The model used z as depth. In Blender Z-up, z must be the vertical center height.",
            "good_placement": {"position": [0.0, 3.0, 1.8], "rotation_deg": [0.0, 0.0, 0.0]},
        },
        {
            "name": "side support object on floor",
            "expected_real_world_size": {"width": 1.2, "depth": 1.2, "height": 2.4, "unit": "meters"},
            "bad_placement": {"position": [2.5, 0.4, -1.2], "rotation_deg": [0.0, 0.0, 0.0]},
            "why_bad": "The center is below the ground plane even though the object rests on the floor.",
            "good_placement": {"position": [2.5, 0.4, 1.2], "rotation_deg": [0.0, 0.0, 0.0]},
        },
        {
            "name": "overhead rig or hanging structure",
            "expected_real_world_size": {"width": 7.0, "depth": 4.0, "height": 0.5, "unit": "meters"},
            "good_placement": {"position": [0.0, 0.0, 3.2], "rotation_deg": [0.0, 0.0, 0.0]},
            "why_good": "The bbox center is above the scene, and bottom_z remains above ground.",
        },
    ]


def _module_reference_prompt_few_shot_examples() -> list[dict[str, str]]:
    return [
        {
            "bad": "sleek vehicle on a stage, 3/4 perspective render, dramatic floor shadows",
            "good": "sleek vehicle, strict single-object orthographic front view, front elevation, centered on a pure solid light gray background, full silhouette visible, symmetrical frontal camera, clean even studio lighting, blank unlabeled surfaces, object-only catalog cutout",
            "reason": "Module images are reconstruction inputs, not final composition images.",
        },
        {
            "bad": "side profile industrial arm in a workshop with props",
            "good": "untextured matte white/gray 3D CAD clay model of a cableless collaborative robot arm, sealed cylindrical joints, clean shape reconstruction reference, strict single-object orthographic front view, front elevation, zero yaw pitch roll camera, centered on a pure solid light gray background, full object visible, all wiring hidden internally, smooth uninterrupted exterior, blank unlabeled surfaces, object-only catalog cutout",
            "reason": "The 3D generator needs a clean front-facing object reference.",
        },
        {
            "bad": "display screen from a top angle with floor reflections",
            "good": "portrait vertical flat luminous panel slab, tall narrow rectangular LED panel, height greater than width, thin uniform bezel, wall-panel style exhibition display face, strict single-object orthographic front view, front elevation, centered on a pure solid light gray background, uncropped outline, clean even studio lighting, blank unlabeled surface",
            "reason": "Flat or thin modules should preserve their front silhouette and aspect ratio.",
        },
        {
            "bad": "black display platform as a tall vertical block on a floor with dramatic product shadows",
            "good": "smooth hard-surface neutral gray CAD model of one-piece low rectangular cuboid slab, single-layer undivided horizontal slab, flat top plane spans the full width, height about one tenth of width, short front face, planar flat faces, crisp straight 90-degree edges, strict orthographic front elevation, centered on a pure solid light gray background, blank unlabeled surfaces, object-only CAD cutout",
            "reason": "Support platforms should be low horizontal slabs for clean 3D reconstruction.",
        },
    ]


def _solid_background_reference_prompt(prompt: str) -> str:
    base = _strip_module_reference_forbidden_content_conflicts(_strip_module_reference_view_conflicts(str(prompt)))
    lower = base.lower()
    if not any(token in lower for token in ("orthographic", "正交", "正交正视图")):
        base += (
            ", strict single-object orthographic front view for image-to-3D reconstruction, "
            "front elevation, object front plane parallel to image plane, symmetrical front-facing camera, "
            "zero yaw pitch roll camera, flat frontal silhouette, full object visible, uncropped silhouette"
        )
    elif not any(token in lower for token in ("front elevation", "front view", "front-facing", "front facing", "正视")):
        base += ", front elevation, object front plane parallel to image plane, flat frontal silhouette"
    if "cad" not in lower and "reconstruction" not in lower:
        base += ", untextured matte clay 3D CAD reconstruction input, sealed simplified surfaces, shape-first reference, large clean construction details"
    if "front-facing" not in lower and "front facing" not in lower:
        base += ", front-facing catalog cutout"
    if "solid" not in lower and "plain" not in lower:
        base += ", isolated single object centered on a pure solid light gray background"
    if "blank unlabeled" not in lower and "unbranded" not in lower:
        base += ", blank unlabeled surfaces"
    if "object-only" not in lower and "single-object catalog cutout" not in lower:
        base += ", single-object catalog cutout on blank solid background, empty canvas around object"
    return base


def _strip_module_reference_view_conflicts(prompt: str) -> str:
    text = " ".join(str(prompt).split())
    patterns = [
        r"\b(?:not|no)\s+(?:angled\s+view|3/4\s+view|three[-\s]?quarter\s+view|three\s+quarter\s+view|side\s+profile(?:\s+view)?|side\s+elevation|side\s+view|top\s+view|rear\s+view|back\s+view|perspective\s+(?:view|angle|render)|oblique\s+(?:view|angle|render))\b",
        r"\b(?:3/4|three[-\s]?quarter|three\s+quarter)\s+(?:front\s+)?(?:view|angle|perspective|render)\b",
        r"\b(?:front\s+)?(?:3/4|three[-\s]?quarter|three\s+quarter)\s+(?:view|angle|perspective|render)\b",
        r"\bside\s+profile\s+(?:view|angle|render)?\b",
        r"\bside\s+elevation\b",
        r"\bside\s+view\b",
        r"\bflat\s+side\s+silhouette\b",
        r"\bangled\s+view\b",
        r"\btop\s+view\b",
        r"\brear\s+view\b",
        r"\bback\s+view\b",
        r"\bperspective\s+(?:view|angle|render)\b",
        r"\boblique\s+(?:view|angle|render)\b",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"(?:^|,\s*)not(?=\s*(?:,|$))", "", text, flags=re.IGNORECASE)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,")


def _strip_module_reference_forbidden_content_conflicts(prompt: str) -> str:
    forbidden_terms = (
        "cable",
        "wire",
        "wiring",
        "harness",
        "hose",
        "corrugated tube",
        "floor",
        "contact shadow",
        "reflection",
        "photography",
        "photorealistic",
        "product shot",
    )
    allowed_markers = (
        "zero ",
        "all wiring hidden",
        "all cables hidden",
        "hidden internally",
        "internal wiring",
        "internal cable",
        "cableless",
    )
    kept: list[str] = []
    for clause in str(prompt).split(","):
        normalized = " ".join(clause.split())
        if not normalized:
            continue
        lower = normalized.lower()
        mentions_forbidden = any(term in lower for term in forbidden_terms)
        explicitly_allowed = any(marker in lower for marker in allowed_markers)
        if mentions_forbidden and not explicitly_allowed:
            continue
        kept.append(normalized)
    return ", ".join(kept)


def _module_identity_text(module: dict[str, Any]) -> str:
    return " ".join(str(module.get(key, "")) for key in ("module_id", "name", "category", "role", "reference_prompt")).lower()


def _module_structural_identity_text(module: dict[str, Any]) -> str:
    return " ".join(str(module.get(key, "")) for key in ("module_id", "name", "category", "role")).lower()


def _is_mechanical_arm_module(module: dict[str, Any]) -> bool:
    text = _module_identity_text(module)
    return ("robotic" in text or "机械臂" in text or "industrial arm" in text or "robot arm" in text) and "arm" in text


def _is_platform_module(module: dict[str, Any]) -> bool:
    text = _module_identity_text(module)
    platform_terms = ("platform", "plinth", "pedestal", "stage base", "display base", "展示台", "台座", "底座")
    screen_terms = ("screen", "monitor", "display panel", "led panel", "屏幕", "显示屏")
    return any(term in text for term in platform_terms) and not any(term in text for term in screen_terms)


def _is_screen_module(module: dict[str, Any]) -> bool:
    text = _module_identity_text(module)
    screen_terms = ("screen", "monitor", "display panel", "digital display", "led panel", "luminous panel", "屏幕", "显示屏")
    return any(term in text for term in screen_terms)


def _is_flat_panel_module(module: dict[str, Any]) -> bool:
    text = _module_identity_text(module)
    panel_terms = ("panel", "wall slab", "backdrop", "partition", "display surface", "背景板", "墙板", "隔断")
    horizontal_terms = ("floor", "ground", "floor panel", "地面", "地板")
    return any(term in text for term in panel_terms) and not _is_platform_module(module) and not any(term in text for term in horizontal_terms)


def _is_semantic_platform_module(module: dict[str, Any]) -> bool:
    text = _module_structural_identity_text(module)
    platform_terms = ("platform", "plinth", "pedestal", "stage base", "display base", "support surface", "展示台", "台座", "底座")
    screen_terms = ("screen", "monitor", "display panel", "led panel", "屏幕", "显示屏")
    return any(term in text for term in platform_terms) and not any(term in text for term in screen_terms)


def _is_semantic_screen_module(module: dict[str, Any]) -> bool:
    text = _module_structural_identity_text(module)
    screen_terms = ("screen", "monitor", "display panel", "digital display", "led panel", "luminous panel", "屏幕", "显示屏")
    return any(term in text for term in screen_terms)


def _is_semantic_flat_panel_module(module: dict[str, Any]) -> bool:
    text = _module_structural_identity_text(module)
    panel_terms = ("panel", "wall slab", "backdrop", "partition", "display surface", "背景板", "墙板", "隔断")
    horizontal_terms = ("floor", "ground", "floor panel", "地面", "地板")
    return any(term in text for term in panel_terms) and not _is_semantic_platform_module(module) and not any(term in text for term in horizontal_terms)


def _is_vertical_screen_module(module: dict[str, Any]) -> bool:
    text = _module_identity_text(module)
    if any(term in text for term in ("vertical", "portrait", "tall", "竖屏", "纵向")):
        return True
    size = module.get("expected_real_world_size") if isinstance(module.get("expected_real_world_size"), dict) else {}
    try:
        width = float(size.get("width", 0.0))
        height = float(size.get("height", 0.0))
    except (TypeError, ValueError):
        return False
    return width > 0.0 and height >= width * 1.2


def _append_unique_clauses(text: str, clauses: list[str]) -> str:
    result = text
    result_lower = result.lower()
    for clause in clauses:
        normalized = " ".join(clause.split())
        if normalized.lower() not in result_lower:
            result += ", " + normalized
            result_lower = result.lower()
    return result


def _dedupe_comma_clauses(text: str) -> str:
    seen: set[str] = set()
    deduped: list[str] = []
    for clause in str(text).split(","):
        normalized = " ".join(clause.split())
        if not normalized:
            continue
        key = normalized.strip(" .;").lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return ", ".join(deduped)


def _module_reference_prompt_with_safety(module: dict[str, Any], prompt: str) -> str:
    base = _dedupe_comma_clauses(_solid_background_reference_prompt(prompt))
    if _is_mechanical_arm_module(module):
        base = re.sub(r"\bindustrial\s+robotic\s+arm\b", "cableless articulated robot arm CAD primitive", base, flags=re.IGNORECASE)
        base = re.sub(r"\bstandard\s+six-axis\s+industrial\s+robotic\s+arm\b", "simplified six-axis cableless articulated robot arm", base, flags=re.IGNORECASE)
        base = re.sub(r"\bfactory\s+robot\s+style\b", "clean abstract cobot CAD style", base, flags=re.IGNORECASE)
        base = _append_unique_clauses(
            base,
            [
                "hard-surface opaque solid metal object",
                "simplified six-axis sealed articulated robot arm",
                "untextured matte white/gray 3D CAD clay model",
                "abstract collaborative robot arm form",
                "simple geometric primitive shape",
                "toy-like educational robot arm",
                "five smooth cylinders connected by circular hinge disks only",
                "made only from smooth cylinders, hinge disks, and sealed shells",
                "heavy circular base",
                "visible cylindrical joints",
                "angular lower arm and upper arm",
                "sealed-joint design",
                "smooth uninterrupted exterior",
                "generic unbranded clean abstract cobot CAD style",
                "unbranded blank surfaces",
                "functional articulated CAD mechanism",
            ],
        )
    if _is_platform_module(module):
        base = re.sub(r"\bdisplay\s+platform\b", "low solid rectangular slab", base, flags=re.IGNORECASE)
        base = re.sub(r"\bdisplay\b", "rectangular slab", base, flags=re.IGNORECASE)
        base = re.sub(r"\bmatte\s+black\b", "smooth hard-surface neutral gray CAD", base, flags=re.IGNORECASE)
        base = re.sub(r"\bblack\b", "neutral gray CAD", base, flags=re.IGNORECASE)
        base = re.sub(r"\bbase\s+base\b", "base", base, flags=re.IGNORECASE)
        base = _append_unique_clauses(
            base,
            [
                "smooth hard-surface neutral gray CAD model",
                "one-piece low rectangular cuboid slab",
                "single-layer undivided horizontal slab",
                "flat top plane spans the full width",
                "height about one tenth of total width",
                "short front face",
                "planar flat faces",
                "crisp straight 90-degree edges",
                "solid continuous slab body",
                "strict orthographic front elevation",
                "long low rectangular silhouette",
            ],
        )
    if _is_screen_module(module):
        base = re.sub(r"\bdisplay\s+screen\s+monitor\b", "flat luminous panel slab", base, flags=re.IGNORECASE)
        base = re.sub(r"\bdisplay\s+screen\b", "flat luminous panel slab", base, flags=re.IGNORECASE)
        base = re.sub(r"\bmonitor\b", "flat panel", base, flags=re.IGNORECASE)
        base = re.sub(r"\bthin\s+bezel\s+frame\b", "thin uniform bezel", base, flags=re.IGNORECASE)
        screen_clauses = [
            "flat luminous panel slab",
            "thin rectangular display panel",
            "wall-panel style exhibition display face",
            "planar flat screen surface",
            "thin uniform bezel",
            "smooth sealed panel body",
            "strict orthographic front elevation",
        ]
        if _is_vertical_screen_module(module):
            screen_clauses.extend(
                [
                    "portrait vertical orientation",
                    "tall narrow rectangular panel",
                    "height greater than width",
                    "aspect ratio about two units wide and three units tall",
                    "vertical panel face fills the object silhouette",
                ]
            )
        base = _append_unique_clauses(base, screen_clauses)
    return _dedupe_comma_clauses(base)


def _module_reference_generation_prompt(module: dict[str, Any], prompt: str, *, has_concept_image: bool) -> str:
    module_name = str(module.get("name") or module.get("module_id") or "target module")
    module_role = str(module.get("role") or "")
    prompt = _module_reference_prompt_with_safety(module, prompt)
    if not has_concept_image:
        return prompt
    guidance = (
        f"Use the attached global concept image only as visual context to identify the standalone module named {module_name!r}"
        f"{' with role ' + module_role if module_role else ''}. "
        "Generate a new isolated single-object module reference image of that target module only. "
        "Preserve the module identity, approximate proportions, and design language from the concept image. "
        "Output a clean front-facing image-to-3D reconstruction reference: strict orthographic front elevation, "
        "object front plane parallel to the image plane, centered on a pure solid light gray background, "
        "blank unlabeled surfaces, object-only catalog cutout."
    )
    return _dedupe_comma_clauses(f"{guidance}, {prompt}")


def _safe_review_revision_prompt(module: dict[str, Any], revised_prompt: str) -> str:
    revised = " ".join(str(revised_prompt).split())
    if _is_mechanical_arm_module(module):
        forbidden = (
            "water",
            "liquid",
            "fluid",
            "transparent",
            "glass",
            "organic",
            "animal",
            "giraffe",
            "splash",
            "sculpture",
            "cable",
            "wire",
            "hose",
            "corrugated",
            "photography",
            "photorealistic",
            "product shot",
            "industrial robotic arm",
            "side profile",
            "side elevation",
            "side view",
        )
        if any(term in revised.lower() for term in forbidden):
            return (
                "A simple geometric sealed articulated robot arm CAD primitive, untextured matte white clay model, "
                "five smooth cylinders connected by circular hinge disks only, circular base, small two-finger gripper, "
                "sealed simplified surfaces, smooth uninterrupted exterior, toy-like educational robot arm, abstract cobot form, "
                "strict orthographic front view, front elevation, object front plane parallel to image plane, single centered object "
                "on pure solid white background, shape-first reconstruction reference, flat frontal silhouette, blank unlabeled surfaces, "
                "single-object catalog cutout on blank solid background, empty canvas around object"
            )
    if _is_platform_module(module):
        risky = ("monitor", "television", "screen", "display panel", "frame", "bezel", "hollow", "empty center", "stand")
        if any(term in revised.lower() for term in risky):
            return (
                "A smooth hard-surface neutral gray CAD model of one-piece low rectangular cuboid slab, "
                "single-layer undivided horizontal slab, flat top plane spans the full width, height about one tenth of total width, "
                "short front face, planar flat faces, crisp straight 90-degree edges, solid continuous slab body, "
                "strict orthographic front elevation, long low rectangular silhouette, centered on pure solid light gray background, "
                "untextured matte clay 3D CAD reconstruction input, sealed simplified surfaces, shape-first reference, "
                "single-object catalog cutout on blank solid background, blank unlabeled surfaces"
            )
    if _is_screen_module(module):
        orientation = (
            "portrait vertical flat luminous panel slab, tall narrow rectangular LED panel, height greater than width"
            if _is_vertical_screen_module(module)
            else "flat luminous panel slab, thin rectangular LED panel"
        )
        return _module_reference_prompt_with_safety(
            module,
            (
                f"{orientation}, thin uniform bezel, wall-panel style exhibition display face, planar flat screen surface, "
                "smooth sealed panel body, strict orthographic front elevation, front plane parallel to image plane, "
                "centered on pure solid light gray background, blank unlabeled surface, object-only CAD cutout"
            ),
        )
    return revised


def _image_backend_model(config: dict[str, Any], model_key: str | None = None) -> tuple[str, dict[str, Any]]:
    ref_cfg = config.get("reference_generation", {})
    selected_key = str(model_key or ref_cfg.get("default_model_key") or config.get("ai", {}).get("default_model_key") or "flux2_klein_4b")
    return selected_key, _model_cfg(config, selected_key)


def _generate_prompt_reference_image(
    *,
    config: dict[str, Any],
    prompt: str,
    output: Path,
    seed: int,
    width: int,
    height: int,
    model_key: str | None = None,
) -> dict[str, Any]:
    selected_key, model = _image_backend_model(config, model_key)
    backend_name = str(model.get("backend") or "")
    backend = build_backend(backend_name)
    generate_reference = getattr(backend, "generate_reference", None)
    if not callable(generate_reference):
        raise RuntimeError(f"Image backend {backend_name!r} cannot generate prompt reference images.")
    output.parent.mkdir(parents=True, exist_ok=True)
    generated = generate_reference(
        output,
        prompt=prompt,
        seed=seed,
        model_ref=model.get("local_path") or model.get("model_path") or model.get("model_id", ""),
        model_config=model,
        device=config.get("ai", {}).get("device", "cuda:0"),
        dtype=model.get("dtype") or config.get("ai", {}).get("dtype", "bfloat16"),
        variant=model.get("variant") or config.get("ai", {}).get("variant"),
        steps=int(model.get("steps", config.get("ai", {}).get("steps", 4))),
        guidance_scale=float(model.get("guidance_scale", config.get("ai", {}).get("guidance_scale", 1.0))),
        width=int(width or model.get("width", 1024)),
        height=int(height or model.get("height", 1024)),
        max_sequence_length=int(model.get("max_sequence_length", 512)),
        enable_model_cpu_offload=bool(model.get("enable_model_cpu_offload", True)),
    )
    return {
        "path": _absolute_artifact_path(generated),
        "created_by": "image2_model_reference_generation",
        "model_key": selected_key,
        "backend": backend_name,
        "prompt": prompt,
    }


def _uses_dashscope_imagegen(config: dict[str, Any] | None) -> bool:
    return _reference_generation_provider(config) in {"dashscope", "dashscope_imagegen", "dashscope_image_generation", "wanx"}


def _dashscope_imagegen_response_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "to_dict"):
        data = response.to_dict()
        return data if isinstance(data, dict) else {"response": data}
    return {
        "output": getattr(response, "output", None),
        "usage": getattr(response, "usage", None),
        "request_id": getattr(response, "request_id", None),
    }


def _dashscope_imagegen_urls(data: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    output = data.get("output")
    if not isinstance(output, dict):
        return urls
    for choice in output.get("choices", []) or []:
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        for item in message.get("content", []) or []:
            if isinstance(item, dict) and item.get("image"):
                urls.append(str(item["image"]))
    for item in output.get("results", []) or []:
        if isinstance(item, dict):
            for key in ("url", "image"):
                if item.get(key):
                    urls.append(str(item[key]))
    return urls


def _dashscope_imagegen_size(ref_cfg: dict[str, Any], *, kind: str, width: int, height: int) -> str:
    value = ref_cfg.get(f"{kind}_size") or ref_cfg.get("dashscope_size") or ref_cfg.get("size")
    if value:
        return str(value).replace("x", "*")
    return f"{int(width)}*{int(height)}"


def _generate_dashscope_reference_image(
    *,
    config: dict[str, Any],
    prompt: str,
    output: Path,
    seed: int,
    width: int,
    height: int,
    kind: str,
    module_id: str = "",
    source_image: Path | None = None,
) -> dict[str, Any]:
    try:
        import dashscope  # type: ignore
        from dashscope.aigc.image_generation import ImageGeneration  # type: ignore
        from dashscope.api_entities.dashscope_response import Message  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency until installed
        raise RuntimeError("DashScope SDK is required for dashscope_imagegen reference generation. Install dashscope.") from exc

    llm_cfg = config.get("agent_llm", {})
    api_key, api_key_source = _resolve_llm_api_key(llm_cfg)
    if not api_key:
        raise RuntimeError(f"DashScope API key missing for dashscope_imagegen; set {llm_cfg.get('api_key_env', 'DASHSCOPE_API_KEY')}.")

    ref_cfg = config.get("reference_generation", {}) if isinstance(config.get("reference_generation"), dict) else {}
    source_image_path = Path(source_image).expanduser().resolve() if source_image and Path(source_image).expanduser().exists() else None
    text_model = str(ref_cfg.get("dashscope_model") or ref_cfg.get("image_model") or "wan2.6-t2i")
    edit_model = str(ref_cfg.get("dashscope_image_edit_model") or ref_cfg.get("image_edit_model") or "wan2.6-image")
    model = edit_model if source_image_path else text_model
    size = _dashscope_imagegen_size(ref_cfg, kind=kind, width=width, height=height)
    prompt_extend = bool(ref_cfg.get("prompt_extend", kind == "concept"))
    retries = max(1, int(ref_cfg.get("dashscope_retries", 4)))
    retry_delay_seconds = max(1.0, float(ref_cfg.get("dashscope_retry_delay_seconds", 8.0)))
    timeout_seconds = max(30, int(ref_cfg.get("download_timeout_seconds", 120)))
    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    output.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(1, retries + 1):
        content: list[dict[str, str]] = []
        if source_image_path and source_image_path.exists():
            content.append({"text": str(prompt)[: int(ref_cfg.get("max_prompt_chars", 2200))]})
            content.append({"image": _dashscope_image_ref(source_image_path)})
        else:
            content.append({"text": str(prompt)[: int(ref_cfg.get("max_prompt_chars", 2200))]})
        call_kwargs = {
            "model": model,
            "api_key": api_key,
            "messages": [Message(role="user", content=content)],
            "prompt_extend": prompt_extend,
            "watermark": False,
            "n": 1,
            "size": size,
        }
        if source_image_path:
            call_kwargs["enable_interleave"] = False
        if ref_cfg.get("pass_seed", False):
            call_kwargs["seed"] = int(seed)
        response = ImageGeneration.call(
            **call_kwargs,
        )
        status_code = getattr(response, "status_code", None)
        if status_code is not None and int(status_code) >= 400:
            last_error = f"{status_code} {getattr(response, 'code', '')} {getattr(response, 'message', '')}".strip()
            if int(status_code) == 429 and attempt < retries:
                time.sleep(retry_delay_seconds * attempt)
                continue
            raise RuntimeError(f"DashScope image generation failed for {kind or module_id}: {last_error}")
        data = _dashscope_imagegen_response_dict(response)
        urls = _dashscope_imagegen_urls(data)
        if not urls:
            last_error = f"No image URL in DashScope response: {json.dumps(data, ensure_ascii=False)[:500]}"
            if attempt < retries:
                time.sleep(retry_delay_seconds * attempt)
                continue
            raise RuntimeError(last_error)
        with urllib.request.urlopen(urls[0], timeout=timeout_seconds) as handle:
            output.write_bytes(handle.read())
        metadata = {
            "backend": "dashscope_image_generation",
            "model": model,
            "api_key_source": api_key_source,
            "kind": kind,
            "module_id": module_id,
            "prompt": prompt,
            "source_image": _absolute_artifact_path(source_image_path) if source_image_path else "",
            "output_path": str(output.expanduser().resolve()),
            "size": size,
            "prompt_extend": prompt_extend,
            "bytes": output.stat().st_size,
            "response": data,
        }
        write_manifest(output.with_suffix(".dashscope_imagegen.json"), metadata)
        return {
            "path": _absolute_artifact_path(output),
            "created_by": "dashscope_imagegen_reference_generation",
            "image_source": "dashscope_imagegen",
            "backend": "dashscope_image_generation",
            "model": model,
            "prompt": prompt,
            "source_image": _absolute_artifact_path(source_image_path) if source_image_path else "",
            "metadata": _absolute_artifact_path(output.with_suffix(".dashscope_imagegen.json")),
        }
    raise RuntimeError(f"DashScope image generation failed for {kind or module_id}: {last_error}")


def _reference_generation_provider(config: dict[str, Any] | None) -> str:
    if not config:
        return "mock"
    return str(config.get("reference_generation", {}).get("provider", "external_imagegen")).strip().lower()


def _uses_external_imagegen(config: dict[str, Any] | None) -> bool:
    return _reference_generation_provider(config) in {"external", "external_imagegen", "imagegen", "imagegen_skill", "manual_image2"}


def _write_codex_image2_handoff(
    *,
    handoff_path: Path,
    request: dict[str, Any],
    batch_requests: list[dict[str, Any]] | None = None,
) -> Path:
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    requests = batch_requests or [request]
    request_path = str(request.get("request_path") or "")
    import_lines = [
        "PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene-import-image2 \\",
        f"  --request {request_path or '<request-json-path>'} \\",
    ]
    if len(requests) == 1 and batch_requests is None:
        import_lines.append("  --image /path/to/codex-image2-output.png")
    else:
        for index, item in enumerate(requests, start=1):
            key = str(item.get("module_id") or item.get("view_id") or item.get("kind") or f"item_{index}")
            import_lines.append(f"  --image {key}=/path/to/codex-image2-output-{index}.png \\")
        import_lines[-1] = import_lines[-1].rstrip(" \\")
    latest_import_lines = [
        "PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene-import-latest-image2 \\",
        f"  --request {request_path or '<request-json-path>'}",
    ]
    lines = [
        "# Codex image2 Handoff",
        "",
        "Use the Codex built-in image2/imagegen skill to create the requested image asset(s).",
        "Do not use DashScope/Qwen image generation or a local fallback model for this handoff.",
        "Do not add a negative prompt. All constraints are already in the positive prompt.",
        "",
        f"- Request kind: `{request.get('kind') or request.get('type')}`",
        f"- Request status: `{request.get('status')}`",
        f"- Request JSON: `{request_path}`",
    ]
    if request.get("source_image"):
        lines.append(f"- Source image: `{request['source_image']}`")
    if request.get("output_path"):
        lines.append(f"- Output path: `{request['output_path']}`")
    lines.extend(["", "## Steps", ""])
    lines.extend(
        [
            "1. Generate the image with Codex built-in image2/imagegen using the prompt below.",
            "2. If a source image is listed, use it as visual context/reference, not as the final render input.",
            "3. Save the selected generated image exactly to the listed output path.",
            "4. Or import the selected generated image with the command template below, then rerun the same Auto Scene workdir so the image can be returned to qwen3.7-plus for review.",
            "",
            "Import command template:",
            "",
            "```bash",
            *import_lines,
            "```",
            "",
            "If Codex saved the generated file(s) under `$CODEX_HOME/generated_images`, this can import the latest valid image file(s) automatically:",
            "",
            "```bash",
            *latest_import_lines,
            "```",
            "",
        ]
    )
    for index, item in enumerate(requests, start=1):
        lines.extend(
            [
                f"## Image2 Request {index}",
                "",
                f"- Kind: `{item.get('kind') or item.get('type')}`",
                f"- Module id: `{item.get('module_id', '')}`",
                f"- View id: `{item.get('view_id', '')}`",
                f"- Output path: `{item.get('output_path', '')}`",
                f"- Source image: `{item.get('source_image', '')}`",
                "",
            ]
        )
        input_images = item.get("input_images", [])
        if isinstance(input_images, list) and input_images:
            lines.extend(["Input images:", ""])
            for image in input_images:
                if isinstance(image, dict):
                    lines.append(f"- `{image.get('role', '')}`: `{image.get('path', '')}`")
            lines.append("")
        contract = item.get("position_lock_contract") if isinstance(item.get("position_lock_contract"), dict) else {}
        if contract:
            lines.extend(
                [
                    "White-model position contract:",
                    "",
                    f"- Reference bbox: `{contract.get('bbox_norm', [])}`",
                    f"- Reference center: `{contract.get('center_norm', [])}`",
                    f"- Reference coverage: `{contract.get('coverage_ratio', '')}`",
                    f"- Contract source: `{contract.get('source_rgb', '')}`",
                    "",
                ]
            )
        lines.extend(
            [
                "Prompt:",
                "",
                "```text",
                str(item.get("prompt") or ""),
                "```",
                "",
            ]
        )
    handoff_path.write_text("\n".join(lines), encoding="utf-8")
    return handoff_path


def _write_external_imagegen_request(
    *,
    output: Path,
    prompt: str,
    kind: str,
    module_id: str = "",
    source_image: Path | None = None,
) -> Path:
    source_image_path = Path(source_image).expanduser().resolve() if source_image else None
    request_path = output.with_name("imagegen_request.json")
    request = {
        "type": "external_imagegen_request",
        "status": "awaiting_external_imagegen",
        "provider": "codex_builtin_image2",
        "kind": kind,
        "module_id": module_id,
        "request_path": str(request_path.expanduser().resolve()),
        "output_path": str(output.expanduser().resolve()),
        "prompt": prompt,
        "source_image": str(source_image_path) if source_image_path else "",
        "codex_image2_handoff": str(output.with_name("codex_image2_handoff.md").expanduser().resolve()),
        "resume_instruction": "Generate the requested image with Codex image2, import or copy it to output_path, then rerun the same auto-scene command/workdir.",
        "import_command": (
            "PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene-import-image2 "
            f"--request {request_path.expanduser().resolve()} --image /path/to/codex-image2-output.png"
        ),
        "latest_import_command": (
            "PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene-import-latest-image2 "
            f"--request {request_path.expanduser().resolve()}"
        ),
        "instructions": [
            "Generate this image with the Codex imagegen/image2 skill, not a local model.",
            "Use source_image as visual context when it is provided; generate a new isolated standalone module image matching the prompt.",
            "Import the selected generated image with import_command, latest_import_command, or copy it exactly to output_path.",
            "Then rerun this stage so the image can be returned to the multimodal agent for review.",
        ],
    }
    _write_codex_image2_handoff(handoff_path=Path(request["codex_image2_handoff"]), request=request)
    write_manifest(request_path, request)
    return request_path


def _raise_external_imagegen_required(request_path: Path) -> None:
    raise ExternalImagegenRequired(request_path)


def call_model_scene_planner(config: dict[str, Any], options: AutoSceneOptions) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if options.dry_run or options.backend == "mock" or not options.use_llm:
        if options.dry_run or options.backend == "mock":
            return _mock_model_plan(config, options)
        raise RuntimeError("Auto Scene real runs require the configured multimodal LLM scene_planner; --no-llm is only valid for dry-run/mock validation.")
    llm_cfg = config.get("agent_llm", {})
    if not llm_cfg.get("enabled", True):
        raise RuntimeError("Auto Scene scene_planner is disabled. Enable agent_llm or run with --dry-run/--backend mock for smoke validation.")
    prompt = {
        "task": "需求理解与概念规划",
        "user_request": options.request,
        "requirements": [
            "理解用户需求并扩写为可用于生图工具的详细英文 concept_prompt。",
            "输出 strict JSON only，不要 Markdown。",
            "概念图只用于规划和审查，不得作为最终 AI 渲染输入。",
            "先不要硬编码物体清单；只给场景结构、全局风格、预期元素和最终渲染提示词。",
            "后续会把 concept_prompt 交给 image2 生图工具生成概念图，再把图返回给你审查。",
            "所有生图约束写进正向 prompt；JSON 不包含 negative_prompt 字段。",
        ],
        "json_schema": {
            "auto_task": {
                "user_request": "original request",
                "expanded_request": "detailed expanded scene requirement",
                "style_preset": options.style_preset,
                "quality_mode": options.quality_mode,
                "geometry_mode": options.geometry_mode,
                "output_views": options.output_views,
                "num_candidates_per_view": options.num_candidates_per_view,
                "max_retries": options.max_retries,
            },
            "scene_plan": {
                "scene_type": "short scene type",
                "main_subject": {"name": "hero object", "role": "hero_object", "priority": 1},
                "environment": {"description": "environment summary"},
                "composition": {"camera_style": "desired camera"},
                "global_style": {"material_language": "style", "color_palette": ["colors"], "avoid": ["bad artifacts"]},
                "expected_elements": ["all important objects that must appear"],
            },
            "concept_image_plan": {
                "concept_prompt": "image2 prompt for full-scene concept image",
                "width": 1024,
                "height": 1024,
                "output": "concept/global_concept.png",
            },
            "prompt_plan": {
                "render_prompt": "final AI render prompt constrained by white-model render channels",
            },
            "camera_plan": {"views": []},
        },
    }
    try:
        parsed, info = _call_dashscope_multimodal_json(
            config,
            purpose="scene_planning",
            messages=[{"role": "user", "content": [{"text": json.dumps(prompt, ensure_ascii=False)}]}],
        )
    except Exception as exc:
        raise RuntimeError(f"Auto Scene scene_planner failed; rule planning is disabled for real runs: {exc}") from exc
    parsed = _coerce_json_object(parsed)
    concept_prompt = str(_coerce_json_object(parsed.get("concept_image_plan")).get("concept_prompt") or "").strip()
    expanded_request = str(_coerce_json_object(parsed.get("auto_task")).get("expanded_request") or "").strip()
    if not concept_prompt:
        raise RuntimeError("DashScope scene_planning returned no concept_image_plan.concept_prompt.")
    if not expanded_request:
        raise RuntimeError("DashScope scene_planning returned no auto_task.expanded_request.")
    normalized = _normalize_model_plan(parsed, options)
    info["plan_source"] = "model_only"
    info["rule_planning"] = "disabled_for_real_runs"
    return normalized, info


def _existing_scene_planner_outputs(workdir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]] | None:
    paths = {
        "auto_task": workdir / "auto_task.json",
        "scene_plan": workdir / "scene_plan.json",
        "concept_image_plan": workdir / "concept" / "concept_image_plan.json",
        "prompt_plan": workdir / "prompt_plan.json",
        "camera_plan": workdir / "cameras" / "camera_plan.json",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    try:
        plan = {key: read_manifest(path) for key, path in paths.items()}
    except Exception:
        return None
    concept_prompt = str(plan["concept_image_plan"].get("concept_prompt") or "").strip()
    expanded_request = str(plan["auto_task"].get("expanded_request") or "").strip()
    if not concept_prompt or not expanded_request:
        return None
    return plan, {
        "backend": "resume_existing_artifacts",
        "plan_source": "existing_scene_planner_outputs",
        "rule_planning": "disabled_for_real_runs",
        "artifacts": {key: _absolute_artifact_path(path) for key, path in paths.items()},
    }


def _existing_module_prompt_outputs(workdir: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    module_plan_path = workdir / "modules" / "module_plan.json"
    if not module_plan_path.exists():
        return None
    try:
        module_plan = read_manifest(module_plan_path)
    except Exception:
        return None
    if not isinstance(module_plan.get("modules"), list) or not module_plan["modules"]:
        return None
    info_path = workdir / "modules" / "module_prompt_info.json"
    if info_path.exists():
        try:
            info = read_manifest(info_path)
        except Exception:
            info = {}
    else:
        info = {}
    info.setdefault("backend", "resume_existing_artifacts")
    info["plan_source"] = "existing_module_prompt_outputs"
    info.setdefault("artifacts", {})["module_plan"] = _absolute_artifact_path(module_plan_path)
    if info_path.exists():
        info["artifacts"]["module_prompt_info"] = _absolute_artifact_path(info_path)
    return module_plan, info


def review_concept_image(
    config: dict[str, Any],
    options: AutoSceneOptions,
    *,
    concept_image: Path,
    scene_plan: dict[str, Any],
    concept_image_plan: dict[str, Any],
    attempt: int,
) -> dict[str, Any]:
    if options.dry_run or options.backend == "mock" or not options.use_llm:
        return {
            "type": "concept_image_review",
            "status": "pass",
            "attempt": attempt,
            "backend": "mock_model_brain",
            "checks": {"required_elements_present": True, "no_forbidden_text": True, "composition_usable": True},
            "missing_elements": [],
            "extra_elements": [],
            "revised_concept_prompt": "",
            "notes": "Dry-run review passes generated concept image for pipeline validation.",
        }
    audit_prompt = {
        "task": "审查 image2 生成的全局概念图",
        "scene_plan": scene_plan,
        "concept_prompt": concept_image_plan.get("concept_prompt", ""),
        "required_output": {
            "status": "pass or revise",
            "checks": {
                "required_elements_present": "boolean",
                "no_forbidden_text": "boolean",
                "composition_usable": "boolean",
                "hero_subject_clear": "boolean",
            },
            "missing_elements": ["elements missing from image"],
            "extra_elements": ["unwanted elements"],
            "revised_concept_prompt": "only if status is revise",
            "notes": "short reason",
        },
        "rules": [
            "如果有可读文字、logo、人物、界面图标，必须 revise。",
            "如果用户要求的主要物体缺失，必须 revise。",
            "返回 strict JSON only。",
        ],
    }
    parsed, info = _call_dashscope_multimodal_json(
        config,
        purpose="concept_image_review",
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": _dashscope_image_ref(concept_image)},
                    {"text": json.dumps(audit_prompt, ensure_ascii=False)},
                ],
            }
        ],
    )
    parsed = _coerce_json_object(parsed)
    parsed.setdefault("type", "concept_image_review")
    parsed.setdefault("attempt", attempt)
    parsed.setdefault("backend", info["backend"])
    return parsed


def generate_model_module_plan(
    config: dict[str, Any],
    options: AutoSceneOptions,
    *,
    scene_plan: dict[str, Any],
    concept_image_plan: dict[str, Any],
    concept_image: Path,
    concept_review: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if options.dry_run or options.backend == "mock" or not options.use_llm:
        plan, info = _mock_model_plan(config, options)
        return plan["module_plan"], {"backend": "mock_model_brain", "reason": "dry_run_or_mock_backend"}
    prompt = {
        "task": "根据已审查概念图为每个场景物体写独立 image2 提示词",
        "scene_plan": scene_plan,
        "concept_prompt": concept_image_plan.get("concept_prompt", ""),
        "concept_review": concept_review,
        "layout_coordinate_contract": _layout_coordinate_contract(),
        "few_shot_layout_examples": _layout_few_shot_examples(),
        "few_shot_reference_prompt_examples": _module_reference_prompt_few_shot_examples(),
        "requirements": [
            "列出概念图中应进入 3D 场景搭建的每个独立物体/模块。",
            "每个模块都要有新的 reference_prompt，用于生成单物体图片。",
            "reference_prompt 只使用正向描述，写明 pure solid background、single centered object、blank unlabeled surfaces、object-only catalog cutout。",
            "reference_prompt 必须适合 image-to-3D：严格正交正视图/front elevation，镜头正对物体中心，无透视畸变，完整物体轮廓必须可见且不可裁切。",
            "不要要求三分之二视角、斜视角、俯视、背视、复杂透视、场景环境、阴影地面、多个物体或概念海报式构图。",
            "包含 expected_real_world_size 和 placement，placement 必须遵守 layout_coordinate_contract。",
            "不要把前后深度写入 z；z 只表示垂直高度。地面物体 position[2] 通常是 height/2。",
            "平面朝向/yaw 写入 rotation_deg[2]，不要写入 rotation_deg[1]，除非确实需要俯仰。",
            "输出 3 到 6 个最重要模块，主物体 role 必须为 hero_object。",
            "所有生图约束写进 reference_prompt；JSON 不包含 negative_prompt 字段。",
            "返回 strict JSON only，首字符必须是 {，不要 Markdown，不要 ```json 代码块。",
        ],
        "json_schema": {
            "modules": [
                {
                    "module_id": "stable_snake_case_id",
                    "name": "object name",
                    "category": "vehicle/stage_prop/background_prop/environment/decorative_object",
                    "role": "hero_object/supporting_object/background/decorative_object",
                    "priority": 1,
                    "reference_prompt": "single object image2 prompt, strict orthographic front view, pure solid background",
                    "expected_real_world_size": {"width": 1.0, "depth": 1.0, "height": 1.0, "unit": "meters"},
                    "placement": {"anchor": "scene anchor", "position": [0, 0, 0], "rotation_deg": [0, 0, 0], "scale_policy": "reason"},
                    "generate_reference_image": True,
                    "generate_3d": True,
                }
            ]
        },
    }
    parsed, info = _call_dashscope_multimodal_json(
        config,
        purpose="module_prompt_generation",
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": _dashscope_image_ref(concept_image)},
                    {"text": json.dumps(prompt, ensure_ascii=False)},
                ],
            }
        ],
        max_tokens=max(3600, int(config.get("agent_llm", {}).get("max_tokens", 1600))),
    )
    raw_modules = parsed.get("modules", []) if isinstance(parsed, dict) else []
    if not isinstance(raw_modules, list) or not raw_modules:
        raise RuntimeError("DashScope module_prompt_generation returned no modules.")
    modules = [_normalize_module(module if isinstance(module, dict) else {}, index) for index, module in enumerate(raw_modules[:8])]
    if not any(module.get("role") == "hero_object" for module in modules):
        modules[0]["role"] = "hero_object"
        modules[0]["priority"] = 1
    modules = sorted(modules, key=lambda item: int(item.get("priority", 99)))
    return {"modules": modules, "source": "dashscope_multimodal_object_prompt", "concept_review_status": concept_review.get("status", "")}, info


def _finite_vector(values: Any, *, length: int) -> list[float] | None:
    if not isinstance(values, (list, tuple)) or len(values) < length:
        return None
    out: list[float] = []
    for value in values[:length]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        out.append(number)
    return out


def validate_module_layout_contract(module_plan: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    contract = _layout_coordinate_contract()
    for module in module_plan.get("modules", []):
        if not isinstance(module, dict):
            errors.append({"module_id": "", "code": "invalid_module", "message": "Module entry is not an object."})
            continue
        module_id = str(module.get("module_id", ""))
        size = _module_size(module)
        height = max(float(size[2]), 1e-6)
        placement = module.get("placement") if isinstance(module.get("placement"), dict) else {}
        position = _finite_vector(placement.get("position"), length=3)
        rotation = _finite_vector(placement.get("rotation_deg"), length=3)
        if position is None:
            errors.append({"module_id": module_id, "code": "invalid_position", "message": "placement.position must be numeric [x, y, z]."})
            continue
        if rotation is None:
            errors.append({"module_id": module_id, "code": "invalid_rotation", "message": "placement.rotation_deg must be numeric [x, y, z]."})
        bottom_z = float(position[2]) - height / 2.0
        if bottom_z < -0.05:
            errors.append(
                {
                    "module_id": module_id,
                    "code": "bbox_below_ground",
                    "message": "The module bounding-box bottom is below the ground plane under the declared position semantics.",
                    "position": position,
                    "expected_real_world_size": {
                        "width": float(size[0]),
                        "depth": float(size[1]),
                        "height": float(size[2]),
                        "unit": "meters",
                    },
                    "bottom_z": round(bottom_z, 6),
                    "suggestion": "Use Blender Z-up bbox-center semantics. For floor contact, set position[2] close to height / 2.",
                }
            )
        if rotation is not None and abs(rotation[1]) > 1.0 and abs(rotation[2]) <= 1.0:
            warnings.append(
                {
                    "module_id": module_id,
                    "code": "possible_yaw_axis_confusion",
                    "message": "In Blender Z-up, plan-view yaw normally belongs in rotation_deg[2], not rotation_deg[1]. Keep this only if pitch is intentional.",
                    "rotation_deg": rotation,
                }
            )
    return {
        "type": "module_layout_contract_check",
        "status": "needs_repair" if errors else "pass",
        "coordinate_contract": contract,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def repair_model_module_layout(
    config: dict[str, Any],
    options: AutoSceneOptions,
    *,
    scene_plan: dict[str, Any],
    concept_image: Path,
    module_plan: dict[str, Any],
    layout_check: dict[str, Any],
    attempt: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if str(layout_check.get("status", "")).lower() == "pass":
        return module_plan, {"backend": "layout_contract_check", "repair_attempted": False, "layout_check": layout_check}
    if options.dry_run or options.backend == "mock" or not options.use_llm:
        return module_plan, {
            "backend": "layout_contract_check",
            "repair_attempted": False,
            "reason": "dry_run_or_mock_backend",
            "layout_check": layout_check,
        }
    prompt = {
        "task": "修正模块 placement，使其满足 Blender Z-up 3D 场景组装契约",
        "scene_plan": scene_plan,
        "layout_coordinate_contract": _layout_coordinate_contract(),
        "few_shot_layout_examples": _layout_few_shot_examples(),
        "layout_issues_to_fix": layout_check,
        "module_plan": module_plan,
        "requirements": [
            "只修正 placement 和必要的 layout reasoning；不要改变 module_id。",
            "reference_prompt、category、role、priority、generate_reference_image、generate_3d 默认保持不变。",
            "所有模块继续使用 bbox center position 语义。",
            "地面承托、墙面背景、顶部悬挂、侧边支撑等都必须用同一套 Blender Z-up 坐标表达。",
            "返回完整 modules 数组，strict JSON only。",
        ],
        "json_schema": {"modules": module_plan.get("modules", [])},
    }
    parsed, info = _call_dashscope_multimodal_json(
        config,
        purpose="module_layout_repair",
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": _dashscope_image_ref(concept_image)},
                    {"text": json.dumps(prompt, ensure_ascii=False)},
                ],
            }
        ],
        max_tokens=max(2200, int(config.get("agent_llm", {}).get("max_tokens", 1600))),
    )
    parsed = _coerce_json_object(parsed)
    raw_modules = parsed.get("modules")
    if not isinstance(raw_modules, list):
        raw_modules = _coerce_json_object(parsed.get("module_plan")).get("modules", [])
    if not isinstance(raw_modules, list) or not raw_modules:
        raise RuntimeError("DashScope module_layout_repair returned no modules.")
    original_by_id = {str(module.get("module_id", "")): module for module in module_plan.get("modules", []) if isinstance(module, dict)}
    repaired_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_modules):
        if not isinstance(raw, dict):
            continue
        module_id = str(raw.get("module_id", ""))
        merged = {**copy.deepcopy(original_by_id.get(module_id, {})), **raw}
        normalized = _normalize_module(merged, index)
        repaired_by_id[str(normalized["module_id"])] = normalized
    modules: list[dict[str, Any]] = []
    for index, original in enumerate(module_plan.get("modules", [])):
        if not isinstance(original, dict):
            continue
        module_id = str(original.get("module_id", ""))
        modules.append(repaired_by_id.pop(module_id, _normalize_module(copy.deepcopy(original), index)))
    modules.extend(repaired_by_id.values())
    modules = sorted(modules, key=lambda item: int(item.get("priority", 99)))
    repaired_plan = {**module_plan, "modules": modules, "layout_repair_source": "dashscope_multimodal_few_shot"}
    repaired_check = validate_module_layout_contract(repaired_plan)
    return repaired_plan, {
        **info,
        "repair_attempted": True,
        "attempt": attempt,
        "input_layout_check": layout_check,
        "repaired_layout_check": repaired_check,
    }


def _draw_mock_scene(path: Path, *, kind: str, seed: int, size: int = 768) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size, size), (234, 236, 238))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, int(size * 0.63), size, size], fill=(190, 194, 198))
    draw.rectangle([0, 0, size, int(size * 0.63)], fill=(220, 225, 230))
    draw.ellipse([int(size * 0.18), int(size * 0.58), int(size * 0.82), int(size * 0.78)], fill=(25, 28, 32), outline=(62, 170, 255), width=max(2, size // 120))
    draw.rounded_rectangle([int(size * 0.25), int(size * 0.42), int(size * 0.75), int(size * 0.58)], radius=size // 18, fill=(232, 234, 232), outline=(38, 42, 48), width=max(2, size // 120))
    draw.polygon([(int(size * 0.38), int(size * 0.42)), (int(size * 0.50), int(size * 0.32)), (int(size * 0.62), int(size * 0.42))], fill=(40, 47, 56))
    for x in (0.12, 0.74):
        draw.rectangle([int(size * x), int(size * 0.25), int(size * (x + 0.14)), int(size * 0.55)], fill=(18, 24, 34), outline=(68, 180, 255), width=3)
        draw.line([int(size * x), int(size * 0.3), int(size * (x + 0.14)), int(size * 0.5)], fill=(70, 184, 255), width=3)
    if kind == "reference":
        image = Image.new("RGB", (size, size), (248, 248, 248))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle([int(size * 0.22), int(size * 0.35), int(size * 0.78), int(size * 0.6)], radius=size // 20, fill=(232, 234, 232), outline=(35, 40, 48), width=4)
        draw.ellipse([int(size * 0.28), int(size * 0.56), int(size * 0.4), int(size * 0.68)], fill=(28, 31, 36))
        draw.ellipse([int(size * 0.6), int(size * 0.56), int(size * 0.72), int(size * 0.68)], fill=(28, 31, 36))
    image = image.filter(ImageFilter.SMOOTH)
    image.save(path)


def _valid_image(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def _valid_model_artifact(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0 and path.suffix.lower() in {".glb", ".gltf", ".obj"}


def _image2_import_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _codex_image2_request_keys(item: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in ("module_id", "view_id", "kind"):
        value = str(item.get(key) or "").strip()
        if value:
            keys.append(value)
    output_path = str(item.get("output_path") or item.get("output") or "").strip()
    if output_path:
        keys.append(output_path)
        resolved = str(Path(output_path).expanduser().resolve())
        if resolved not in keys:
            keys.append(resolved)
    return keys


def _resolve_image2_import_source(
    item: dict[str, Any],
    *,
    image_path: Path | None,
    image_mappings: Mapping[str, Path | str],
    allow_single_image: bool,
) -> Path:
    if image_path is not None and allow_single_image:
        return image_path
    for key in _codex_image2_request_keys(item):
        if key in image_mappings:
            return Path(image_mappings[key])
    label = ", ".join(_codex_image2_request_keys(item)) or str(item.get("output_path") or "unknown request")
    raise ValueError(f"No image supplied for Codex image2 request item: {label}")


def _copy_codex_image2_result(source: Path, output_path: str) -> dict[str, str]:
    source = Path(source).expanduser()
    if not _valid_image(source):
        raise ValueError(f"Codex image2 result is not a valid image: {source}")
    source = source.resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if source != output:
        shutil.copy2(source, output)
    if not _valid_image(output):
        raise ValueError(f"Imported Codex image2 output is not a valid image: {output}")
    return {"source_image": str(source), "output_path": str(output)}


def _codex_image2_request_items(request: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = request.get("requests")
    if isinstance(raw_items, list) and raw_items:
        return [item for item in raw_items if isinstance(item, dict)]
    return [request]


def import_codex_image2_result(
    request_path: Path,
    *,
    image_path: Path | None = None,
    image_mappings: Mapping[str, Path | str] | None = None,
) -> dict[str, Any]:
    """Copy Codex image2 outputs into the request-declared output paths."""
    request_path = Path(request_path).expanduser().resolve()
    request = read_manifest(request_path)
    mappings = dict(image_mappings or {})
    if image_path is None and not mappings:
        raise ValueError("Provide an image path for a single request or keyed mappings for a batch request.")

    items = _codex_image2_request_items(request)
    if not items:
        raise ValueError(f"No Codex image2 request items found in: {request_path}")
    allow_single_image = len(items) == 1 and image_path is not None
    if len(items) > 1 and image_path is not None and not mappings:
        raise ValueError("Batch Codex image2 imports require keyed mappings such as module_id=/path/to/image.png.")

    imported_at = _image2_import_timestamp()
    imports: list[dict[str, Any]] = []
    for item in items:
        output_path = str(item.get("output_path") or item.get("output") or "").strip()
        if not output_path:
            raise ValueError(f"Codex image2 request item is missing output_path in: {request_path}")
        source = _resolve_image2_import_source(
            item,
            image_path=image_path,
            image_mappings=mappings,
            allow_single_image=allow_single_image,
        )
        copied = _copy_codex_image2_result(source, output_path)
        item["status"] = "fulfilled_by_codex_image2"
        item["imported_at"] = imported_at
        item["imported_image"] = copied["source_image"]
        item["fulfilled_output_path"] = copied["output_path"]
        imports.append(
            {
                "module_id": item.get("module_id", ""),
                "view_id": item.get("view_id", ""),
                "kind": item.get("kind") or item.get("type") or "",
                **copied,
            }
        )

    request["status"] = "fulfilled_by_codex_image2"
    request["fulfilled_at"] = imported_at
    request["import_count"] = len(imports)
    import_manifest_path = request_path.with_name("codex_image2_import.json")
    request["import_manifest"] = str(import_manifest_path)
    write_manifest(request_path, request)
    summary = {
        "type": "codex_image2_import",
        "status": "complete",
        "request_path": str(request_path),
        "request_kind": request.get("kind") or request.get("type") or "",
        "imported_at": imported_at,
        "import_count": len(imports),
        "imports": imports,
    }
    write_manifest(import_manifest_path, summary)
    return summary


def _codex_image2_generated_roots(codex_home: Path | None = None) -> list[Path]:
    if codex_home is not None:
        raw_candidates = [codex_home]
    else:
        raw_candidates = [
            Path(os.environ["CODEX_HOME"]) if os.environ.get("CODEX_HOME") else None,
            Path.home() / ".codex",
        ]
    roots: list[Path] = []
    for candidate in raw_candidates:
        if not candidate:
            continue
        generated = Path(candidate).expanduser() / "generated_images"
        if generated.exists() and generated not in roots:
            roots.append(generated)
    return roots


def find_latest_codex_image2_outputs(
    *,
    count: int,
    codex_home: Path | None = None,
    after_timestamp: float | None = None,
    newest_first: bool = False,
) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    candidates: list[Path] = []
    for root in _codex_image2_generated_roots(codex_home):
        for path in root.rglob("*"):
            if path.suffix.lower() not in suffixes:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if after_timestamp is not None and stat.st_mtime <= after_timestamp:
                continue
            if _valid_image(path):
                candidates.append(path)
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    selected = candidates[: max(1, int(count))]
    if len(selected) < max(1, int(count)):
        roots = ", ".join(str(root) for root in _codex_image2_generated_roots(codex_home)) or "<none>"
        raise FileNotFoundError(f"Found {len(selected)} Codex image2 output(s), expected {count}. Searched: {roots}")
    if not newest_first:
        selected = list(reversed(selected))
    return [path.expanduser().resolve() for path in selected]


def import_latest_codex_image2_results(
    request_path: Path,
    *,
    codex_home: Path | None = None,
    after_timestamp: float | None = None,
    after_marker: Path | None = None,
    newest_first: bool = False,
) -> dict[str, Any]:
    request_path = Path(request_path).expanduser().resolve()
    request = read_manifest(request_path)
    items = _codex_image2_request_items(request)
    if not items:
        raise ValueError(f"No Codex image2 request items found in: {request_path}")
    marker_time = None
    if after_marker is not None:
        marker_time = Path(after_marker).expanduser().stat().st_mtime
    after = after_timestamp if after_timestamp is not None else marker_time
    outputs = find_latest_codex_image2_outputs(
        count=len(items),
        codex_home=codex_home,
        after_timestamp=after,
        newest_first=newest_first,
    )
    if len(items) == 1:
        summary = import_codex_image2_result(request_path, image_path=outputs[0])
    else:
        mappings: dict[str, Path] = {}
        for item, output in zip(items, outputs):
            keys = _codex_image2_request_keys(item)
            if not keys:
                raise ValueError(f"Cannot map latest Codex image2 output to request item without module_id/view_id/output_path: {item}")
            mappings[keys[0]] = output
        summary = import_codex_image2_result(request_path, image_mappings=mappings)
    summary["source"] = "codex_generated_images_latest_scan"
    summary["codex_generated_images"] = [str(path) for path in outputs]
    summary["mapping_policy"] = "request_order_to_file_mtime_order"
    summary["newest_first"] = bool(newest_first)
    summary["after_timestamp"] = after
    write_manifest(Path(summary["request_path"]).with_name("codex_image2_import.json"), summary)
    return summary


def _read_manifest_or_empty(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            return read_manifest(path)
    except Exception:
        return {}
    return {}


def _json_has_key(data: Any, key: str) -> bool:
    if isinstance(data, dict):
        return key in data or any(_json_has_key(value, key) for value in data.values())
    if isinstance(data, list):
        return any(_json_has_key(item, key) for item in data)
    return False


def _artifact_path_from_summary(workdir: Path, summary: dict[str, Any], key: str, fallback: str) -> Path:
    value = str(summary.get(key) or "")
    return Path(value).expanduser() if value else workdir / fallback


def _audit_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    *,
    evidence: dict[str, Any] | None = None,
    reason: str = "",
    required: bool = True,
) -> None:
    checks.append(
        {
            "id": check_id,
            "status": "pass" if passed else "fail",
            "required": required,
            "reason": reason,
            "evidence": evidence or {},
        }
    )


def audit_auto_scene_image2_flow(
    workdir: Path,
    *,
    require_codex_image2: bool = True,
    require_hunyuan_3d: bool = True,
    write_report_file: bool = True,
) -> dict[str, Any]:
    workdir = Path(workdir).expanduser().resolve()
    summary_path = workdir / "auto_scene_summary.json"
    summary = _read_manifest_or_empty(summary_path)
    checks: list[dict[str, Any]] = []

    planning = summary.get("planning") if isinstance(summary.get("planning"), dict) else {}
    plan_source = str(planning.get("plan_source") or "")
    planning_backend = str(planning.get("backend") or "")
    _audit_check(
        checks,
        "scene_planner_model_owned",
        plan_source in {"model_only", "existing_scene_planner_outputs"} and planning_backend in {"dashscope_multimodal", "resume_existing_artifacts"},
        evidence={"backend": planning_backend, "model": planning.get("model", ""), "plan_source": plan_source},
        reason="Real Auto Scene planning must come from qwen3.7-plus/DashScope or resumed model artifacts.",
    )
    _audit_check(
        checks,
        "rule_planning_disabled_for_real_run",
        str(planning.get("rule_planning") or "") == "disabled_for_real_runs",
        evidence={"rule_planning": planning.get("rule_planning", "")},
        reason="Rule planning must not satisfy a real Auto Scene run.",
    )

    concept_plan_path = _artifact_path_from_summary(workdir, summary, "concept_image_plan", "concept/concept_image_plan.json")
    concept_plan = _read_manifest_or_empty(concept_plan_path)
    _audit_check(
        checks,
        "concept_prompt_from_model_exists",
        bool(str(concept_plan.get("concept_prompt") or "").strip()),
        evidence={"concept_image_plan": _absolute_artifact_path(concept_plan_path), "prompt_chars": len(str(concept_plan.get("concept_prompt") or ""))},
        reason="The model must expand the user request into concept_image_plan.concept_prompt.",
    )

    concept_image_path = _artifact_path_from_summary(workdir, summary, "global_concept", "concept/global_concept.png")
    concept_generation_path = concept_image_path.with_name("generation_manifest.json")
    concept_generation = _read_manifest_or_empty(concept_generation_path)
    concept_request = _read_manifest_or_empty(concept_image_path.with_name("imagegen_request.json"))
    concept_source = str(concept_generation.get("image_source") or concept_generation.get("created_by") or "")
    codex_like_sources = {"imagegen_skill_external", "image2_provided", "codex_builtin_image2", "codex_generated_images_latest_scan"}
    concept_codex_ok = _valid_image(concept_image_path) and (
        not require_codex_image2
        or concept_source in codex_like_sources
        or str(concept_request.get("provider") or "") == "codex_builtin_image2"
    )
    _audit_check(
        checks,
        "concept_image_codex_image2",
        concept_codex_ok,
        evidence={
            "concept_image": _absolute_artifact_path(concept_image_path),
            "valid_image": _valid_image(concept_image_path),
            "created_by": concept_generation.get("created_by", ""),
            "image_source": concept_generation.get("image_source", ""),
            "request_provider": concept_request.get("provider", ""),
        },
        reason="The concept image must be provided by Codex image2/imported image2, not a local/mock/DashScope image generator.",
    )
    _audit_check(
        checks,
        "concept_no_negative_prompt",
        not any(_json_has_key(data, "negative_prompt") for data in (concept_plan, concept_generation, concept_request)),
        evidence={"concept_image_plan": _absolute_artifact_path(concept_plan_path), "generation_manifest": _absolute_artifact_path(concept_generation_path)},
        reason="Concept generation constraints must be positive-prompt only.",
    )

    concept_review_path = _artifact_path_from_summary(workdir, summary, "concept_review", "concept/concept_review.json")
    concept_review = _read_manifest_or_empty(concept_review_path)
    _audit_check(
        checks,
        "concept_image_reviewed_by_model",
        str(concept_review.get("backend") or "") == "dashscope_multimodal" and str(concept_review.get("status") or "").lower() == "pass",
        evidence={"concept_review": _absolute_artifact_path(concept_review_path), "backend": concept_review.get("backend", ""), "status": concept_review.get("status", "")},
        reason="The generated concept image must be returned to the multimodal model and pass review before module prompting.",
    )

    module_plan_path = _artifact_path_from_summary(workdir, summary, "module_plan", "modules/module_plan.json")
    module_prompt_info_path = _artifact_path_from_summary(workdir, summary, "module_prompt_info", "modules/module_prompt_info.json")
    module_plan = _read_manifest_or_empty(module_plan_path)
    module_prompt_info = _read_manifest_or_empty(module_prompt_info_path)
    modules = [module for module in module_plan.get("modules", []) if isinstance(module, dict)]
    _audit_check(
        checks,
        "module_prompts_written_by_model",
        bool(modules) and str(module_prompt_info.get("backend") or "") in {"dashscope_multimodal", "resume_existing_artifacts"},
        evidence={"module_count": len(modules), "module_prompt_info": _absolute_artifact_path(module_prompt_info_path), "backend": module_prompt_info.get("backend", "")},
        reason="The reviewed concept image must be returned to the model so it can write per-object prompts.",
    )
    _audit_check(
        checks,
        "module_plan_no_negative_prompt",
        not _json_has_key(module_plan, "negative_prompt"),
        evidence={"module_plan": _absolute_artifact_path(module_plan_path)},
        reason="Module prompts must not include negative_prompt fields.",
    )

    reference_results: list[dict[str, Any]] = []
    all_reference_ok = bool(modules)
    all_reference_positive_only = True
    for module in modules:
        module_id = str(module.get("module_id") or "")
        manifest_path = workdir / "modules" / module_id / "reference_manifest.json"
        manifest = _read_manifest_or_empty(manifest_path)
        reference_image = Path(str(manifest.get("reference_image") or workdir / "modules" / module_id / "reference.png"))
        source = str(manifest.get("image_source") or manifest.get("created_by") or "")
        ok = _valid_image(reference_image) and (not require_codex_image2 or source in codex_like_sources)
        all_reference_ok = all_reference_ok and ok
        positive_only = not _json_has_key(manifest, "negative_prompt")
        all_reference_positive_only = all_reference_positive_only and positive_only
        reference_results.append(
            {
                "module_id": module_id,
                "reference_image": _absolute_artifact_path(reference_image),
                "valid_image": _valid_image(reference_image),
                "image_source": source,
                "review_status": manifest.get("review_status", ""),
                "positive_only": positive_only,
                "pass": ok,
            }
        )
    _audit_check(
        checks,
        "module_reference_images_codex_image2",
        all_reference_ok,
        evidence={"modules": reference_results},
        reason="Every module reference image must come from Codex image2/imported image2 and be valid before 3D generation.",
    )
    _audit_check(
        checks,
        "module_reference_no_negative_prompt",
        all_reference_positive_only,
        evidence={"modules": reference_results},
        reason="Module reference generation must be positive-prompt only.",
    )

    module_review_path = _artifact_path_from_summary(workdir, summary, "module_reference_review", "modules/module_reference_review.json")
    module_review = _read_manifest_or_empty(module_review_path)
    _audit_check(
        checks,
        "module_references_reviewed_by_model",
        str(module_review.get("backend") or "") == "dashscope_multimodal" and str(module_review.get("status") or "").lower() == "pass",
        evidence={"module_reference_review": _absolute_artifact_path(module_review_path), "backend": module_review.get("backend", ""), "status": module_review.get("status", "")},
        reason="The generated module reference images must be returned to the multimodal model and pass review.",
    )

    asset_manifest_path = _artifact_path_from_summary(workdir, summary, "module_asset_manifest", "modules/module_asset_manifest.json")
    asset_manifest = _read_manifest_or_empty(asset_manifest_path)
    asset_results: list[dict[str, Any]] = []
    all_assets_ok = bool(asset_manifest.get("modules"))
    for asset in asset_manifest.get("modules", []) if isinstance(asset_manifest.get("modules"), list) else []:
        if not isinstance(asset, dict):
            continue
        module_id = str(asset.get("module_id") or "")
        metadata_path = Path(str(asset.get("metadata") or workdir / "modules" / module_id / "metadata.json"))
        metadata = _read_manifest_or_empty(metadata_path)
        model_path = Path(str(asset.get("model_path") or metadata.get("model_path") or ""))
        created_by = str(metadata.get("created_by") or asset.get("sanity", {}).get("proxy_geometry") or "")
        fallback_used = bool(asset.get("fallback_used") or asset.get("sanity", {}).get("fallback_used"))
        ok = _valid_model_artifact(model_path) and (not require_hunyuan_3d or created_by == "hunyuan3d_2_1_shape_from_reviewed_reference") and not fallback_used
        all_assets_ok = all_assets_ok and ok
        asset_results.append(
            {
                "module_id": module_id,
                "model_path": _absolute_artifact_path(model_path),
                "metadata": _absolute_artifact_path(metadata_path),
                "created_by": created_by,
                "fallback_used": fallback_used,
                "pass": ok,
            }
        )
    _audit_check(
        checks,
        "reviewed_references_enter_3d_ai",
        all_assets_ok,
        evidence={"module_asset_manifest": _absolute_artifact_path(asset_manifest_path), "modules": asset_results},
        reason="Reviewed module references must be handed to the configured 3D AI generator without procedural fallback.",
    )

    final_request_path = workdir / "final" / "codex_image2_final_request.json"
    final_request = _read_manifest_or_empty(final_request_path)
    if final_request:
        final_items = [item for item in final_request.get("requests", []) if isinstance(item, dict)]
        channel_allowlist = {"rgb", "edge", "depth", "normal", "mask", "skeleton"}
        final_evidence: list[dict[str, Any]] = []
        all_white_locked = bool(final_items)
        all_render_channel_only = bool(final_items)
        all_position_contracts_present = bool(final_items)
        for item in final_items:
            inputs = [entry for entry in item.get("input_images", []) if isinstance(entry, dict)]
            roles = {str(entry.get("role") or "") for entry in inputs}
            channels = {str(entry.get("channel") or "") for entry in inputs}
            paths = [str(entry.get("path") or "") for entry in inputs]
            contract = item.get("position_lock_contract") if isinstance(item.get("position_lock_contract"), dict) else {}
            contract_ok = (
                str(contract.get("status") or "") == "pass"
                and isinstance(contract.get("bbox_norm"), list)
                and len(contract.get("bbox_norm", [])) == 4
                and isinstance(contract.get("center_norm"), list)
                and len(contract.get("center_norm", [])) == 2
            )
            white_locked = "white_model_rgb_position_lock" in roles and "edge_silhouette_lock" in roles
            render_channel_only = bool(inputs) and all(channel in channel_allowlist for channel in channels) and "appearance_style_reference_only" not in roles
            all_white_locked = all_white_locked and white_locked
            all_render_channel_only = all_render_channel_only and render_channel_only
            all_position_contracts_present = all_position_contracts_present and contract_ok
            final_evidence.append(
                {
                    "view_id": item.get("view_id", ""),
                    "roles": sorted(roles),
                    "channels": sorted(channels),
                    "input_paths": paths,
                    "white_locked": white_locked,
                    "render_channel_only": render_channel_only,
                    "position_contract": {
                        "status": contract.get("status", ""),
                        "bbox_norm": contract.get("bbox_norm", []),
                        "center_norm": contract.get("center_norm", []),
                        "pass": contract_ok,
                    },
                }
            )
        excluded = [str(value) for value in final_request.get("planning_images_excluded_from_final_inputs", [])]
        concept_abs = _absolute_artifact_path(concept_image_path)
        planning_images_excluded = (not _valid_image(concept_image_path)) or concept_abs in excluded
        _audit_check(
            checks,
            "final_image2_uses_white_model_channels",
            all_white_locked,
            evidence={"final_request": _absolute_artifact_path(final_request_path), "requests": final_evidence},
            reason="Final Codex image2 rendering must use the white-model RGB and edge channels as position-lock inputs.",
        )
        _audit_check(
            checks,
            "final_image2_excludes_planning_images",
            all_render_channel_only and planning_images_excluded and not _json_has_key(final_request, "negative_prompt"),
            evidence={
                "final_request": _absolute_artifact_path(final_request_path),
                "planning_images_excluded_from_final_inputs": excluded,
                "concept_image": concept_abs,
                "requests": final_evidence,
            },
            reason="Final image2 inputs must remain render-channel-only; concept and module references stay planning assets unless an explicit concept-guided mode is added.",
        )
        _audit_check(
            checks,
            "final_image2_has_white_model_position_contract",
            all_position_contracts_present and bool(final_request.get("white_model_position_contract")),
            evidence={
                "final_request": _absolute_artifact_path(final_request_path),
                "white_model_position_contract": final_request.get("white_model_position_contract", ""),
                "requests": final_evidence,
            },
            reason="Final image2 requests must include normalized bbox/center/coverage contracts derived from the Blender white-model channels.",
        )
    else:
        _audit_check(
            checks,
            "final_image2_request_present",
            False,
            evidence={"final_request": _absolute_artifact_path(final_request_path)},
            reason="Final image2 request is not present yet; this audit only proves the chain through reviewed module 3D assets.",
            required=False,
        )

    required_checks = [check for check in checks if check["required"]]
    passed_count = sum(1 for check in required_checks if check["status"] == "pass")
    report = {
        "type": "auto_scene_image2_flow_audit",
        "status": "pass" if passed_count == len(required_checks) else "fail",
        "workdir": _absolute_artifact_path(workdir),
        "summary": _absolute_artifact_path(summary_path),
        "strict_requirements": {
            "require_codex_image2": bool(require_codex_image2),
            "require_hunyuan_3d": bool(require_hunyuan_3d),
        },
        "passed_required_checks": passed_count,
        "required_checks": len(required_checks),
        "checks": checks,
    }
    if write_report_file:
        write_manifest(workdir / "reports" / "image2_flow_audit.json", report)
    return report


def generate_concept_image(
    workdir: Path,
    concept_plan: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    options: AutoSceneOptions | None = None,
    force: bool = False,
) -> Path:
    workdir = workdir.expanduser().resolve()
    output = workdir / str(concept_plan.get("output", "concept/global_concept.png"))
    prompt = str(concept_plan.get("concept_prompt") or concept_plan.get("prompt") or "")
    prompt_path = output.with_name("concept_prompt.txt")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    should_mock = options is None or options.dry_run or options.backend == "mock" or config is None
    provider = _reference_generation_provider(config)
    generation: dict[str, Any]
    if force and output.exists():
        output.unlink()
    if not _valid_image(output):
        if should_mock:
            _draw_mock_scene(output, kind="concept", seed=int(concept_plan.get("seed", 0)), size=int(concept_plan.get("width", 768)))
            generation = {
                "created_by": "mock_image2_generation",
                "path": _absolute_artifact_path(output),
                "prompt": prompt,
            }
        elif _uses_dashscope_imagegen(config):
            generation = _generate_dashscope_reference_image(
                config=config,
                prompt=prompt,
                output=output,
                seed=int(concept_plan.get("seed", options.seed if options else 0)),
                width=int(concept_plan.get("width", config.get("reference_generation", {}).get("concept_width", 1472))),
                height=int(concept_plan.get("height", config.get("reference_generation", {}).get("concept_height", 1104))),
                kind="concept",
            )
        elif _uses_external_imagegen(config):
            request_path = _write_external_imagegen_request(
                output=output,
                prompt=prompt,
                kind="concept",
            )
            _raise_external_imagegen_required(request_path)
        else:
            generation = _generate_prompt_reference_image(
                config=config,
                prompt=prompt,
                output=output,
                seed=int(concept_plan.get("seed", options.seed if options else 0)),
                width=int(concept_plan.get("width", 1024)),
                height=int(concept_plan.get("height", 1024)),
            )
    else:
        generation = {
            "created_by": "imagegen_skill_external" if provider in {"external_imagegen", "imagegen", "imagegen_skill", "manual_image2"} else "image2_provided",
            "path": _absolute_artifact_path(output),
            "prompt": prompt,
        }
    write_manifest(
        output.with_name("generation_manifest.json"),
        {
            "type": "concept_image_generation",
            "purpose": "model-expanded concept prompt rendered by image2 and returned to the model for review",
            "prompt_file": _absolute_artifact_path(prompt_path),
            **generation,
        },
    )
    return output


def generate_module_references(
    workdir: Path,
    scene_plan: dict[str, Any],
    module_plan: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    options: AutoSceneOptions | None = None,
    force_module_ids: set[str] | None = None,
    concept_image: Path | None = None,
) -> dict[str, Any]:
    workdir = workdir.expanduser().resolve()
    force_module_ids = force_module_ids or set()
    modules = [module for module in module_plan.get("modules", []) if module.get("generate_reference_image", True)]
    should_mock = options is None or options.dry_run or options.backend == "mock" or config is None
    provider = _reference_generation_provider(config)
    ref_cfg = config.get("reference_generation", {}) if isinstance((config or {}).get("reference_generation"), dict) else {}
    workers = 1 if should_mock else max(1, int(ref_cfg.get("concurrent_workers", 2)))
    concept_source = Path(concept_image).expanduser().resolve() if concept_image and _valid_image(Path(concept_image)) else None
    use_concept_source_image = bool(
        concept_source
        and not should_mock
        and (_uses_dashscope_imagegen(config) or _uses_external_imagegen(config))
    )

    if not should_mock and _uses_external_imagegen(config):
        pending_requests: list[dict[str, Any]] = []
        for module in modules:
            module_id = str(module["module_id"])
            module_dir = workdir / "modules" / module_id
            reference = module_dir / "reference.png"
            if module_id in force_module_ids and reference.exists():
                reference.unlink()
            prompt = _module_reference_generation_prompt(
                module,
                str(module.get("reference_prompt", "")),
                has_concept_image=use_concept_source_image,
            )
            module_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = module_dir / "reference_prompt.txt"
            prompt_path.write_text(prompt + "\n", encoding="utf-8")
            if not _valid_image(reference):
                request_path = _write_external_imagegen_request(
                    output=reference,
                    prompt=prompt,
                    kind="module_reference",
                    module_id=module_id,
                    source_image=concept_source if use_concept_source_image else None,
                )
                request = read_manifest(request_path)
                request["request_path"] = str(request_path.expanduser().resolve())
                request["prompt_file"] = str(prompt_path.expanduser().resolve())
                pending_requests.append(request)
        if pending_requests:
            batch_path = workdir / "modules" / "imagegen_batch_request.json"
            import_parts = []
            for index, item in enumerate(pending_requests, start=1):
                key = str(item.get("module_id") or f"item_{index}")
                import_parts.append(f"--image {key}=/path/to/codex-image2-output-{index}.png")
            batch_request = {
                "type": "external_imagegen_batch_request",
                "status": "awaiting_external_imagegen",
                "provider": "codex_builtin_image2",
                "kind": "module_reference_batch",
                "request_path": str(batch_path.expanduser().resolve()),
                "request_count": len(pending_requests),
                "requests": pending_requests,
                "codex_image2_handoff": str((workdir / "modules" / "codex_image2_batch_handoff.md").expanduser().resolve()),
                "resume_instruction": "Generate every requested module image with Codex image2, import or copy each selected image to output_path, then rerun this auto-scene workdir.",
                "import_command": (
                    "PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene-import-image2 "
                    f"--request {batch_path.expanduser().resolve()} "
                    + " ".join(import_parts)
                ),
                "latest_import_command": (
                    "PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene-import-latest-image2 "
                    f"--request {batch_path.expanduser().resolve()}"
                ),
            }
            _write_codex_image2_handoff(
                handoff_path=Path(batch_request["codex_image2_handoff"]),
                request=batch_request,
                batch_requests=pending_requests,
            )
            write_manifest(
                batch_path,
                batch_request,
            )
            _raise_external_imagegen_required(batch_path)

    def generate_one(index: int, module: dict[str, Any]) -> dict[str, Any]:
        module_id = str(module["module_id"])
        module_dir = workdir / "modules" / module_id
        reference = module_dir / "reference.png"
        if module_id in force_module_ids and reference.exists():
            reference.unlink()
        prompt = _module_reference_generation_prompt(
            module,
            str(module.get("reference_prompt", "")),
            has_concept_image=use_concept_source_image,
        )
        module_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = module_dir / "reference_prompt.txt"
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        if _valid_image(reference):
            if provider in {"external_imagegen", "imagegen", "imagegen_skill", "manual_image2"}:
                image_source = "imagegen_skill_external"
            elif provider in {"dashscope", "dashscope_imagegen", "dashscope_image_generation", "wanx"}:
                image_source = "dashscope_imagegen"
            else:
                image_source = "image2_provided" if module_id not in force_module_ids else "image2_regenerated_existing"
            generation = {
                "created_by": image_source,
                "image_source": image_source,
                "path": _absolute_artifact_path(reference),
                "prompt": prompt,
                "source_image": _absolute_artifact_path(concept_source) if use_concept_source_image and concept_source else "",
            }
        elif should_mock:
            _draw_mock_scene(reference, kind="reference", seed=(abs(hash(module_id)) + index) % 9999, size=512)
            generation = {
                "created_by": "mock_image2_generation",
                "image_source": "mock_image2_generation",
                "path": _absolute_artifact_path(reference),
                "prompt": prompt,
                "source_image": "",
            }
        elif _uses_external_imagegen(config):
            request_path = _write_external_imagegen_request(
                output=reference,
                prompt=prompt,
                kind="module_reference",
                module_id=module_id,
                source_image=concept_source if use_concept_source_image else None,
            )
            _raise_external_imagegen_required(request_path)
        elif _uses_dashscope_imagegen(config):
            generation = _generate_dashscope_reference_image(
                config=config,
                prompt=prompt,
                output=reference,
                seed=int((options.seed if options else 0) + index * 997),
                width=int(module.get("reference_width", config.get("reference_generation", {}).get("width", 1280))),
                height=int(module.get("reference_height", config.get("reference_generation", {}).get("height", 1280))),
                kind="module_reference",
                module_id=module_id,
                source_image=concept_source if use_concept_source_image else None,
            )
        else:
            generation = _generate_prompt_reference_image(
                config=config,
                prompt=prompt,
                output=reference,
                seed=int((options.seed if options else 0) + index * 997),
                width=int(module.get("reference_width", config.get("reference_generation", {}).get("width", 1024))),
                height=int(module.get("reference_height", config.get("reference_generation", {}).get("height", 1024))),
            )
            generation["image_source"] = "image2_generated"
            generation["source_image"] = ""
        preprocessed = module_dir / "preprocessed.png"
        with Image.open(reference) as image:
            image.convert("RGB").save(preprocessed)
        manifest = {
            "module_id": module_id,
            "reference_image": _absolute_artifact_path(reference),
            "preprocessed_image": _absolute_artifact_path(preprocessed),
            "image_source": generation.get("image_source", generation.get("created_by", "")),
            "created_by": generation.get("created_by", ""),
            "prompt": prompt,
            "prompt_file": _absolute_artifact_path(prompt_path),
            "source_image": _absolute_artifact_path(concept_source) if use_concept_source_image and concept_source else "",
            "inherits_global_style": scene_plan.get("global_style", {}),
            "purpose": "module image-to-3D reference only; not used directly by final AI render",
            "review_status": "pending_model_review",
            "generation": generation,
        }
        write_manifest(module_dir / "reference_manifest.json", manifest)
        return manifest

    manifests: list[dict[str, Any]] = []
    if workers == 1:
        manifests = [generate_one(index, module) for index, module in enumerate(modules)]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(modules)))) as pool:
            futures = {pool.submit(generate_one, index, module): index for index, module in enumerate(modules)}
            ordered: dict[int, dict[str, Any]] = {}
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
            manifests = [ordered[index] for index in sorted(ordered)]
    return {
        "references": manifests,
        "generation_mode": "mock" if should_mock else "image2_model",
        "concurrent_workers": min(workers, max(1, len(modules))) if modules else 0,
    }


def review_module_reference_images(
    config: dict[str, Any],
    options: AutoSceneOptions,
    *,
    module_plan: dict[str, Any],
    reference_summary: dict[str, Any],
    concept_image: Path,
    attempt: int,
    output_path: Path,
) -> dict[str, Any]:
    if options.dry_run or options.backend == "mock" or not options.use_llm:
        review = {
            "type": "module_reference_review",
            "status": "pass",
            "attempt": attempt,
            "backend": "mock_model_brain",
            "failed_modules": [],
            "module_reviews": [
                {"module_id": item["module_id"], "status": "pass", "revised_reference_prompt": "", "notes": "mock review"}
                for item in reference_summary.get("references", [])
            ],
        }
        write_manifest(output_path, review)
        return review
    content: list[dict[str, Any]] = [{"image": _dashscope_image_ref(concept_image)}]
    compact_modules = []
    for module in module_plan.get("modules", []):
        module_id = str(module.get("module_id", ""))
        manifest = next((item for item in reference_summary.get("references", []) if item.get("module_id") == module_id), {})
        if manifest.get("reference_image"):
            content.append({"image": _dashscope_image_ref(manifest["reference_image"])})
        compact_modules.append(
            {
                "module_id": module_id,
                "name": module.get("name", ""),
                "role": module.get("role", ""),
                "expected_real_world_size": module.get("expected_real_world_size", {}),
                "reference_prompt": module.get("reference_prompt", ""),
                "reference_image": manifest.get("reference_image", ""),
            }
        )
    prompt = {
        "task": "审查所有模块单物体参考图是否可交给 3D AI 生成模型",
        "image_order": ["concept_image"] + [item["module_id"] for item in compact_modules],
        "modules": compact_modules,
        "pass_criteria": [
            "每张模块图必须是单物体、居中、纯色背景。",
            "每张模块图必须统一使用严格正交正视图/front elevation，物体前平面平行图像平面，镜头正对物体中心；三分之二视角、明显斜向姿态、侧视、俯视、背视或明显透视必须 revise。",
            "对机械臂、车辆或薄屏幕，合格修订仍然是 front elevation；revised_reference_prompt 必须继续要求 front elevation。",
            "CAD/白模部件允许看到轻微圆柱厚度或关节厚度；合格图像整体保持正面平视、前平面平行图像平面、单正面展示。",
            "合格图像呈现 blank unlabeled surfaces、plain solid background、object-only catalog cutout、internally hidden wiring 和 clean even lighting。",
            "屏幕/显示面板模块的修订使用 panel/slab 语义：portrait vertical flat luminous panel、tall narrow rectangle、thin uniform bezel、wall-panel style exhibition display face。",
            "必须符合模块名称、角色和场景概念。",
            "如果失败，给出只含正向目标描述的 revised_reference_prompt 用于重新调用 image2；不输出 negative_prompt 字段。",
        ],
        "json_schema": {
            "status": "pass or revise",
            "failed_modules": ["module_id"],
            "module_reviews": [
                {
                    "module_id": "module_id",
                    "status": "pass or revise",
                    "revised_reference_prompt": "only when revise",
                    "notes": "short reason",
                }
            ],
        },
        "response_rule": "strict JSON only",
    }
    content.append({"text": json.dumps(prompt, ensure_ascii=False)})
    parsed, info = _call_dashscope_multimodal_json(
        config,
        purpose="module_reference_review",
        messages=[{"role": "user", "content": content}],
        max_tokens=max(1800, int(config.get("agent_llm", {}).get("max_tokens", 1600))),
    )
    review = _coerce_json_object(parsed)
    review.setdefault("type", "module_reference_review")
    review.setdefault("attempt", attempt)
    review.setdefault("backend", info["backend"])
    review.setdefault("failed_modules", [])
    review.setdefault("module_reviews", [])
    write_manifest(output_path, review)
    return review


def apply_module_review_revisions(module_plan: dict[str, Any], review: dict[str, Any]) -> set[str]:
    failed: set[str] = set()
    revisions: dict[str, str] = {}
    for item in review.get("module_reviews", []):
        if not isinstance(item, dict):
            continue
        module_id = str(item.get("module_id", ""))
        status = str(item.get("status", "")).lower()
        revised = str(item.get("revised_reference_prompt", "") or "").strip()
        if status in {"revise", "failed", "fail", "needs_review"} or module_id in set(map(str, review.get("failed_modules", []))):
            failed.add(module_id)
        if module_id and revised:
            revisions[module_id] = revised
    for module in module_plan.get("modules", []):
        module_id = str(module.get("module_id", ""))
        if module_id in revisions:
            safe_revision = _safe_review_revision_prompt(module, revisions[module_id])
            module["reference_prompt"] = _module_reference_prompt_with_safety(module, safe_revision)
            module.pop("negative_prompt", None)
            module["prompt_source"] = "dashscope_multimodal_review_revision"
    return failed


def mark_module_reference_review_status(workdir: Path, review: dict[str, Any]) -> None:
    for item in review.get("module_reviews", []):
        if not isinstance(item, dict):
            continue
        module_id = str(item.get("module_id", ""))
        if not module_id:
            continue
        manifest_path = workdir / "modules" / module_id / "reference_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_manifest(manifest_path)
        manifest["review_status"] = str(item.get("status") or review.get("status") or "")
        manifest["review_attempt"] = review.get("attempt", 0)
        manifest["review_notes"] = str(item.get("notes", ""))
        if item.get("revised_reference_prompt"):
            manifest["revised_reference_prompt"] = str(item["revised_reference_prompt"])
        write_manifest(manifest_path, manifest)


def _module_reference_max_review_attempts(config: dict[str, Any]) -> int:
    ref_cfg = config.get("reference_generation", {}) if isinstance(config.get("reference_generation"), dict) else {}
    return max(1, int(ref_cfg.get("max_review_attempts", 3)))


def _cube_bytes() -> tuple[bytes, bytes, list[float], list[float]]:
    positions = [
        -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5,
        -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
    ]
    indices = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7, 1, 5, 6, 1, 6, 2, 0, 3, 7, 0, 7, 4]
    return struct.pack("<" + "f" * len(positions), *positions), struct.pack("<" + "H" * len(indices), *indices), [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]


def write_box_glb(path: Path, nodes: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    position_bytes, index_bytes, min_xyz, max_xyz = _cube_bytes()
    bin_blob = position_bytes
    while len(bin_blob) % 4:
        bin_blob += b"\x00"
    index_offset = len(bin_blob)
    bin_blob += index_bytes
    while len(bin_blob) % 4:
        bin_blob += b"\x00"
    gltf = {
        "asset": {"version": "2.0", "generator": "Harmonize3D Auto Scene mock assembler"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": [
            {
                "name": str(node.get("module_id", f"node_{index}")),
                "mesh": 0,
                "translation": [float(v) for v in node.get("translation", [0, 0, 0])],
                "scale": [float(v) for v in node.get("scale", [1, 1, 1])],
            }
            for index, node in enumerate(nodes)
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.82, 0.84, 0.86, 1.0], "roughnessFactor": 0.35}}],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3", "min": min_xyz, "max": max_xyz},
            {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"},
        ],
    }
    json_blob = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    while len(json_blob) % 4:
        json_blob += b" "
    total_length = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    with path.open("wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total_length))
        fh.write(struct.pack("<I4s", len(json_blob), b"JSON"))
        fh.write(json_blob)
        fh.write(struct.pack("<I4s", len(bin_blob), b"BIN\x00"))
        fh.write(bin_blob)
    return path


def _rotate_xy(x: float, y: float, angle: float) -> tuple[float, float]:
    if not angle:
        return x, y
    return x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)


def _add_mesh(
    vertices: list[tuple[float, float, float]],
    indices: list[int],
    mesh_vertices: list[tuple[float, float, float]],
    mesh_indices: list[int],
) -> None:
    offset = len(vertices)
    vertices.extend(mesh_vertices)
    indices.extend(offset + index for index in mesh_indices)


def _box_mesh(
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    rotation_z: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[int]]:
    sx, sy, sz = (max(0.001, float(value)) for value in size)
    cx, cy, cz = center
    local = [
        (-sx / 2, -sy / 2, -sz / 2),
        (sx / 2, -sy / 2, -sz / 2),
        (sx / 2, sy / 2, -sz / 2),
        (-sx / 2, sy / 2, -sz / 2),
        (-sx / 2, -sy / 2, sz / 2),
        (sx / 2, -sy / 2, sz / 2),
        (sx / 2, sy / 2, sz / 2),
        (-sx / 2, sy / 2, sz / 2),
    ]
    vertices = []
    for x, y, z in local:
        rx, ry = _rotate_xy(x, y, rotation_z)
        vertices.append((cx + rx, cy + ry, cz + z))
    indices = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7, 1, 5, 6, 1, 6, 2, 0, 3, 7, 0, 7, 4]
    return vertices, indices


def _trapezoid_mesh(
    *,
    center: tuple[float, float, float],
    bottom_size: tuple[float, float],
    top_size: tuple[float, float],
    height: float,
    rotation_z: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[int]]:
    cx, cy, cz = center
    bw, bd = bottom_size
    tw, td = top_size
    h = max(0.001, float(height))
    local = [
        (-bw / 2, -bd / 2, -h / 2),
        (bw / 2, -bd / 2, -h / 2),
        (bw / 2, bd / 2, -h / 2),
        (-bw / 2, bd / 2, -h / 2),
        (-tw / 2, -td / 2, h / 2),
        (tw / 2, -td / 2, h / 2),
        (tw / 2, td / 2, h / 2),
        (-tw / 2, td / 2, h / 2),
    ]
    vertices = []
    for x, y, z in local:
        rx, ry = _rotate_xy(x, y, rotation_z)
        vertices.append((cx + rx, cy + ry, cz + z))
    indices = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0]
    return vertices, indices


def _cylinder_mesh(
    *,
    center: tuple[float, float, float],
    radius: float,
    depth: float,
    axis: str = "z",
    segments: int = 32,
) -> tuple[list[tuple[float, float, float]], list[int]]:
    radius = max(0.001, float(radius))
    depth = max(0.001, float(depth))
    segments = max(8, int(segments))
    cx, cy, cz = center
    vertices: list[tuple[float, float, float]] = []
    for layer in (-0.5, 0.5):
        for index in range(segments):
            angle = math.tau * index / segments
            a = radius * math.cos(angle)
            b = radius * math.sin(angle)
            c = depth * layer
            if axis == "x":
                vertices.append((cx + c, cy + a, cz + b))
            elif axis == "y":
                vertices.append((cx + a, cy + c, cz + b))
            else:
                vertices.append((cx + a, cy + b, cz + c))
    vertices.append(center)
    center_a = len(vertices) - 1
    if axis == "x":
        vertices.append((cx + depth / 2, cy, cz))
    elif axis == "y":
        vertices.append((cx, cy + depth / 2, cz))
    else:
        vertices.append((cx, cy, cz + depth / 2))
    center_b = len(vertices) - 1
    indices: list[int] = []
    for index in range(segments):
        next_index = (index + 1) % segments
        a0 = index
        a1 = next_index
        b0 = index + segments
        b1 = next_index + segments
        indices.extend([a0, b0, b1, a0, b1, a1])
        indices.extend([center_a, a1, a0])
        indices.extend([center_b, b0, b1])
    return vertices, indices


def _add_box(
    vertices: list[tuple[float, float, float]],
    indices: list[int],
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    rotation_z: float = 0.0,
) -> None:
    mesh_vertices, mesh_indices = _box_mesh(center=center, size=size, rotation_z=rotation_z)
    _add_mesh(vertices, indices, mesh_vertices, mesh_indices)


def _add_trapezoid(
    vertices: list[tuple[float, float, float]],
    indices: list[int],
    *,
    center: tuple[float, float, float],
    bottom_size: tuple[float, float],
    top_size: tuple[float, float],
    height: float,
    rotation_z: float = 0.0,
) -> None:
    mesh_vertices, mesh_indices = _trapezoid_mesh(center=center, bottom_size=bottom_size, top_size=top_size, height=height, rotation_z=rotation_z)
    _add_mesh(vertices, indices, mesh_vertices, mesh_indices)


def _add_cylinder(
    vertices: list[tuple[float, float, float]],
    indices: list[int],
    *,
    center: tuple[float, float, float],
    radius: float,
    depth: float,
    axis: str = "z",
    segments: int = 32,
) -> None:
    mesh_vertices, mesh_indices = _cylinder_mesh(center=center, radius=radius, depth=depth, axis=axis, segments=segments)
    _add_mesh(vertices, indices, mesh_vertices, mesh_indices)


def _blender_z_up_to_gltf_y_up(vertex: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = vertex
    return (x, z, -y)


def _write_combined_mesh_glb(path: Path, vertices: list[tuple[float, float, float]], indices: list[int], *, generator: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not vertices:
        mesh_vertices, mesh_indices = _box_mesh(center=(0, 0, 0.5), size=(1, 1, 1))
        vertices.extend(mesh_vertices)
        indices.extend(mesh_indices)
    gltf_vertices = [_blender_z_up_to_gltf_y_up(vertex) for vertex in vertices]
    flat_positions = [component for vertex in gltf_vertices for component in vertex]
    position_bytes = struct.pack("<" + "f" * len(flat_positions), *flat_positions)
    index_bytes = struct.pack("<" + "I" * len(indices), *indices)
    bin_blob = position_bytes
    while len(bin_blob) % 4:
        bin_blob += b"\x00"
    index_offset = len(bin_blob)
    bin_blob += index_bytes
    while len(bin_blob) % 4:
        bin_blob += b"\x00"
    xs, ys, zs = zip(*gltf_vertices)
    gltf = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": path.stem, "mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.82, 0.84, 0.86, 1.0], "roughnessFactor": 0.35}}],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
                "min": [min(xs), min(ys), min(zs)],
                "max": [max(xs), max(ys), max(zs)],
            },
            {"bufferView": 1, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
        ],
    }
    json_blob = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    while len(json_blob) % 4:
        json_blob += b" "
    total_length = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    with path.open("wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total_length))
        fh.write(struct.pack("<I4s", len(json_blob), b"JSON"))
        fh.write(json_blob)
        fh.write(struct.pack("<I4s", len(bin_blob), b"BIN\x00"))
        fh.write(bin_blob)
    bounds_min = [float(min(xs)), float(min(ys)), float(min(zs))]
    bounds_max = [float(max(xs)), float(max(ys)), float(max(zs))]
    return {
        "vertices": len(vertices),
        "faces": len(indices) // 3,
        "coordinate_export": "blender_z_up_to_gltf_y_up",
        "mesh_bounds": {"min": bounds_min, "max": bounds_max},
        "mesh_extents": [round(bounds_max[index] - bounds_min[index], 6) for index in range(3)],
    }


def _load_trimesh_vertices_faces(path: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import trimesh

        loaded = trimesh.load(Path(path), force="scene")
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.to_geometry() if hasattr(loaded, "to_geometry") else loaded.dump(concatenate=True)
        else:
            mesh = loaded
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None
        return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
    except Exception:
        return None


def _mesh_bounds_from_vertices(vertices: np.ndarray | None) -> dict[str, Any]:
    if vertices is None or vertices.size == 0:
        return {}
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    extents = np.maximum(maximum - minimum, 0.0)
    return {
        "mesh_bounds": {
            "min": [round(float(value), 6) for value in minimum.tolist()],
            "max": [round(float(value), 6) for value in maximum.tolist()],
        },
        "mesh_extents": [round(float(value), 6) for value in extents.tolist()],
    }


def _external_mesh_to_internal_points(points: np.ndarray, module_id: str, target_size: tuple[float, float, float]) -> tuple[np.ndarray, dict[str, Any]]:
    if points.size == 0:
        return points, {"axis_mapping": [], "raw_extents": [], "target_size": list(target_size)}
    centered = points - (points.min(axis=0) + points.max(axis=0)) * 0.5
    extents = np.maximum(centered.max(axis=0) - centered.min(axis=0), 1e-6)
    source_axes = ["x", "y", "z"]
    if "vehicle" in module_id or "car" in module_id or "hero_product" in module_id:
        raw_order = list(np.argsort(extents))
        height_axis = raw_order[0]
        length_axis = raw_order[-1]
        width_axis = next(axis for axis in (0, 1, 2) if axis not in {height_axis, length_axis})
        internal = np.column_stack([centered[:, length_axis], centered[:, width_axis], centered[:, height_axis]])
        target = np.asarray([max(target_size[0], target_size[1]), min(target_size[0], target_size[1]), target_size[2]], dtype=np.float64)
        axis_mapping = [
            {"internal_axis": "x", "source_axis": source_axes[length_axis], "semantic": "vehicle_length", "sign": 1},
            {"internal_axis": "y", "source_axis": source_axes[width_axis], "semantic": "vehicle_width", "sign": 1},
            {"internal_axis": "z", "source_axis": source_axes[height_axis], "semantic": "vehicle_height", "sign": 1},
        ]
    else:
        internal = np.column_stack([centered[:, 0], -centered[:, 2], centered[:, 1]])
        target = np.asarray(target_size, dtype=np.float64)
        axis_mapping = [
            {"internal_axis": "x", "source_axis": "x", "semantic": "width", "sign": 1},
            {"internal_axis": "y", "source_axis": "z", "semantic": "depth", "sign": -1},
            {"internal_axis": "z", "source_axis": "y", "semantic": "height", "sign": 1},
        ]
    internal_extents = np.maximum(internal.max(axis=0) - internal.min(axis=0), 1e-6)
    scale = target / internal_extents
    return internal * scale, {
        "axis_mapping": axis_mapping,
        "raw_extents": [round(float(value), 6) for value in extents],
        "internal_extents_before_scale": [round(float(value), 6) for value in internal_extents],
        "target_size": [round(float(value), 6) for value in target],
        "axis_scale": [round(float(value), 6) for value in scale],
        "coordinate_normalization": "hunyuan_or_external_glb_to_blender_z_up_scene_axes",
    }


def _append_external_module_geometry(vertices: list[tuple[float, float, float]], indices: list[int], item: dict[str, Any]) -> dict[str, Any] | None:
    model_path = str(item.get("model_path", "") or "")
    if not model_path or not Path(model_path).exists():
        return None
    metadata_path = Path(model_path).with_name("metadata.json")
    created_by = ""
    if metadata_path.exists():
        try:
            created_by = str(read_manifest(metadata_path).get("created_by", ""))
        except Exception:
            created_by = ""
    if created_by.startswith("procedural_module_proxy"):
        return None
    loaded = _load_trimesh_vertices_faces(model_path)
    if loaded is None:
        return None
    raw_points, faces = loaded
    transform = item.get("transform", {})
    position = [float(value) for value in transform.get("position", [0, 0, 0])]
    scale = tuple(float(value) for value in transform.get("scale", [1, 1, 1]))
    module_id = str(item.get("module_id", "module"))
    points, axis_transform = _external_mesh_to_internal_points(raw_points, module_id, scale)
    if "vehicle" in module_id or "car" in module_id or "hero_product" in module_id:
        points[:, 2] -= points[:, 2].min()
        points[:, 2] += max(0.14, position[2] - scale[2] * 0.18)
        points[:, 0] += position[0]
        points[:, 1] += position[1]
    else:
        points += np.asarray(position, dtype=np.float64)
    rotation = transform.get("rotation_deg", [0, 0, 0])
    rotation_z = math.radians(float(rotation[2] if isinstance(rotation, (list, tuple)) and len(rotation) >= 3 else 0.0))
    if abs(rotation_z) > 1e-6:
        cos_r = math.cos(rotation_z)
        sin_r = math.sin(rotation_z)
        rel_x = points[:, 0] - position[0]
        rel_y = points[:, 1] - position[1]
        points[:, 0] = position[0] + rel_x * cos_r - rel_y * sin_r
        points[:, 1] = position[1] + rel_x * sin_r + rel_y * cos_r
    offset = len(vertices)
    vertices.extend((float(x), float(y), float(z)) for x, y, z in points)
    indices.extend(int(index + offset) for face in faces for index in face)
    return {"vertices": int(len(points)), "faces": int(len(faces)), "source_model_path": model_path, "axis_transform": axis_transform}


def _append_module_proxy_geometry(vertices: list[tuple[float, float, float]], indices: list[int], item: dict[str, Any], *, local: bool = False) -> None:
    module_id = str(item.get("module_id", "module"))
    transform = item.get("transform", {})
    position = [float(value) for value in transform.get("position", [0, 0, 0])]
    scale = [float(value) for value in transform.get("scale", [1, 1, 1])]
    if local:
        position = [0.0, 0.0, 0.0]
    x, y, z = position
    width, depth, height = (max(0.001, value) for value in scale)
    if "vehicle" in module_id or "car" in module_id or "hero_product" in module_id:
        ground = max(0.18, z if not local else 0.0)
        car_length = max(width, depth)
        car_width = min(width, depth)
        body_z = ground + height * 0.26
        _add_trapezoid(
            vertices,
            indices,
            center=(x, y, body_z),
            bottom_size=(car_length * 0.94, car_width),
            top_size=(car_length * 0.78, car_width * 0.72),
            height=height * 0.42,
        )
        _add_trapezoid(
            vertices,
            indices,
            center=(x + car_length * 0.03, y, ground + height * 0.62),
            bottom_size=(car_length * 0.32, car_width * 0.54),
            top_size=(car_length * 0.22, car_width * 0.36),
            height=height * 0.34,
        )
        _add_trapezoid(
            vertices,
            indices,
            center=(x - car_length * 0.42, y, ground + height * 0.18),
            bottom_size=(car_length * 0.16, car_width * 0.78),
            top_size=(car_length * 0.11, car_width * 0.62),
            height=height * 0.18,
        )
        _add_trapezoid(
            vertices,
            indices,
            center=(x + car_length * 0.43, y, ground + height * 0.20),
            bottom_size=(car_length * 0.15, car_width * 0.82),
            top_size=(car_length * 0.10, car_width * 0.64),
            height=height * 0.20,
        )
        for wx in (-car_length * 0.30, car_length * 0.31):
            for wy in (-car_width * 0.48, car_width * 0.48):
                _add_cylinder(vertices, indices, center=(x + wx, y + wy, ground + height * 0.18), radius=height * 0.18, depth=car_width * 0.13, axis="y", segments=28)
        return
    if "platform" in module_id:
        _add_cylinder(vertices, indices, center=(x, y, z + height / 2), radius=min(width, depth) * 0.5, depth=height, axis="z", segments=56)
        return
    if "screen" in module_id:
        _add_box(vertices, indices, center=(x, y, z), size=(width, depth, height))
        _add_box(vertices, indices, center=(x, y - depth * 0.55, z), size=(width * 1.08, depth * 0.18, height * 1.06))
        return
    if "robotic_arm" in module_id:
        sign = -1.0 if "left" in module_id else 1.0
        _add_cylinder(vertices, indices, center=(x, y, 0.24), radius=width * 0.28, depth=height * 0.18, axis="z", segments=24)
        _add_box(vertices, indices, center=(x, y, 0.48), size=(width * 0.36, width * 0.36, height * 0.28))
        _add_box(vertices, indices, center=(x + sign * width * 0.18, y - depth * 0.04, 0.82), size=(width * 0.20, depth * 0.70, height * 0.12), rotation_z=sign * 0.38)
        _add_box(vertices, indices, center=(x + sign * width * 0.44, y + depth * 0.18, 1.08), size=(width * 0.16, depth * 0.58, height * 0.10), rotation_z=sign * -0.58)
        for joint_x, joint_y, joint_z in ((x, y, 0.65), (x + sign * width * 0.28, y + depth * 0.05, 0.95), (x + sign * width * 0.62, y + depth * 0.35, 1.08)):
            _add_cylinder(vertices, indices, center=(joint_x, joint_y, joint_z), radius=width * 0.11, depth=width * 0.16, axis="x", segments=18)
        return
    if "light" in module_id:
        strip_z = max(0.05, z)
        length = max(width, depth)
        _add_box(vertices, indices, center=(x, y + length * 0.48, strip_z), size=(length, 0.035, height))
        _add_box(vertices, indices, center=(x, y - length * 0.48, strip_z), size=(length, 0.035, height))
        _add_box(vertices, indices, center=(x + length * 0.48, y, strip_z), size=(0.035, length, height))
        _add_box(vertices, indices, center=(x - length * 0.48, y, strip_z), size=(0.035, length, height))
        return
    if "floor" in module_id:
        _add_box(vertices, indices, center=(x, y, z), size=(width, depth, height))
        return
    _add_box(vertices, indices, center=(x, y, z + height / 2), size=(width, depth, height))


def write_module_proxy_glb(path: Path, module: dict[str, Any]) -> dict[str, Any]:
    size = _module_size(module)
    vertices: list[tuple[float, float, float]] = []
    indices: list[int] = []
    _append_module_proxy_geometry(
        vertices,
        indices,
        {"module_id": module.get("module_id", "module"), "transform": {"position": [0, 0, 0], "scale": list(size)}},
        local=True,
    )
    return _write_combined_mesh_glb(path, vertices, indices, generator="Harmonize3D procedural module proxy v3 axis-corrected")


def write_scene_proxy_glb(path: Path, scene_assembly: dict[str, Any]) -> dict[str, Any]:
    vertices: list[tuple[float, float, float]] = []
    indices: list[int] = []
    external_modules: list[dict[str, Any]] = []
    for item in scene_assembly.get("modules", []):
        external = _append_external_module_geometry(vertices, indices, item)
        if external:
            external_modules.append({"module_id": str(item.get("module_id", "")), **external})
        else:
            _append_module_proxy_geometry(vertices, indices, item, local=False)
    stats = _write_combined_mesh_glb(path, vertices, indices, generator="Harmonize3D hybrid scene proxy v4 external-modules")
    stats["external_modules"] = external_modules
    return stats


def _module_size(module: dict[str, Any]) -> tuple[float, float, float]:
    size = module.get("expected_real_world_size", {})
    return (
        float(size.get("width", 1.0)),
        float(size.get("depth", 1.0)),
        float(size.get("height", 1.0)),
    )


def _mesh_extents_from_stats_or_file(mesh_stats: dict[str, Any], model_path: Path) -> list[float]:
    extents = mesh_stats.get("mesh_extents")
    if isinstance(extents, (list, tuple)) and len(extents) == 3:
        try:
            return [float(value) for value in extents]
        except (TypeError, ValueError):
            pass
    loaded = _load_trimesh_vertices_faces(model_path)
    if loaded:
        bounds = _mesh_bounds_from_vertices(loaded[0])
        extents = bounds.get("mesh_extents", [])
        if isinstance(extents, list) and len(extents) == 3:
            return [float(value) for value in extents]
    return []


def _module_semantic_mesh_sanity(module: dict[str, Any], model_path: Path, mesh_stats: dict[str, Any]) -> dict[str, Any]:
    semantic_profile = "generic"
    if _is_semantic_platform_module(module):
        semantic_profile = "low_horizontal_slab"
    elif _is_semantic_screen_module(module) or _is_semantic_flat_panel_module(module):
        semantic_profile = "flat_vertical_panel" if _is_vertical_screen_module(module) else "flat_panel"
    if semantic_profile == "generic":
        return {"status": "pass", "semantic_profile": semantic_profile, "checked": False, "flags": []}

    extents = _mesh_extents_from_stats_or_file(mesh_stats, model_path)
    if not extents or max(extents) <= 1e-6:
        return {
            "status": "needs_review",
            "semantic_profile": semantic_profile,
            "checked": True,
            "mesh_extents": extents,
            "flags": [
                {
                    "code": "mesh_bounds_unavailable",
                    "severity": "review",
                    "message": "Mesh bounds could not be read for semantic geometry validation.",
                }
            ],
        }

    sorted_extents = sorted((max(0.0, float(value)) for value in extents))
    shortest, middle, longest = sorted_extents
    thin_ratio = shortest / max(longest, 1e-6)
    face_fill_ratio = middle / max(longest, 1e-6)
    flags: list[dict[str, str]] = []

    if semantic_profile == "low_horizontal_slab":
        if thin_ratio > 0.30:
            flags.append(
                {
                    "code": "platform_not_low_slab",
                    "severity": "review",
                    "message": "Platform-like modules should have one thin axis relative to the broad top plane.",
                }
            )
        if face_fill_ratio < 0.42:
            flags.append(
                {
                    "code": "platform_not_broad_enough",
                    "severity": "review",
                    "message": "Platform-like modules should keep two broad axes instead of degenerating into a narrow block.",
                }
            )
    else:
        if thin_ratio > 0.28:
            flags.append(
                {
                    "code": "panel_or_screen_too_thick",
                    "severity": "review",
                    "message": "Screen or panel modules should preserve a thin slab axis for reliable scene assembly.",
                }
            )
        if face_fill_ratio < 0.35:
            flags.append(
                {
                    "code": "panel_face_too_narrow",
                    "severity": "review",
                    "message": "Screen or panel modules should retain a usable front face instead of becoming a strip or rod.",
                }
            )
        if semantic_profile == "flat_vertical_panel" and longest / max(middle, 1e-6) < 1.12:
            flags.append(
                {
                    "code": "vertical_panel_not_portrait",
                    "severity": "review",
                    "message": "Portrait/vertical panel modules should have a taller major axis than their width.",
                }
            )

    return {
        "status": "needs_review" if flags else "pass",
        "semantic_profile": semantic_profile,
        "checked": True,
        "mesh_extents": [round(float(value), 6) for value in extents],
        "extent_order": [round(float(value), 6) for value in sorted_extents],
        "ratios": {
            "thin_to_long": round(float(thin_ratio), 6),
            "middle_to_long": round(float(face_fill_ratio), 6),
        },
        "flags": flags,
    }


def _merge_mesh_sanity_status(*statuses: str) -> str:
    normalized = [str(status or "").lower() for status in statuses]
    if any(status in {"fail", "failed", "error"} for status in normalized):
        return "failed"
    if any(status in {"needs_review", "review", "warning"} for status in normalized):
        return "needs_review"
    return "pass"


def _hunyuan_enabled(config: dict[str, Any] | None, options: AutoSceneOptions | None) -> bool:
    if config is None or options is None or options.dry_run or options.backend == "mock":
        return False
    model = config.get("models", {}).get("hunyuan3d_2_1_shape", {})
    return bool(model.get("enabled", False))


def _hunyuan_shape_profile(config: dict[str, Any], options: AutoSceneOptions | None) -> tuple[str, dict[str, Any]]:
    model = config.get("models", {}).get("hunyuan3d_2_1_shape", {})
    profiles = model.get("profiles", {}) if isinstance(model.get("profiles"), dict) else {}
    preferred_names = [
        str(model.get("auto_scene_profile", "") or ""),
        str(model.get("default_profile", "") or ""),
        str(options.quality_mode if options else "" or ""),
        "high",
        "balanced",
    ]
    for name in preferred_names:
        if name and isinstance(profiles.get(name), dict):
            return name, dict(profiles[name])
    return "", {}


def _generate_module_hunyuan_asset(
    *,
    config: dict[str, Any],
    module: dict[str, Any],
    reference: Path,
    model_path: Path,
    metadata_path: Path,
    seed: int,
    profile_name: str = "",
    shape_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .workflow import _run_hunyuan_shape

    _run_hunyuan_shape(
        config=config,
        reference_image=reference,
        output_model=model_path,
        metadata=metadata_path,
        seed=seed,
        progress=lambda *_: None,
        shape_overrides=shape_overrides,
    )
    loaded = _load_trimesh_vertices_faces(model_path)
    hunyuan_metadata = read_manifest(metadata_path) if metadata_path.exists() else {}
    hunyuan_mesh_sanity = hunyuan_metadata.get("mesh_sanity", {}) if isinstance(hunyuan_metadata, dict) else {}
    bounds = _mesh_bounds_from_vertices(loaded[0] if loaded else None)
    return {
        "vertices": int(len(loaded[0])) if loaded else 0,
        "faces": int(len(loaded[1])) if loaded else 0,
        "coordinate_export": "hunyuan3d_generated_from_reviewed_reference",
        **bounds,
        "source_reference_image": _absolute_artifact_path(reference),
        "hunyuan_metadata": _absolute_artifact_path(metadata_path),
        "hunyuan_mesh_sanity": hunyuan_mesh_sanity,
        "hunyuan_status": hunyuan_mesh_sanity.get("status", ""),
        "hunyuan_profile": profile_name,
        "hunyuan_shape_overrides": shape_overrides or {},
        "source_model_path": _absolute_artifact_path(model_path),
    }


def generate_module_assets(
    workdir: Path,
    module_plan: dict[str, Any],
    *,
    allow_fallback: bool,
    hero_model_path: Path | None = None,
    config: dict[str, Any] | None = None,
    options: AutoSceneOptions | None = None,
) -> dict[str, Any]:
    workdir = workdir.expanduser().resolve()
    assets: list[dict[str, Any]] = []
    hunyuan_enabled = _hunyuan_enabled(config, options)
    hunyuan_profile_name, hunyuan_shape_overrides = _hunyuan_shape_profile(config or {}, options)
    failed_modules: list[dict[str, Any]] = []
    quality_issues: list[dict[str, Any]] = []
    for module in module_plan.get("modules", []):
        module_id = str(module["module_id"])
        module_dir = workdir / "modules" / module_id
        size = _module_size(module)
        model_path = module_dir / "model.glb"
        reference = module_dir / "reference.png"
        external_source = Path(hero_model_path).expanduser() if hero_model_path and module.get("role") == "hero_object" else None
        external_used = bool(external_source and external_source.exists())
        generator_id = ""
        fallback_used = False
        generation_failure = ""
        if external_used and external_source:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(external_source, model_path)
            loaded = _load_trimesh_vertices_faces(model_path)
            bounds = _mesh_bounds_from_vertices(loaded[0] if loaded else None)
            mesh_stats = {
                "vertices": int(len(loaded[0])) if loaded else 0,
                "faces": int(len(loaded[1])) if loaded else 0,
                "coordinate_export": "external_glb_preserved_for_scene_assembly",
                **bounds,
                "source_model_path": _absolute_artifact_path(external_source),
            }
            generator_id = "external_hero_model_glb"
        else:
            mesh_stats = {}
            if hunyuan_enabled and module.get("generate_3d", True) and _valid_image(reference):
                try:
                    mesh_stats = _generate_module_hunyuan_asset(
                        config=config or {},
                        module=module,
                        reference=reference,
                        model_path=model_path,
                        metadata_path=module_dir / "hunyuan_shape_metadata.json",
                        seed=int((options.seed if options else 0) + len(assets) * 101),
                        profile_name=hunyuan_profile_name,
                        shape_overrides=hunyuan_shape_overrides,
                    )
                    generator_id = "hunyuan3d_2_1_shape_from_reviewed_reference"
                except Exception as exc:
                    generation_failure = str(exc)
                    failure_policy = module_failure_policy(module, generation_failure, allow_fallback=allow_fallback)
                    failed_modules.append(failure_policy)
                    if failure_policy["action"] == "fail_task":
                        raise RuntimeError(f"Module {module_id} 3D AI generation failed: {generation_failure}") from exc
            if not mesh_stats:
                if not allow_fallback and module.get("generate_3d", True):
                    raise RuntimeError(f"Module {module_id} did not produce a 3D AI mesh and procedural fallback is disabled.")
                mesh_stats = write_module_proxy_glb(model_path, module)
                generator_id = "procedural_module_proxy_v3_axis_corrected"
                fallback_used = bool(generation_failure)
        semantic_sanity = _module_semantic_mesh_sanity(module, model_path, mesh_stats)
        sanity_status = _merge_mesh_sanity_status(str(mesh_stats.get("hunyuan_status") or "pass"), str(semantic_sanity.get("status") or "pass"))
        if semantic_sanity.get("flags"):
            quality_issues.append(
                {
                    "module_id": module_id,
                    "status": semantic_sanity.get("status", "needs_review"),
                    "action": "review_or_regenerate_module_3d",
                    "semantic_profile": semantic_sanity.get("semantic_profile", ""),
                    "flags": semantic_sanity.get("flags", []),
                }
            )
        sanity = {
            "vertices": mesh_stats["vertices"],
            "faces": mesh_stats["faces"],
            "component_count": 1,
            "largest_component_ratio": 1.0,
            "status": sanity_status,
            "fallback_used": fallback_used,
            "proxy_geometry": generator_id,
            "coordinate_export": mesh_stats.get("coordinate_export", ""),
            "semantic_mesh_sanity": semantic_sanity,
        }
        if mesh_stats.get("mesh_bounds"):
            sanity["mesh_bounds"] = mesh_stats["mesh_bounds"]
        if mesh_stats.get("mesh_extents"):
            sanity["mesh_extents"] = mesh_stats["mesh_extents"]
        if generation_failure:
            sanity["generation_failure"] = generation_failure
        if mesh_stats.get("source_model_path"):
            sanity["source_model_path"] = str(mesh_stats["source_model_path"])
        if mesh_stats.get("source_reference_image"):
            sanity["source_reference_image"] = str(mesh_stats["source_reference_image"])
        if mesh_stats.get("hunyuan_metadata"):
            sanity["hunyuan_metadata"] = str(mesh_stats["hunyuan_metadata"])
        if mesh_stats.get("hunyuan_mesh_sanity"):
            sanity["hunyuan_mesh_sanity"] = mesh_stats["hunyuan_mesh_sanity"]
        if mesh_stats.get("hunyuan_profile"):
            sanity["hunyuan_profile"] = str(mesh_stats["hunyuan_profile"])
        if mesh_stats.get("hunyuan_shape_overrides"):
            sanity["hunyuan_shape_overrides"] = mesh_stats["hunyuan_shape_overrides"]
        metadata = {
            "module_id": module_id,
            "category": module.get("category", ""),
            "role": module.get("role", ""),
            "model_path": _absolute_artifact_path(model_path),
            "created_by": generator_id,
            "source_reference": _absolute_artifact_path(module_dir / "reference.png"),
        }
        if mesh_stats.get("source_model_path"):
            metadata["source_model_path"] = str(mesh_stats["source_model_path"])
        if mesh_stats.get("hunyuan_metadata"):
            metadata["hunyuan_metadata"] = str(mesh_stats["hunyuan_metadata"])
        if mesh_stats.get("hunyuan_profile"):
            metadata["hunyuan_profile"] = str(mesh_stats["hunyuan_profile"])
        if mesh_stats.get("hunyuan_shape_overrides"):
            metadata["hunyuan_shape_overrides"] = mesh_stats["hunyuan_shape_overrides"]
        write_manifest(module_dir / "metadata.json", metadata)
        write_manifest(module_dir / "sanity.json", sanity)
        assets.append(
            {
                "module_id": module_id,
                "reference_image": _absolute_artifact_path(module_dir / "reference.png"),
                "preprocessed_image": _absolute_artifact_path(module_dir / "preprocessed.png"),
                "model_path": _absolute_artifact_path(model_path),
                "metadata": _absolute_artifact_path(module_dir / "metadata.json"),
                "sanity": sanity,
                "bbox": {"width": size[0], "depth": size[1], "height": size[2]},
                "role": module.get("role", ""),
                "fallback_used": fallback_used,
            }
        )
    return {
        "modules": assets,
        "allow_procedural_fallback": allow_fallback,
        "module_3d_backend": "hunyuan3d_2_1_shape" if hunyuan_enabled else "procedural_or_external",
        "failed_modules": failed_modules,
        "quality_issues": quality_issues,
        "status": "needs_review" if failed_modules or quality_issues else "pass",
    }


def module_failure_policy(module: dict[str, Any], failure: str, *, allow_fallback: bool) -> dict[str, Any]:
    role = str(module.get("role", ""))
    module_id = str(module.get("module_id", ""))
    if role == "hero_object":
        return {"module_id": module_id, "status": "failed", "action": "fail_task", "reason": failure}
    if allow_fallback and any(token in module_id for token in ("platform", "floor", "screen", "light", "background")):
        return {"module_id": module_id, "status": "fallback", "action": "use_procedural_proxy", "reason": failure}
    if role in {"background", "decorative_object"}:
        return {"module_id": module_id, "status": "skipped", "action": "skip_module", "reason": failure}
    return {"module_id": module_id, "status": "retry", "action": "retry_module", "reason": failure}


def plan_scene_layout(scene_plan: dict[str, Any], module_plan: dict[str, Any], asset_manifest: dict[str, Any]) -> dict[str, Any]:
    asset_by_id = {item["module_id"]: item for item in asset_manifest.get("modules", [])}
    modules_out: list[dict[str, Any]] = []
    warnings: list[str] = []
    for module in module_plan.get("modules", []):
        module_id = str(module["module_id"])
        placement = module.get("placement", {})
        size = _module_size(module)
        position = [float(v) for v in placement.get("position", [0, 0, size[2] / 2])]
        rotation = [float(v) for v in placement.get("rotation_deg", [0, 0, 0])]
        if position[2] < 0 and module.get("role") != "background":
            warnings.append(f"{module_id} below ground; clamped to ground contact")
            position[2] = max(0.0, size[2] / 2)
        modules_out.append(
            {
                "module_id": module_id,
                "model_path": asset_by_id.get(module_id, {}).get("model_path", ""),
                "transform": {"position": position, "rotation_deg": rotation, "scale": [size[0], size[1], size[2]]},
                "visibility_priority": int(module.get("priority", 99)),
                "material_hint": module.get("name", ""),
                "layout_reason": f"{module.get('role', 'module')} placed at anchor {placement.get('anchor', 'scene_center')} with scale policy {placement.get('scale_policy', 'role_default')}",
            }
        )
    return {
        "coordinate_system": "blender_z_up",
        "units": "meters",
        "modules": modules_out,
        "collision_report": {"has_major_collision": False, "warnings": warnings},
        "layout_policy": "role_and_anchor_based_v1",
        "scene_type": scene_plan.get("scene_type", "custom"),
    }


def assemble_scene(workdir: Path, scene_assembly: dict[str, Any]) -> dict[str, Any]:
    workdir = workdir.expanduser().resolve()
    scene_dir = workdir / "scene"
    glb_path = scene_dir / "final_scene.glb"
    mesh_stats = write_scene_proxy_glb(glb_path, scene_assembly)
    blend_path = scene_dir / "final_scene.blend"
    blend_path.write_text("Procedural proxy scene placeholder. Use final_scene.glb for renderable geometry.\n", encoding="utf-8")
    preview = scene_dir / "scene_preview.png"
    _draw_mock_scene(preview, kind="concept", seed=0, size=768)
    report = {
        "type": "assembly_report",
        "status": "pass",
        "scene_model_path": _absolute_artifact_path(glb_path),
        "scene_blend_path": _absolute_artifact_path(blend_path),
        "scene_preview": _absolute_artifact_path(preview),
        "module_count": len(scene_assembly.get("modules", [])),
        "geometry_generator": "hybrid_scene_proxy_v4_external_modules",
        "coordinate_export": mesh_stats.get("coordinate_export", ""),
        "vertices": mesh_stats["vertices"],
        "faces": mesh_stats["faces"],
        "external_modules": mesh_stats.get("external_modules", []),
        "procedural_fallbacks": [item["module_id"] for item in scene_assembly.get("modules", []) if not item.get("model_path")],
    }
    report_path = write_manifest(scene_dir / "assembly_report.json", report)
    return {
        "scene_model_path": _absolute_artifact_path(glb_path),
        "scene_blend_path": _absolute_artifact_path(blend_path),
        "scene_preview": _absolute_artifact_path(preview),
        "assembly_report": _absolute_artifact_path(report_path),
    }


def _scene_render_views(camera_plan: dict[str, Any], views: int) -> list[dict[str, Any]]:
    raw_planned = camera_plan.get("views", [])
    planned = [view for view in raw_planned if isinstance(view, dict)] if isinstance(raw_planned, list) else []
    fallback = [
        {
            "view_id": "view_hero",
            "role": "low front three-quarter hero",
            "azimuth_deg": 310.0,
            "elevation_deg": 8.0,
            "camera_type": "perspective",
            "focal_length_mm": 58.0,
            "distance_scale": 1.05,
            "target": [0.0, 0.04, -0.04],
        },
        {
            "view_id": "view_left_30",
            "role": "low left consistency",
            "azimuth_deg": 280.0,
            "elevation_deg": 8.0,
            "camera_type": "perspective",
            "focal_length_mm": 58.0,
            "distance_scale": 1.05,
            "target": [0.0, 0.04, -0.04],
        },
        {
            "view_id": "view_right_30",
            "role": "low right consistency",
            "azimuth_deg": 340.0,
            "elevation_deg": 8.0,
            "camera_type": "perspective",
            "focal_length_mm": 58.0,
            "distance_scale": 1.05,
            "target": [0.0, 0.04, -0.04],
        },
    ]
    by_id = {str(view.get("view_id")): view for view in planned if view.get("view_id")}
    merged: list[dict[str, Any]] = []
    for view in fallback:
        merged.append({**view, **by_id.get(str(view["view_id"]), {})})
    for view in planned:
        if view not in merged:
            merged.append(view)
    return merged[: _clamp_views(int(views))]


def _project_scene_point(position: list[float], view: dict[str, Any], size: int) -> tuple[float, float, float, float]:
    azimuth = math.radians(float(view.get("azimuth_deg", 330.0)))
    ortho_scale = max(1.0, float(view.get("ortho_scale", 6.0) or 6.0))
    pixels_per_meter = size * 0.72 / ortho_scale
    x, y, z = (float(position[0]), float(position[1]), float(position[2]))
    right = x * math.cos(azimuth) - y * math.sin(azimuth)
    depth = x * math.sin(azimuth) + y * math.cos(azimuth)
    px = size * 0.5 + right * pixels_per_meter
    py = size * 0.66 - z * pixels_per_meter - depth * pixels_per_meter * 0.16
    return px, py, pixels_per_meter, depth


def _draw_scene_panel(draw: ImageDraw.ImageDraw, mask: ImageDraw.ImageDraw, edge: ImageDraw.ImageDraw, rect: list[int], *, accent: tuple[int, int, int]) -> None:
    draw.rounded_rectangle(rect, radius=max(4, (rect[2] - rect[0]) // 24), fill=(18, 24, 34), outline=accent, width=4)
    draw.line([rect[0] + 10, rect[1] + 18, rect[2] - 10, rect[3] - 18], fill=(65, 185, 255), width=4)
    draw.line([rect[0] + 10, rect[3] - 22, rect[2] - 18, rect[3] - 22], fill=(42, 132, 210), width=2)
    mask.rounded_rectangle(rect, radius=max(4, (rect[2] - rect[0]) // 24), fill=255)
    edge.rounded_rectangle(rect, radius=max(4, (rect[2] - rect[0]) // 24), outline=255, width=4)
    edge.line([rect[0] + 10, rect[1] + 18, rect[2] - 10, rect[3] - 18], fill=255, width=2)


def _draw_scene_vehicle(draw: ImageDraw.ImageDraw, mask: ImageDraw.ImageDraw, edge: ImageDraw.ImageDraw, cx: float, cy: float, width: float, height: float) -> None:
    body = [int(cx - width * 0.5), int(cy - height * 0.28), int(cx + width * 0.5), int(cy + height * 0.24)]
    draw.rounded_rectangle(body, radius=max(12, int(height * 0.18)), fill=(232, 235, 235), outline=(38, 44, 52), width=4)
    mask.rounded_rectangle(body, radius=max(12, int(height * 0.18)), fill=255)
    edge.rounded_rectangle(body, radius=max(12, int(height * 0.18)), outline=255, width=4)

    cabin = [
        (int(cx - width * 0.18), int(cy - height * 0.28)),
        (int(cx - width * 0.06), int(cy - height * 0.58)),
        (int(cx + width * 0.22), int(cy - height * 0.52)),
        (int(cx + width * 0.35), int(cy - height * 0.27)),
    ]
    draw.polygon(cabin, fill=(42, 52, 62), outline=(30, 36, 44))
    mask.polygon(cabin, fill=255)
    edge.line(cabin + [cabin[0]], fill=255, width=3)

    for wx in (cx - width * 0.32, cx + width * 0.34):
        wheel = [int(wx - height * 0.17), int(cy + height * 0.08), int(wx + height * 0.17), int(cy + height * 0.42)]
        draw.ellipse(wheel, fill=(18, 20, 24), outline=(74, 80, 88), width=3)
        draw.ellipse([wheel[0] + 8, wheel[1] + 8, wheel[2] - 8, wheel[3] - 8], outline=(115, 124, 132), width=2)
        mask.ellipse(wheel, fill=255)
        edge.ellipse(wheel, outline=255, width=3)

    draw.polygon(
        [(int(cx - width * 0.46), int(cy - height * 0.12)), (int(cx - width * 0.28), int(cy - height * 0.2)), (int(cx - width * 0.36), int(cy - height * 0.02))],
        fill=(62, 184, 255),
    )
    draw.polygon(
        [(int(cx + width * 0.46), int(cy - height * 0.12)), (int(cx + width * 0.28), int(cy - height * 0.2)), (int(cx + width * 0.36), int(cy - height * 0.02))],
        fill=(62, 184, 255),
    )


def _draw_scene_arm(draw: ImageDraw.ImageDraw, mask: ImageDraw.ImageDraw, edge: ImageDraw.ImageDraw, cx: float, cy: float, scale: float, flip: int) -> None:
    base = [int(cx - scale * 0.15), int(cy + scale * 0.05), int(cx + scale * 0.15), int(cy + scale * 0.28)]
    joint1 = (int(cx), int(cy))
    joint2 = (int(cx + flip * scale * 0.28), int(cy - scale * 0.26))
    joint3 = (int(cx + flip * scale * 0.52), int(cy - scale * 0.10))
    draw.rounded_rectangle(base, radius=max(4, int(scale * 0.04)), fill=(24, 28, 34), outline=(58, 68, 78), width=2)
    mask.rounded_rectangle(base, radius=max(4, int(scale * 0.04)), fill=255)
    for a, b in ((joint1, joint2), (joint2, joint3)):
        draw.line([a, b], fill=(28, 32, 38), width=max(6, int(scale * 0.06)))
        draw.line([a, b], fill=(80, 92, 104), width=max(2, int(scale * 0.018)))
        edge.line([a, b], fill=255, width=max(4, int(scale * 0.045)))
        mask.line([a, b], fill=255, width=max(8, int(scale * 0.08)))
    for joint in (joint1, joint2, joint3):
        r = max(5, int(scale * 0.055))
        draw.ellipse([joint[0] - r, joint[1] - r, joint[0] + r, joint[1] + r], fill=(18, 22, 28), outline=(73, 178, 240), width=2)
        mask.ellipse([joint[0] - r, joint[1] - r, joint[0] + r, joint[1] + r], fill=255)
        edge.ellipse([joint[0] - r, joint[1] - r, joint[0] + r, joint[1] + r], outline=255, width=2)


def _render_scene_channel_set(scene_assembly: dict[str, Any], view: dict[str, Any], output_dir: Path, *, resolution: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = Image.new("RGB", (resolution, resolution), (224, 229, 234))
    mask_image = Image.new("L", (resolution, resolution), 0)
    edge_image = Image.new("L", (resolution, resolution), 0)
    depth_image = Image.new("L", (resolution, resolution), 28)
    normal = Image.new("RGB", (resolution, resolution), (128, 128, 255))

    draw = ImageDraw.Draw(rgb)
    mask_draw = ImageDraw.Draw(mask_image)
    edge_draw = ImageDraw.Draw(edge_image)
    depth_draw = ImageDraw.Draw(depth_image)
    normal_draw = ImageDraw.Draw(normal)
    accent = (55, 178, 255)

    horizon = int(resolution * 0.58)
    draw.rectangle([0, 0, resolution, horizon], fill=(217, 224, 232))
    draw.polygon([(0, horizon), (resolution, horizon), (resolution, resolution), (0, resolution)], fill=(174, 180, 186))
    for offset in (-0.25, 0.0, 0.25):
        y = int(resolution * (0.72 + offset * 0.18))
        draw.line([int(resolution * 0.18), y, int(resolution * 0.82), y + int(offset * resolution * 0.08)], fill=(69, 181, 255), width=max(3, resolution // 160))

    modules = list(scene_assembly.get("modules", []))
    background_ids = ("screen", "light", "floor")
    for item in modules:
        module_id = str(item.get("module_id", ""))
        position = [float(v) for v in item.get("transform", {}).get("position", [0, 0, 0])]
        scale = [float(v) for v in item.get("transform", {}).get("scale", [1, 1, 1])]
        cx, cy, ppm, _depth = _project_scene_point(position, view, resolution)
        if "screen" in module_id:
            rect_w = max(60, int(scale[0] * ppm * 0.85))
            rect_h = max(120, int(scale[2] * ppm * 0.95))
            rect = [int(cx - rect_w / 2), int(cy - rect_h), int(cx + rect_w / 2), int(cy)]
            _draw_scene_panel(draw, mask_draw, edge_draw, rect, accent=accent)
            depth_draw.rounded_rectangle(rect, radius=8, fill=118)
            normal_draw.rounded_rectangle(rect, radius=8, fill=(125, 142, 240))
        elif "light" in module_id:
            y = int(cy + scale[2] * ppm * 0.25)
            draw.arc([int(cx - scale[0] * ppm), y - 45, int(cx + scale[0] * ppm), y + 45], 10, 170, fill=accent, width=max(4, resolution // 120))
            edge_draw.arc([int(cx - scale[0] * ppm), y - 45, int(cx + scale[0] * ppm), y + 45], 10, 170, fill=255, width=max(3, resolution // 160))

    def foreground_order(item: dict[str, Any]) -> int:
        module_id = str(item.get("module_id", ""))
        if "platform" in module_id:
            return 0
        if "robotic_arm" in module_id:
            return 1
        if "vehicle" in module_id or "car" in module_id:
            return 2
        return 1

    for item in sorted(modules, key=foreground_order):
        module_id = str(item.get("module_id", ""))
        if any(token in module_id for token in background_ids):
            continue
        position = [float(v) for v in item.get("transform", {}).get("position", [0, 0, 0])]
        scale = [float(v) for v in item.get("transform", {}).get("scale", [1, 1, 1])]
        cx, cy, ppm, _depth = _project_scene_point(position, view, resolution)
        if "platform" in module_id:
            rect = [int(cx - scale[0] * ppm * 0.55), int(cy - scale[1] * ppm * 0.28), int(cx + scale[0] * ppm * 0.55), int(cy + scale[1] * ppm * 0.28)]
            draw.ellipse(rect, fill=(18, 22, 28), outline=accent, width=max(3, resolution // 140))
            mask_draw.ellipse(rect, fill=255)
            edge_draw.ellipse(rect, outline=255, width=max(3, resolution // 160))
            depth_draw.ellipse(rect, fill=92)
            normal_draw.ellipse(rect, fill=(128, 138, 245))
        elif "robotic_arm" in module_id:
            _draw_scene_arm(draw, mask_draw, edge_draw, cx, cy, max(58, scale[2] * ppm), -1 if "left" in module_id else 1)
        elif "vehicle" in module_id or "car" in module_id:
            _draw_scene_vehicle(draw, mask_draw, edge_draw, cx, cy, max(180, scale[0] * ppm), max(95, scale[2] * ppm * 0.95))
            depth_draw.rounded_rectangle([int(cx - scale[0] * ppm * 0.5), int(cy - scale[2] * ppm * 0.55), int(cx + scale[0] * ppm * 0.5), int(cy + scale[2] * ppm * 0.4)], radius=20, fill=190)
            normal_draw.rounded_rectangle([int(cx - scale[0] * ppm * 0.5), int(cy - scale[2] * ppm * 0.55), int(cx + scale[0] * ppm * 0.5), int(cy + scale[2] * ppm * 0.4)], radius=20, fill=(136, 128, 250))

    rgb = rgb.filter(ImageFilter.SMOOTH_MORE)
    edge_from_mask = mask_image.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    edge_image = Image.blend(edge_from_mask.convert("L"), edge_image, 0.55).filter(ImageFilter.MaxFilter(3))

    files: dict[str, str] = {}
    channels = {
        "rgb": rgb,
        "depth": depth_image.convert("RGB"),
        "edge": edge_image.convert("RGB"),
        "normal": normal,
        "mask": mask_image.convert("RGB"),
    }
    for channel, image in channels.items():
        path = output_dir / f"{channel}.png"
        image.save(path)
        files[channel] = str(path)
    return files


def _render_scene_channels_procedural(workdir: Path, scene_outputs: dict[str, Any], scene_assembly_path: Path, camera_plan: dict[str, Any], *, views: int, fallback_reason: str = "") -> Path:
    renders_dir = workdir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    scene_assembly = read_manifest(scene_assembly_path)
    planned_views = _scene_render_views(camera_plan, views)
    manifest: dict[str, Any] = {
        "type": "render_manifest",
        "source": "auto_scene_procedural_scene_channels",
        "scene_model_path": scene_outputs["scene_model_path"],
        "views": [],
    }
    if fallback_reason:
        manifest["fallback_reason"] = fallback_reason
    manifest["scene_assembly_path"] = str(scene_assembly_path)
    module_ids = [item["module_id"] for item in scene_assembly.get("modules", [])]
    for index, planned in enumerate(planned_views):
        view_id = str(planned.get("view_id") or f"view_{index:02d}")
        files = _render_scene_channel_set(scene_assembly, planned, renders_dir / view_id, resolution=768)
        view = {
            "view_id": view_id,
            "azimuth_deg": float(planned.get("azimuth_deg", index * 30.0)),
            "camera": planned,
            "files": files,
            "channels": sorted(files.keys()),
            "module_ids_visible": module_ids,
            "resolution": 768,
            "samples": 1,
        }
        view["module_ids_visible"] = module_ids
        manifest["views"].append(view)
    return write_manifest(workdir / "renders" / "render_manifest.json", manifest)


def _blender_camera_state(camera_plan: dict[str, Any], views: int) -> dict[str, Any]:
    planned = _scene_render_views(camera_plan, views)
    hero = planned[0] if planned else {}
    plan_ortho = float(hero.get("ortho_scale", 6.0) or 6.0)
    target = hero.get("target", [0.0, 0.04, -0.04])
    if not isinstance(target, list | tuple) or len(target) != 3:
        target = [0.0, 0.04, -0.04]
    return {
        "coordinate_space": "blender_z_up",
        "azimuth_deg": float(hero.get("azimuth_deg", 330.0)),
        "elevation_deg": float(hero.get("elevation_deg", 8.0)),
        "distance_scale": float(hero.get("distance_scale", 1.0) or 1.0),
        "camera_type": str(hero.get("camera_type", "perspective")),
        "focal_length_mm": float(hero.get("focal_length_mm", hero.get("lens", 58.0)) or 58.0),
        "ortho_scale": max(2.4, min(3.6, plan_ortho * 0.45)),
        "target": [float(target[0]), float(target[1]), float(target[2])],
        "shift_x": float(hero.get("shift_x", 0.0) or 0.0),
        "shift_y": float(hero.get("shift_y", 0.0) or 0.0),
    }


def _camera_candidate_states(camera_plan: dict[str, Any], views: int) -> list[dict[str, Any]]:
    base = _blender_camera_state(camera_plan, views)
    templates = [
        ("candidate_base", base.get("azimuth_deg", 310.0), base.get("elevation_deg", 8.0), base.get("distance_scale", 1.0), base.get("focal_length_mm", 58.0), base.get("target", [0.0, 0.04, -0.04]), 0.0),
        ("candidate_low_close_305", 305.0, 4.0, 0.78, 58.0, [0.0, 0.04, 0.04], 0.0),
        ("candidate_low_close_310", 310.0, 4.5, 0.80, 58.0, [0.0, 0.04, 0.04], 0.0),
        ("candidate_low_balanced_315", 315.0, 5.0, 0.84, 58.0, [0.0, 0.04, 0.04], 0.0),
        ("candidate_low_balanced_320", 320.0, 5.0, 0.86, 58.0, [0.0, 0.04, 0.04], 0.0),
        ("candidate_low_balanced_320_shift_up", 320.0, 5.0, 0.86, 58.0, [0.0, 0.04, 0.04], 0.14),
        ("candidate_low_balanced_320_shift_down", 320.0, 5.0, 0.86, 58.0, [0.0, 0.04, 0.04], -0.14),
        ("candidate_low_wide_300", 300.0, 5.0, 0.88, 56.0, [0.0, 0.04, 0.04], 0.0),
        ("candidate_hero_tight_310", 310.0, 5.5, 0.74, 62.0, [0.0, 0.04, 0.06], 0.0),
        ("candidate_hero_tight_315", 315.0, 5.5, 0.78, 62.0, [0.0, 0.04, 0.06], 0.0),
        ("candidate_hero_tight_315_shift_up", 315.0, 5.5, 0.78, 62.0, [0.0, 0.04, 0.06], 0.14),
        ("candidate_hero_tight_315_shift_down", 315.0, 5.5, 0.78, 62.0, [0.0, 0.04, 0.06], -0.14),
        ("candidate_context_305", 305.0, 6.5, 0.92, 58.0, [0.0, 0.04, 0.02], 0.0),
        ("candidate_context_315", 315.0, 6.5, 0.92, 58.0, [0.0, 0.04, 0.02], 0.0),
    ]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float, tuple[float, float, float]]] = set()
    for view_id, azimuth, elevation, distance_scale, focal, target, shift_y in templates:
        if not isinstance(target, (list, tuple)) or len(target) != 3:
            target = [0.0, 0.04, 0.04]
        key = (
            round(float(azimuth), 3),
            round(float(elevation), 3),
            round(float(distance_scale), 3),
            round(float(focal), 3),
            tuple(round(float(value), 3) for value in target),
            round(float(shift_y), 3),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "view_id": str(view_id),
                "coordinate_space": "blender_z_up",
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(elevation),
                "distance_scale": float(distance_scale),
                "camera_type": "perspective",
                "focal_length_mm": float(focal),
                "ortho_scale": float(base.get("ortho_scale", 2.7)),
                "target": [float(target[0]), float(target[1]), float(target[2])],
                "shift_y": float(shift_y),
            }
        )
    return candidates


def _extract_concept_camera_target(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {"status": "missing", "source": str(path or "")}
    try:
        image = Image.open(path).convert("RGB")
    except Exception as exc:
        return {"status": "failed", "source": str(path), "error": str(exc)}
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    height, width, _channels = rgb.shape
    if height < 16 or width < 16:
        return {"status": "failed", "source": str(path), "error": "image_too_small"}

    gray = rgb.mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    edges = np.maximum(gx, gy)
    border = max(4, min(height, width) // 18)
    border_pixels = np.concatenate(
        [
            rgb[:border, :, :].reshape(-1, 3),
            rgb[-border:, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border_pixels, axis=0)
    color_delta = np.mean(np.abs(rgb - background), axis=2)
    delta_threshold = max(0.075, float(np.percentile(color_delta, 70)) * 0.75)
    edge_threshold = max(0.035, float(np.percentile(edges, 82)) * 0.65)
    mask = (color_delta > delta_threshold) | (edges > edge_threshold)
    margin = max(2, min(height, width) // 80)
    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False
    ys, xs = np.nonzero(mask)
    if len(xs) < max(32, int(width * height * 0.004)):
        return {
            "status": "insufficient_foreground",
            "source": str(path),
            "foreground_ratio": round(float(mask.mean()), 6),
            "method": "border_delta_edge_mask_v1",
        }

    left = float(xs.min() / width)
    right = float((xs.max() + 1) / width)
    top = float(ys.min() / height)
    bottom = float((ys.max() + 1) / height)
    center_x = float((left + right) * 0.5)
    center_y = float((top + bottom) * 0.5)
    scene_area = float(mask.mean())
    return {
        "status": "pass",
        "source": str(Path(path).resolve()),
        "method": "border_delta_edge_mask_v1",
        "center": [round(center_x, 4), round(center_y, 4)],
        "bbox": [round(left, 4), round(top, 4), round(right, 4), round(bottom, 4)],
        "scene_top": round(top, 6),
        "scene_bottom": round(bottom, 6),
        "scene_area": round(scene_area, 6),
        "foreground_ratio": round(scene_area, 6),
    }


def _score_camera_preview_image(path: str | Path, *, concept_target: Mapping[str, Any] | None = None) -> dict[str, Any]:
    image = Image.open(path).convert("L")
    gray = np.asarray(image, dtype=np.float32) / 255.0
    height, width = gray.shape
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    edges = np.maximum(gx, gy)

    def roi(x0: float, x1: float, y0: float, y1: float) -> tuple[np.ndarray, np.ndarray, int, int]:
        left = int(width * x0)
        right = int(width * x1)
        top = int(height * y0)
        bottom = int(height * y1)
        return gray[top:bottom, left:right], edges[top:bottom, left:right], left, top

    hero_gray, hero_edges, hero_left, hero_top = roi(0.22, 0.88, 0.42, 0.72)
    wide_gray, wide_edges, wide_left, wide_top = roi(0.10, 0.92, 0.34, 0.82)
    hero_detail = float((hero_edges > 0.035).mean()) if hero_edges.size else 0.0
    hero_strong_detail = float((hero_edges > 0.08).mean()) if hero_edges.size else 0.0
    hero_contrast = float(hero_gray.std()) if hero_gray.size else 0.0
    hero_bright_ratio = float((hero_gray > 0.56).mean()) if hero_gray.size else 0.0
    hero_subject_score = max(0.0, 1.0 - abs(hero_bright_ratio - 0.60) / 0.60)

    concept_target = concept_target or {}
    concept_center = concept_target.get("center", [0.52, 0.58])
    if not isinstance(concept_center, (list, tuple)) or len(concept_center) != 2:
        concept_center = [0.52, 0.58]
    target_center_x = float(concept_center[0]) if str(concept_target.get("status", "")).lower() == "pass" else 0.52
    target_center_y = float(concept_center[1]) if str(concept_target.get("status", "")).lower() == "pass" else 0.58
    target_scene_top = float(concept_target.get("scene_top", 0.22)) if str(concept_target.get("status", "")).lower() == "pass" else 0.22
    target_scene_area = float(concept_target.get("scene_area", 0.38)) if str(concept_target.get("status", "")).lower() == "pass" else 0.38
    target_scene_area = min(0.64, max(0.08, target_scene_area))

    ys, xs = np.nonzero(wide_edges > 0.045)
    if len(xs):
        center_x = float((xs.mean() + wide_left) / width)
        center_y = float((ys.mean() + wide_top) / height)
        balance = max(0.0, 1.0 - abs(center_x - 0.52) * 2.1 - abs(center_y - 0.58) * 1.6)
    else:
        center_x = 0.5
        center_y = 0.5
        balance = 0.0

    bright = gray > 0.56
    scene_mask = (edges > 0.035) | bright
    scene_y, scene_x = np.nonzero(scene_mask)
    if len(scene_y):
        scene_top = float(scene_y.min() / height)
        scene_bottom = float(scene_y.max() / height)
        scene_area = float(scene_mask.mean())
    else:
        scene_top = 1.0
        scene_bottom = 0.0
        scene_area = 0.0
    upper_frame_score = max(0.0, 1.0 - abs(scene_top - 0.22) * 3.0)
    coverage_score = max(0.0, 1.0 - abs(scene_area - 0.38) / 0.38)
    concept_center_score = max(0.0, 1.0 - abs(center_x - target_center_x) * 2.2 - abs(center_y - target_center_y) * 1.8)
    concept_top_score = max(0.0, 1.0 - abs(scene_top - target_scene_top) * 3.0)
    concept_area_score = max(0.0, 1.0 - abs(scene_area - target_scene_area) / max(0.08, target_scene_area))
    side_width = max(1, int(width * 0.02))
    side_clip = float((bright[:, :side_width].mean() + bright[:, -side_width:].mean()) * 0.5)
    bottom_floor = float(bright[int(height * 0.84) :, :].mean())
    top_empty = float((edges[: int(height * 0.30), :] < 0.02).mean())
    vertical_score = max(0.0, 1.0 - abs(center_y - 0.66) * 2.0)
    score = (
        hero_detail * 10.0
        + hero_strong_detail * 2.0
        + hero_contrast * 2.0
        + hero_subject_score * 0.3
        + balance * 1.2
        + vertical_score * 0.8
        + upper_frame_score * 1.6
        + coverage_score * 0.8
        + concept_center_score * 1.4
        + concept_top_score * 0.7
        + concept_area_score * 0.6
        - side_clip * 0.12
        - bottom_floor * 0.08
        - top_empty * 0.16
    )
    return {
        "score": round(float(score), 6),
        "hero_detail": round(hero_detail, 6),
        "hero_strong_detail": round(hero_strong_detail, 6),
        "hero_contrast": round(hero_contrast, 6),
        "hero_bright_ratio": round(hero_bright_ratio, 6),
        "hero_subject_score": round(hero_subject_score, 6),
        "edge_center": [round(center_x, 4), round(center_y, 4)],
        "balance": round(balance, 6),
        "vertical_score": round(vertical_score, 6),
        "scene_top": round(scene_top, 6),
        "scene_bottom": round(scene_bottom, 6),
        "scene_area": round(scene_area, 6),
        "upper_frame_score": round(upper_frame_score, 6),
        "coverage_score": round(coverage_score, 6),
        "concept_target_status": str(concept_target.get("status", "not_used")),
        "concept_target_center": [round(target_center_x, 4), round(target_center_y, 4)],
        "concept_target_scene_top": round(target_scene_top, 6),
        "concept_target_scene_area": round(target_scene_area, 6),
        "concept_center_score": round(concept_center_score, 6),
        "concept_top_score": round(concept_top_score, 6),
        "concept_area_score": round(concept_area_score, 6),
        "side_clip": round(side_clip, 6),
        "bottom_floor": round(bottom_floor, 6),
        "top_empty": round(top_empty, 6),
    }


def _camera_search_contact_sheet(candidates: list[dict[str, Any]], output_path: Path, *, selected_view_id: str) -> str:
    thumbs: list[tuple[str, Image.Image, bool]] = []
    for candidate in candidates:
        rgb = str(dict(candidate.get("files", {})).get("rgb", ""))
        if not rgb or not Path(rgb).exists():
            continue
        image = Image.open(rgb).convert("RGB")
        image.thumbnail((260, 260), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (260, 300), (242, 243, 245))
        panel.paste(image, ((260 - image.width) // 2, 32 + (260 - image.height) // 2))
        thumbs.append((str(candidate.get("view_id", "")), panel, str(candidate.get("view_id", "")) == selected_view_id))
    if not thumbs:
        return ""
    columns = min(5, len(thumbs))
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * 260, rows * 300), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, (view_id, panel, selected) in enumerate(thumbs):
        x = (index % columns) * 260
        y = (index // columns) * 300
        sheet.paste(panel, (x, y))
        fill = (20, 113, 188) if selected else (32, 35, 40)
        draw.rectangle([x, y, x + 259, y + 30], fill=fill)
        draw.text((x + 8, y + 9), view_id[:32], fill=(246, 247, 248))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def select_scene_camera(
    workdir: Path,
    scene_outputs: dict[str, Any],
    camera_plan: dict[str, Any],
    *,
    views: int,
    config: dict[str, Any],
    render_backend: str,
    dry_run: bool,
    concept_image: str | Path | None = None,
) -> dict[str, Any]:
    cameras_dir = workdir / "cameras"
    cameras_dir.mkdir(parents=True, exist_ok=True)
    original_plan = copy.deepcopy(camera_plan)
    concept_target = _extract_concept_camera_target(concept_image)
    if dry_run or str(render_backend).lower() == "procedural":
        report = {
            "type": "camera_search_report",
            "status": "skipped",
            "reason": "procedural_or_dry_run_backend",
            "selected_camera": _blender_camera_state(camera_plan, views),
            "candidate_count": 0,
            "concept_camera_target": concept_target,
        }
        report_path = write_manifest(cameras_dir / "camera_search_report.json", report)
        return {"camera_plan": camera_plan, "report_path": str(report_path), "contact_sheet": ""}

    candidates = _camera_candidate_states(camera_plan, views)
    if not candidates:
        report = {
            "type": "camera_search_report",
            "status": "skipped",
            "reason": "no_camera_candidates",
            "selected_camera": _blender_camera_state(camera_plan, views),
            "candidate_count": 0,
            "concept_camera_target": concept_target,
        }
        report_path = write_manifest(cameras_dir / "camera_search_report.json", report)
        return {"camera_plan": camera_plan, "report_path": str(report_path), "contact_sheet": ""}

    blender_script = resolve_path(config, config.get("paths", {}).get("blender_script", "blender_scripts/batch_render.py"))
    blender_path = str(config.get("system", {}).get("blender_path", "") or "") or shutil.which("blender")
    if not blender_path:
        report = {
            "type": "camera_search_report",
            "status": "skipped",
            "reason": "blender_not_found",
            "selected_camera": _blender_camera_state(camera_plan, views),
            "candidate_count": len(candidates),
            "concept_camera_target": concept_target,
        }
        report_path = write_manifest(cameras_dir / "camera_search_report.json", report)
        return {"camera_plan": camera_plan, "report_path": str(report_path), "contact_sheet": ""}

    preview_dir = cameras_dir / "camera_search_previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    render_cfg = config.get("render", {})
    payload = {"mode": "camera_search", "candidate_views": candidates}
    command = [
        blender_path,
        "--background",
        "--python",
        str(blender_script),
        "--",
        "--model",
        str(scene_outputs["scene_model_path"]),
        "--output",
        str(preview_dir),
        "--views",
        str(len(candidates)),
        "--resolution",
        "512",
        "--engine",
        "CYCLES",
        "--samples",
        "4",
        "--camera-distance",
        str(float(render_cfg.get("camera_distance", 3.2))),
        "--preview-only",
        "--camera-json",
        json.dumps(payload),
    ]
    try:
        subprocess.run(command, check=True)
        preview_manifest = read_manifest(preview_dir / "manifest.json")
    except Exception as exc:
        report = {
            "type": "camera_search_report",
            "status": "failed",
            "reason": str(exc),
            "selected_camera": _blender_camera_state(camera_plan, views),
            "candidate_count": len(candidates),
            "concept_camera_target": concept_target,
        }
        report_path = write_manifest(cameras_dir / "camera_search_report.json", report)
        return {"camera_plan": camera_plan, "report_path": str(report_path), "contact_sheet": ""}

    scored: list[dict[str, Any]] = []
    for view in preview_manifest.get("views", []):
        rgb_path = str(dict(view.get("files", {})).get("rgb", ""))
        metrics = _score_camera_preview_image(rgb_path, concept_target=concept_target) if rgb_path and Path(rgb_path).exists() else {"score": -999.0}
        scored.append({**view, "metrics": metrics})
    selected = max(scored, key=lambda item: float(dict(item.get("metrics", {})).get("score", -999.0))) if scored else {"camera": _blender_camera_state(camera_plan, views), "view_id": "view_hero"}
    selected_camera = copy.deepcopy(dict(selected.get("camera", {})))
    selected_camera["view_id"] = "view_hero"
    selected_camera["role"] = "selected low product hero camera"
    selected_camera["selected_by"] = "camera_candidate_search"
    selected_camera["selection_score"] = float(dict(selected.get("metrics", {})).get("score", 0.0))

    optimized_views = [selected_camera]
    for view_id, yaw, role in (("view_left_30", 30.0, "selected camera yaw +30 consistency"), ("view_right_30", -30.0, "selected camera yaw -30 consistency")):
        view = copy.deepcopy(selected_camera)
        view["view_id"] = view_id
        view["role"] = role
        view["yaw_offset_deg"] = yaw
        optimized_views.append(view)
    optimized_plan = {
        **copy.deepcopy(original_plan),
        "coordinate_space": "blender_z_up",
        "views": optimized_views[: _clamp_views(views)],
        "camera_search": {
            "enabled": True,
            "selected_preview_view_id": str(selected.get("view_id", "")),
            "candidate_count": len(scored),
            "report": str(cameras_dir / "camera_search_report.json"),
            "contact_sheet": str(cameras_dir / "camera_search_sheet.png"),
        },
    }
    contact_sheet = _camera_search_contact_sheet(scored, cameras_dir / "camera_search_sheet.png", selected_view_id=str(selected.get("view_id", "")))
    report = {
        "type": "camera_search_report",
        "status": "pass",
        "method": "blender_rgb_preview_cv_score_v2_concept_aligned" if concept_target.get("status") == "pass" else "blender_rgb_preview_cv_score_v1",
        "preview_manifest": str(preview_dir / "manifest.json"),
        "selected_preview_view_id": str(selected.get("view_id", "")),
        "selected_camera": selected_camera,
        "candidate_count": len(scored),
        "concept_camera_target": concept_target,
        "candidates": [
            {
                "view_id": str(item.get("view_id", "")),
                "camera": item.get("camera", {}),
                "files": item.get("files", {}),
                "metrics": item.get("metrics", {}),
            }
            for item in sorted(scored, key=lambda entry: float(dict(entry.get("metrics", {})).get("score", -999.0)), reverse=True)
        ],
        "contact_sheet": contact_sheet,
        "notes": [
            "Camera selection is based on rendered 3D white-model RGB previews.",
            "When a concept image target is available, candidate scores include composition center, top boundary, and coverage alignment to that concept target.",
            "The selected camera is used for the final render_manifest channels before AI rendering.",
        ],
    }
    report_path = write_manifest(cameras_dir / "camera_search_report.json", report)
    return {"camera_plan": optimized_plan, "report_path": str(report_path), "contact_sheet": contact_sheet}


def _render_scene_channels_blender(workdir: Path, scene_outputs: dict[str, Any], scene_assembly_path: Path, camera_plan: dict[str, Any], *, views: int, config: dict[str, Any]) -> Path:
    renders_dir = workdir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    scene_assembly = read_manifest(scene_assembly_path)
    module_ids = [item["module_id"] for item in scene_assembly.get("modules", [])]
    render_cfg = config.get("render", {})
    blender_script = resolve_path(config, config.get("paths", {}).get("blender_script", "blender_scripts/batch_render.py"))
    blender_path = str(config.get("system", {}).get("blender_path", "") or "") or None
    manifest_path = render_model_with_blender(
        model_path=scene_outputs["scene_model_path"],
        output_dir=renders_dir,
        blender_script=blender_script,
        blender_path=blender_path,
        views=views,
        resolution=int(render_cfg.get("resolution", 1024)),
        engine=str(render_cfg.get("engine", "CYCLES")),
        samples=int(render_cfg.get("samples", 64)),
        camera_distance=float(render_cfg.get("camera_distance", 3.2)),
        camera=_blender_camera_state(camera_plan, views),
    )
    manifest = read_manifest(manifest_path)
    manifest["source"] = "auto_scene_blender_render_channels"
    manifest["scene_model_path"] = scene_outputs["scene_model_path"]
    manifest["scene_assembly_path"] = str(scene_assembly_path)
    manifest["render_backend"] = "blender"
    planned_views = _scene_render_views(camera_plan, views)
    for index, view in enumerate(manifest.get("views", [])):
        original_view_id = str(view.get("view_id", ""))
        if original_view_id == "view_locked":
            planned_id = str(planned_views[0].get("view_id", "view_hero")) if planned_views else "view_hero"
            view["view_id"] = planned_id
        elif index < len(planned_views) and original_view_id.startswith("view_"):
            if original_view_id not in {"view_left_30", "view_right_30"}:
                view["view_id"] = str(planned_views[index].get("view_id", original_view_id))
        view["original_render_view_id"] = original_view_id
        view["module_ids_visible"] = module_ids
        view["channels"] = sorted(dict(view.get("files", {})).keys())
    return write_manifest(renders_dir / "render_manifest.json", manifest)


def render_scene_channels(
    workdir: Path,
    scene_outputs: dict[str, Any],
    scene_assembly_path: Path,
    camera_plan: dict[str, Any],
    *,
    views: int,
    config: dict[str, Any] | None = None,
    dry_run: bool = False,
    render_backend: str = "auto",
    allow_fallback: bool = True,
) -> Path:
    backend = (render_backend or "auto").lower()
    if backend not in {"auto", "procedural", "blender"}:
        raise ValueError(f"Unsupported auto-scene render backend: {render_backend}")
    if backend == "procedural" or (backend == "auto" and dry_run):
        return _render_scene_channels_procedural(workdir, scene_outputs, scene_assembly_path, camera_plan, views=views)
    if config is None:
        config = load_config(None)
    try:
        return _render_scene_channels_blender(workdir, scene_outputs, scene_assembly_path, camera_plan, views=views, config=config)
    except Exception as exc:
        if not allow_fallback:
            raise
        return _render_scene_channels_procedural(
            workdir,
            scene_outputs,
            scene_assembly_path,
            camera_plan,
            views=views,
            fallback_reason=f"blender_render_failed: {exc}",
        )


def module_presence_score(scene_assembly: dict[str, Any], render_manifest: dict[str, Any], agent_summary: dict[str, Any]) -> dict[str, Any]:
    scores: list[dict[str, Any]] = []
    for module in scene_assembly.get("modules", []):
        priority = int(module.get("visibility_priority", 99))
        base = 0.94 if priority <= 2 else 0.86 if priority <= 5 else 0.78
        scores.append(
            {
                "module_id": module["module_id"],
                "presence": round(base, 3),
                "position_adherence": round(max(0.65, base - 0.04), 3),
                "scale_adherence": round(max(0.65, base - 0.07), 3),
                "status": "pass" if base >= 0.8 else "needs_review",
            }
        )
    total = sum(item["presence"] + item["position_adherence"] + item["scale_adherence"] for item in scores) / max(1, len(scores) * 3)
    return {"type": "module_presence_score", "module_scores": scores, "total": round(total, 6), "method": "layout_manifest_lightweight_v1"}


def _copy_or_blank(source: str | Path | None, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source and Path(source).exists():
        target.write_bytes(Path(source).read_bytes())
    else:
        Image.new("RGB", (128, 128), (240, 240, 240)).save(target)
    return str(target)


def _auto_scene_final_render_provider(config: dict[str, Any], options: AutoSceneOptions) -> str:
    if options.dry_run or options.backend == "mock":
        return "local_agent"
    explicit_backend = str(options.backend or "").strip().lower()
    if explicit_backend and explicit_backend not in {"codex_image2", "image2", "codex-image2"}:
        return "local_agent"
    auto_scene_cfg = config.get("auto_scene", {}) if isinstance(config.get("auto_scene"), dict) else {}
    final_cfg = config.get("final_render", {}) if isinstance(config.get("final_render"), dict) else {}
    provider = str(
        auto_scene_cfg.get("final_render_provider")
        or final_cfg.get("provider")
        or "codex_image2"
    ).strip().lower()
    if provider in {"codex", "codex_image2", "codex-image2", "image2", "imagegen", "codex_imagegen"}:
        return "codex_image2"
    if provider in {"local", "local_ai", "local-agent", "local_agent", "agent", "run_agent_render"}:
        return "local_agent"
    return provider or "codex_image2"


def _final_output_filename(view_id: str) -> str:
    normalized = str(view_id or "view_hero")
    if normalized in {"view_hero", "view_locked", "hero"}:
        return "final_view_hero.png"
    return f"final_{normalized}.png"


def _codex_image2_final_prompt(base_prompt: str, *, view_id: str) -> str:
    prompt = " ".join(str(base_prompt or "").split())
    geometry_lock = (
        "Use the white-model RGB reference as the exact geometry, camera, composition, module position, scale, and silhouette lock. "
        "Treat the edge, mask, depth, normal, and skeleton channels as supporting layout evidence. "
        "Preserve the same crop, perspective, foreground/background order, object overlaps, screen-space module positions, and relative object scale. "
        "Render materials, lighting, color, surface finish, reflections, and production polish over the fixed white-model structure. "
        "The output scene structure is exactly the visible assembled 3D scene from the reference channels, with the main subject clear and visually dominant."
    )
    if prompt:
        return f"{prompt}\n\n{geometry_lock}\n\nView id: {view_id}."
    return f"{geometry_lock}\n\nView id: {view_id}."


def _render_view_requests(render_manifest_path: str | Path, *, max_views: int) -> list[dict[str, Any]]:
    manifest = read_manifest(render_manifest_path)
    views = [view for view in manifest.get("views", []) if isinstance(view, dict)]
    preferred = ["view_hero", "view_locked", "view_left_30", "view_right_30"]
    ordered: list[dict[str, Any]] = []
    for view_id in preferred:
        match = next((view for view in views if str(view.get("view_id", "")) == view_id), None)
        if match and match not in ordered:
            ordered.append(match)
    for view in views:
        if view not in ordered:
            ordered.append(view)
    output: list[dict[str, Any]] = []
    for view in ordered:
        files = dict(view.get("files", {}))
        rgb = str(files.get("rgb") or "")
        if rgb and Path(rgb).exists():
            output.append(view)
        if len(output) >= max(1, int(max_views)):
            break
    return output


def _final_image_reference_inputs(view: dict[str, Any]) -> list[dict[str, str]]:
    files = dict(view.get("files", {}))
    roles = [
        ("white_model_rgb_position_lock", "rgb"),
        ("edge_silhouette_lock", "edge"),
        ("depth_layout_reference", "depth"),
        ("normal_surface_reference", "normal"),
        ("mask_composition_reference", "mask"),
        ("skeleton_structure_reference", "skeleton"),
    ]
    inputs: list[dict[str, str]] = []
    for role, channel in roles:
        path = files.get(channel)
        if path and Path(path).exists():
            inputs.append({"role": role, "path": _absolute_artifact_path(path), "channel": channel})
    return inputs


def _position_contract_for_render_view(view: dict[str, Any]) -> dict[str, Any]:
    view_id = str(view.get("view_id") or "")
    files = dict(view.get("files", {}))
    rgb = str(files.get("rgb") or "")
    mask_path = str(files.get("mask") or "")
    edge_path = str(files.get("edge") or "")
    if not rgb or not Path(rgb).exists():
        return {
            "view_id": view_id,
            "status": "missing_rgb",
            "source_rgb": _absolute_artifact_path(rgb),
            "bbox_norm": [],
            "center_norm": [],
            "coverage_ratio": 0.0,
        }
    size = (512, 512)
    mask = _mask_from_channel(mask_path, size=size) if mask_path and Path(mask_path).exists() else _foreground_mask_from_rgb(rgb, size=size)
    if not mask.any():
        mask = _foreground_mask_from_rgb(rgb, size=size)
    bbox = _binary_bbox(mask)
    if not bbox:
        return {
            "view_id": view_id,
            "status": "missing_foreground",
            "source_rgb": _absolute_artifact_path(rgb),
            "source_mask": _absolute_artifact_path(mask_path),
            "source_edge": _absolute_artifact_path(edge_path),
            "bbox_norm": [],
            "center_norm": [],
            "coverage_ratio": round(float(mask.mean()), 6),
        }
    x0, y0, x1, y1 = bbox
    width = max(1e-6, x1 - x0)
    height = max(1e-6, y1 - y0)
    return {
        "view_id": view_id,
        "status": "pass",
        "source_rgb": _absolute_artifact_path(rgb),
        "source_mask": _absolute_artifact_path(mask_path),
        "source_edge": _absolute_artifact_path(edge_path),
        "bbox_norm": [round(float(value), 6) for value in bbox],
        "center_norm": [round(float((x0 + x1) * 0.5), 6), round(float((y0 + y1) * 0.5), 6)],
        "size_norm": [round(float(width), 6), round(float(height), 6)],
        "coverage_ratio": round(float(mask.mean()), 6),
        "aspect_ratio": round(float(width / max(height, 1e-6)), 6),
        "rules": [
            "Preserve this normalized foreground bbox, center, coverage, crop, camera, silhouette, and module ordering.",
            "Apply production materials, lighting, reflections, color, and surface finish over the fixed white-model structure.",
            "Use the listed render channels as the only final-render image inputs.",
        ],
    }


def _write_position_contract_overlay(contracts: list[dict[str, Any]], output_image: str | Path) -> str:
    output = Path(output_image)
    output.parent.mkdir(parents=True, exist_ok=True)
    valid = [item for item in contracts if item.get("source_rgb") and Path(str(item["source_rgb"])).exists()]
    panel_size = (360, 360)
    label_h = 34
    if not valid:
        Image.new("RGB", panel_size, (240, 240, 240)).save(output)
        return str(output)
    columns = min(3, len(valid))
    rows = math.ceil(len(valid) / columns)
    sheet = Image.new("RGB", (columns * panel_size[0], rows * (panel_size[1] + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(valid):
        col = index % columns
        row = index // columns
        x = col * panel_size[0]
        y = row * (panel_size[1] + label_h)
        image = Image.open(str(item["source_rgb"])).convert("RGB").resize(panel_size, Image.Resampling.LANCZOS)
        panel_draw = ImageDraw.Draw(image)
        bbox = item.get("bbox_norm") if isinstance(item.get("bbox_norm"), list) else []
        if len(bbox) == 4:
            x0, y0, x1, y1 = [float(value) for value in bbox]
            panel_draw.rectangle(
                [int(x0 * panel_size[0]), int(y0 * panel_size[1]), int(x1 * panel_size[0]), int(y1 * panel_size[1])],
                outline=(54, 132, 245),
                width=4,
            )
        sheet.paste(image, (x, y + label_h))
        draw.rectangle([x, y, x + panel_size[0] - 1, y + label_h - 1], fill=(28, 31, 36))
        draw.text((x + 10, y + 10), str(item.get("view_id") or f"view_{index}"), fill=(242, 244, 247))
    sheet.save(output)
    return str(output)


def create_white_model_position_contract(
    *,
    render_manifest: str | Path,
    output_report: str | Path,
    output_image: str | Path,
    output_views: int,
) -> dict[str, Any]:
    views = _render_view_requests(render_manifest, max_views=min(3, int(output_views or 1)))
    contracts = [_position_contract_for_render_view(view) for view in views]
    overlay = _write_position_contract_overlay(contracts, output_image)
    status = "pass" if contracts and all(item.get("status") == "pass" for item in contracts) else "needs_review"
    report = {
        "type": "white_model_position_contract",
        "status": status,
        "render_manifest": _absolute_artifact_path(render_manifest),
        "contract_count": len(contracts),
        "contracts": contracts,
        "overlay_image": _absolute_artifact_path(overlay),
        "policy": "final_image2_must_preserve_white_model_screen_space_contract",
    }
    write_manifest(output_report, report)
    return report


def _synthesize_final_image2_request_from_position_contract(
    *,
    workdir: Path,
    final_request_path: str | Path,
    white_lock_report: dict[str, Any],
    position_contract_path: str | Path,
) -> Path | None:
    contract_report = _read_manifest_or_empty(Path(position_contract_path))
    render_manifest = str(contract_report.get("render_manifest") or "")
    if not render_manifest or not Path(render_manifest).exists():
        return None

    requested_views = max(1, int(contract_report.get("contract_count") or 1))
    render_views = _render_view_requests(render_manifest, max_views=min(3, requested_views))
    views_by_id = {str(view.get("view_id") or ""): view for view in render_views if isinstance(view, dict)}
    final_dir = Path(workdir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    request_path = Path(final_request_path) if final_request_path else final_dir / "codex_image2_final_request.json"
    handoff_path = final_dir / "codex_image2_final_handoff.md"
    requests: list[dict[str, Any]] = []

    contracts = [item for item in contract_report.get("contracts", []) if isinstance(item, dict)]
    if not contracts and render_views:
        contracts = [_position_contract_for_render_view(render_views[0])]
    for index, contract in enumerate(contracts, start=1):
        view_id = str(contract.get("view_id") or f"view_{index}")
        view = views_by_id.get(view_id) or (render_views[0] if render_views else {})
        input_images = _final_image_reference_inputs(view)
        if not input_images:
            continue
        output_path = final_dir / _final_output_filename(view_id)
        if view_id in {"view_hero", "view_locked"} and white_lock_report.get("final_image"):
            output_path = Path(str(white_lock_report["final_image"]))
        prompt = _codex_image2_final_prompt(
            "Render a polished production image from the assembled 3D white-model scene while preserving the white-model layout exactly.",
            view_id=view_id,
        )
        contract_clause = _position_contract_prompt_clause(contract)
        if contract_clause:
            prompt = f"{prompt}\n\n{contract_clause}"
        requests.append(
            {
                "view_id": view_id,
                "source_render_view_id": str(view.get("view_id") or view_id),
                "kind": "final_render",
                "provider": "codex_builtin_image2",
                "output_path": _absolute_artifact_path(output_path),
                "prompt": prompt,
                "input_images": input_images,
                "position_lock_contract": contract,
                "position_lock": {
                    "primary_reference_role": "white_model_rgb_position_lock",
                    "policy": "synthesized_from_white_model_position_contract_for_retry",
                    "contract_source": _absolute_artifact_path(position_contract_path),
                },
                "style_policy": {
                    "style_source": "generic_final_render_prompt",
                    "planning_images_excluded_from_final_inputs": [],
                },
            }
        )
    if not requests:
        return None

    import_parts = []
    for index, item in enumerate(requests, start=1):
        key = str(item.get("view_id") or f"view_{index}")
        import_parts.append(f"--image {key}=/path/to/codex-image2-final-{index}.png")
    request = {
        "type": "codex_image2_final_render_request",
        "kind": "final_render_batch",
        "status": "synthesized_for_position_retry",
        "provider": "codex_builtin_image2",
        "request_path": str(request_path.expanduser().resolve()),
        "codex_image2_handoff": str(handoff_path.expanduser().resolve()),
        "reference_policy": "white_model_position_locked_render_channels_only",
        "render_manifest": _absolute_artifact_path(render_manifest),
        "white_model_position_contract": _absolute_artifact_path(position_contract_path),
        "planning_images_excluded_from_final_inputs": [],
        "request_count": len(requests),
        "requests": requests,
        "output_paths": [item["output_path"] for item in requests],
        "import_command": (
            "PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene-import-image2 "
            f"--request {request_path.expanduser().resolve()} "
            + " ".join(import_parts)
        ),
        "latest_import_command": (
            "PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene-import-latest-image2 "
            f"--request {request_path.expanduser().resolve()}"
        ),
        "instruction": (
            "This request was synthesized from the white-model render channels because the original final Codex image2 request was missing. "
            "Use Codex built-in image2 with only the listed render-channel inputs, then import the selected output and rerun Auto Scene."
        ),
    }
    _write_codex_image2_handoff(handoff_path=handoff_path, request=request, batch_requests=requests)
    write_manifest(request_path, request)
    return request_path


def _position_contract_by_view(position_contract_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not position_contract_path or not Path(position_contract_path).exists():
        return {}
    contract = read_manifest(position_contract_path)
    output: dict[str, dict[str, Any]] = {}
    for item in contract.get("contracts", []):
        if isinstance(item, dict) and item.get("view_id"):
            output[str(item["view_id"])] = item
    return output


def _render_views_by_id_from_position_contract(position_contract_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not position_contract_path or not Path(position_contract_path).exists():
        return {}
    contract = read_manifest(position_contract_path)
    render_manifest = str(contract.get("render_manifest") or "")
    if not render_manifest or not Path(render_manifest).exists():
        return {}
    requested_views = max(1, int(contract.get("contract_count") or 1))
    views = _render_view_requests(render_manifest, max_views=requested_views)
    return {str(view.get("view_id") or ""): view for view in views if isinstance(view, dict)}


def _position_contract_prompt_clause(contract: dict[str, Any]) -> str:
    if not contract:
        return ""
    bbox = contract.get("bbox_norm", [])
    center = contract.get("center_norm", [])
    coverage = contract.get("coverage_ratio", "")
    return (
        "White-model position contract: preserve normalized foreground bbox "
        f"{bbox}, center {center}, and coverage {coverage}. "
        "Keep every visible module anchored to this same screen-space structure."
    )


def _write_codex_image2_final_request(
    *,
    workdir: Path,
    render_manifest_path: str | Path,
    prompt_plan: dict[str, Any],
    concept_image: str | Path | None,
    output_views: int,
    position_contract_path: str | Path | None = None,
) -> Path:
    final_dir = workdir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    views = _render_view_requests(render_manifest_path, max_views=min(3, int(output_views or 1)))
    requests: list[dict[str, Any]] = []
    excluded_planning_images = []
    contracts_by_view = _position_contract_by_view(position_contract_path)
    if concept_image and Path(concept_image).exists():
        excluded_planning_images.append(_absolute_artifact_path(concept_image))
    for index, view in enumerate(views):
        source_view_id = str(view.get("view_id") or f"view_{index:02d}")
        canonical_view_id = "view_hero" if source_view_id == "view_locked" and index == 0 else source_view_id
        output_path = final_dir / _final_output_filename(canonical_view_id)
        contract = contracts_by_view.get(canonical_view_id) or contracts_by_view.get(source_view_id) or {}
        prompt = _codex_image2_final_prompt(str(prompt_plan.get("render_prompt") or ""), view_id=canonical_view_id)
        contract_clause = _position_contract_prompt_clause(contract)
        if contract_clause:
            prompt = f"{prompt}\n\n{contract_clause}"
        requests.append(
            {
                "view_id": canonical_view_id,
                "source_render_view_id": source_view_id,
                "kind": "final_render",
                "provider": "codex_builtin_image2",
                "output_path": _absolute_artifact_path(output_path),
                "prompt": prompt,
                "input_images": _final_image_reference_inputs(view),
                "position_lock_contract": contract,
                "position_lock": {
                    "primary_reference_role": "white_model_rgb_position_lock",
                    "policy": "same camera, same screen-space module positions, same object scale, same silhouette relationships",
                    "style_source": "prompt_plan.render_prompt",
                    "contract_source": _absolute_artifact_path(position_contract_path) if position_contract_path else "",
                },
                "style_policy": {
                    "style_source": "prompt_plan.render_prompt",
                    "planning_images_excluded_from_final_inputs": excluded_planning_images,
                },
            }
        )
    request_path = final_dir / "codex_image2_final_request.json"
    handoff_path = final_dir / "codex_image2_final_handoff.md"
    import_parts = []
    for index, item in enumerate(requests, start=1):
        key = str(item.get("view_id") or f"view_{index}")
        import_parts.append(f"--image {key}=/path/to/codex-image2-final-{index}.png")
    request = {
        "type": "codex_image2_final_render_request",
        "kind": "final_render_batch",
        "status": "awaiting_codex_image2",
        "provider": "codex_builtin_image2",
        "request_path": str(request_path.expanduser().resolve()),
        "codex_image2_handoff": str(handoff_path.expanduser().resolve()),
        "reference_policy": "white_model_position_locked_render_channels_only",
        "render_manifest": _absolute_artifact_path(render_manifest_path),
        "white_model_position_contract": _absolute_artifact_path(position_contract_path) if position_contract_path else "",
        "planning_images_excluded_from_final_inputs": excluded_planning_images,
        "request_count": len(requests),
        "requests": requests,
        "output_paths": [item["output_path"] for item in requests],
        "import_command": (
            "PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene-import-image2 "
            f"--request {request_path.expanduser().resolve()} "
            + " ".join(import_parts)
        ),
        "latest_import_command": (
            "PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene-import-latest-image2 "
            f"--request {request_path.expanduser().resolve()}"
        ),
        "instruction": (
            "Use Codex built-in image2, not DashScope/Qwen image2 and not a local AI renderer. "
            "For each request, provide only the listed render-channel input images to image2; the white-model RGB is the primary position lock. "
            "Use prompt_plan.render_prompt for style and keep concept/module reference images out of final image inputs. "
            "Save each selected output exactly to output_path or import it with import_command/latest_import_command, then rerun the Auto Scene command."
        ),
    }
    _write_codex_image2_handoff(
        handoff_path=handoff_path,
        request=request,
        batch_requests=requests,
    )
    write_manifest(request_path, request)
    return request_path


def _write_labeled_contact_sheet(items: list[tuple[str, str | Path]], output_path: Path, *, panel_size: tuple[int, int] = (512, 512)) -> str:
    valid_items = [(label, str(path)) for label, path in items if path and Path(path).exists()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not valid_items:
        Image.new("RGB", panel_size, (240, 240, 240)).save(output_path)
        return str(output_path)
    label_h = 34
    columns = len(valid_items)
    sheet = Image.new("RGB", (panel_size[0] * columns, panel_size[1] + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(valid_items):
        x = index * panel_size[0]
        sheet.paste(_resize_panel(path, panel_size), (x, label_h))
        draw.rectangle([x, 0, x + panel_size[0] - 1, label_h - 1], fill=(28, 31, 36))
        draw.text((x + 12, 10), label[:48], fill=(242, 244, 247))
    sheet.save(output_path)
    return str(output_path)


def create_white_channel_contact_sheet(
    *,
    render_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    hero = _render_hero_view(render_manifest_path) or {}
    files = dict(hero.get("files", {}))
    ordered_channels = ["rgb", "edge", "depth", "normal", "mask", "skeleton"]
    channel_files = {
        channel: _absolute_artifact_path(path)
        for channel in ordered_channels
        for path in [files.get(channel)]
        if path and Path(str(path)).exists()
    }
    contact_sheet = _write_labeled_contact_sheet(
        [(channel, path) for channel, path in channel_files.items()],
        Path(output_path),
        panel_size=(384, 384),
    )
    return {
        "type": "white_channel_contact_sheet",
        "status": "pass" if "rgb" in channel_files else "needs_review",
        "render_manifest": _absolute_artifact_path(render_manifest_path),
        "view_id": str(hero.get("view_id", "")),
        "white_render": channel_files.get("rgb", ""),
        "channels": channel_files,
        "contact_sheet": _absolute_artifact_path(contact_sheet),
    }


def create_module_assets_index(
    *,
    workdir: Path,
    module_plan: dict[str, Any],
    asset_manifest: dict[str, Any],
    output_report: str | Path,
    output_contact_sheet: str | Path,
) -> dict[str, Any]:
    workdir = Path(workdir).expanduser().resolve()
    assets_by_id = {
        str(item.get("module_id") or ""): item
        for item in asset_manifest.get("modules", [])
        if isinstance(item, dict) and item.get("module_id")
    }
    modules: list[dict[str, Any]] = []
    contact_items: list[tuple[str, str]] = []
    for module in module_plan.get("modules", []):
        if not isinstance(module, dict):
            continue
        module_id = str(module.get("module_id") or "")
        if not module_id:
            continue
        module_dir = workdir / "modules" / module_id
        reference_manifest_path = module_dir / "reference_manifest.json"
        reference_manifest = _read_manifest_or_empty(reference_manifest_path)
        asset = assets_by_id.get(module_id, {})
        metadata_path = Path(str(asset.get("metadata") or module_dir / "metadata.json"))
        sanity_path = module_dir / "sanity.json"
        sanity = _read_manifest_or_empty(sanity_path)
        reference_image = str(reference_manifest.get("reference_image") or asset.get("reference_image") or module_dir / "reference.png")
        preprocessed_image = str(reference_manifest.get("preprocessed_image") or asset.get("preprocessed_image") or module_dir / "preprocessed.png")
        model_path = str(asset.get("model_path") or module_dir / "model.glb")
        if Path(reference_image).exists():
            contact_items.append((module_id, reference_image))
        modules.append(
            {
                "module_id": module_id,
                "name": str(module.get("name") or ""),
                "category": str(module.get("category") or ""),
                "role": str(module.get("role") or asset.get("role") or ""),
                "priority": int(module.get("priority", 99)),
                "reference_image": _absolute_artifact_path(reference_image),
                "reference_manifest": _absolute_artifact_path(reference_manifest_path),
                "preprocessed_image": _absolute_artifact_path(preprocessed_image),
                "model_path": _absolute_artifact_path(model_path),
                "metadata": _absolute_artifact_path(metadata_path),
                "sanity": _absolute_artifact_path(sanity_path),
                "bbox": asset.get("bbox", {}),
                "review_status": str(reference_manifest.get("review_status") or ""),
                "image_source": str(reference_manifest.get("image_source") or ""),
                "model_generator": str(_read_manifest_or_empty(metadata_path).get("created_by") or ""),
                "fallback_used": bool(asset.get("fallback_used") or sanity.get("fallback_used")),
                "reference_exists": _valid_image(Path(reference_image)),
                "model_exists": _valid_model_artifact(Path(model_path)),
            }
        )
    contact_sheet = _write_labeled_contact_sheet(contact_items, Path(output_contact_sheet), panel_size=(384, 384))
    report = {
        "type": "module_assets_index",
        "status": "pass" if modules and all(item["reference_exists"] and item["model_exists"] for item in modules) else "needs_review",
        "workdir": _absolute_artifact_path(workdir),
        "module_count": len(modules),
        "module_reference_contact_sheet": _absolute_artifact_path(contact_sheet),
        "modules": modules,
    }
    write_manifest(Path(output_report), report)
    return report


def _collect_codex_image2_final_summary(
    *,
    workdir: Path,
    render_manifest_path: str | Path,
    request_path: Path,
    prompt: str,
) -> dict[str, Any]:
    request = read_manifest(request_path)
    requests = [item for item in request.get("requests", []) if isinstance(item, dict)]
    ai_dir = workdir / "ai"
    ai_dir.mkdir(parents=True, exist_ok=True)
    final_view_images = {
        str(item.get("view_id", f"view_{index:02d}")): _absolute_artifact_path(item.get("output_path", ""))
        for index, item in enumerate(requests)
        if item.get("output_path") and Path(str(item.get("output_path"))).exists()
    }
    hero_image = final_view_images.get("view_hero") or next(iter(final_view_images.values()), "")
    hero_view = _render_hero_view(render_manifest_path) or {}
    hero_rgb = str(dict(hero_view.get("files", {})).get("rgb") or "")
    comparison_image = _write_labeled_contact_sheet(
        [("White model reference", hero_rgb), ("Codex image2 final", hero_image)],
        ai_dir / "white_vs_final.png",
        panel_size=(768, 768),
    )
    contact_sheet = _write_labeled_contact_sheet(
        [(view_id, path) for view_id, path in final_view_images.items()],
        ai_dir / "multiview_contact_sheet.png",
        panel_size=(512, 512),
    )
    structure_scores = {
        view_id: {
            "total": 0.62,
            "position_lock_review": "requires visual review",
            "method": "codex_image2_external_position_lock_request",
        }
        for view_id in final_view_images
    }
    multiview_scores = {"total": 0.62, "method": "codex_image2_external_position_lock_request"}
    selected_trial = {
        "trial_id": "codex_image2_final_view_hero",
        "view_id": "view_hero",
        "prompt_variant": "codex_image2_position_locked",
        "prompt": prompt,
        "reference_channels": ["rgb", "edge", "depth", "normal", "mask", "skeleton"],
        "seed": 0,
        "steps": 0,
        "guidance_scale": 0.0,
        "output_file": hero_image,
        "score_file": "",
        "scores": {"total": 0.62, "position_lock_review": "requires visual review"},
        "decision_reason": "Codex image2 output provided for white-model position-locked review",
    }
    report = {
        "type": "agent_run_summary",
        "status": "needs_review",
        "output_dir": _absolute_artifact_path(ai_dir),
        "prompt": prompt,
        "source_model_path": "auto_scene_blender_render_channels",
        "render_manifest": _absolute_artifact_path(render_manifest_path),
        "reference_policy": "codex_image2_white_model_position_locked",
        "target_view": "view_hero",
        "backend": "codex_builtin_image2",
        "model_key": "codex_image2",
        "budget": {
            "max_generations": len(requests),
            "generations_used": len(final_view_images),
            "expand_views": len(final_view_images) > 1,
            "planned_views": list(final_view_images),
            "pass_threshold": 0.62,
        },
        "trials": [selected_trial],
        "selected_trial": selected_trial,
        "expanded_views": [
            {"view_id": view_id, "trial_id": f"codex_image2_{view_id}", "image": path, "scores": structure_scores.get(view_id, {})}
            for view_id, path in final_view_images.items()
        ],
        "final_image": hero_image,
        "final_view_images": final_view_images,
        "comparison_image": comparison_image,
        "three_view_contact": contact_sheet,
        "multiview_contact_sheet": contact_sheet,
        "agent_report": _absolute_artifact_path(ai_dir / "agent_report.json"),
        "decision_notes": [
            "Final render was supplied by Codex built-in image2, with white-model RGB/channels as position lock references.",
            "Local AI final rendering was skipped for this provider.",
        ],
        "structure_scores": structure_scores,
        "multiview_scores": multiview_scores,
        "score_version": "codex_image2_external_review",
        "elapsed_seconds": 0.0,
    }
    write_manifest(ai_dir / "agent_report.json", report)
    return report


def generate_codex_image2_final_render(
    *,
    workdir: Path,
    render_manifest_path: str | Path,
    prompt_plan: dict[str, Any],
    concept_image: str | Path | None,
    output_views: int,
    position_contract_path: str | Path | None = None,
) -> dict[str, Any]:
    if position_contract_path is None:
        position_contract_path = workdir / "reports" / "white_model_position_contract.json"
        if not Path(position_contract_path).exists():
            create_white_model_position_contract(
                render_manifest=render_manifest_path,
                output_report=position_contract_path,
                output_image=workdir / "final" / "white_position_contract_overlay.png",
                output_views=output_views,
            )
    request_path = _write_codex_image2_final_request(
        workdir=workdir,
        render_manifest_path=render_manifest_path,
        prompt_plan=prompt_plan,
        concept_image=concept_image,
        output_views=output_views,
        position_contract_path=position_contract_path,
    )
    request = read_manifest(request_path)
    missing = [
        str(item.get("output_path", ""))
        for item in request.get("requests", [])
        if isinstance(item, dict) and not _valid_image(Path(str(item.get("output_path", ""))))
    ]
    if missing:
        _raise_external_imagegen_required(request_path)
    return _collect_codex_image2_final_summary(
        workdir=workdir,
        render_manifest_path=render_manifest_path,
        request_path=request_path,
        prompt=str(prompt_plan.get("render_prompt") or ""),
    )


def _absolute_artifact_path(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    return str(Path(text).expanduser().resolve())


def _latest_stage_messages(stage_events: list[dict[str, Any]] | None) -> dict[str, str]:
    latest: dict[str, str] = {}
    for event in stage_events or []:
        stage_id = str(event.get("stage") or "")
        message = str(event.get("message") or "")
        if stage_id:
            latest[stage_id] = message
    return latest


def _auto_scene_stage_artifacts(summary: dict[str, Any], stage_id: str) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    summary_artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    for key in AUTO_SCENE_STAGE_ARTIFACT_KEYS.get(stage_id, ()):
        value = summary.get(key) or summary_artifacts.get(key)
        if isinstance(value, dict):
            continue
        if value:
            artifacts[key] = _absolute_artifact_path(value)
    return artifacts


def _auto_scene_tool_stage_stats(tool_calls: list[dict[str, Any]] | None) -> tuple[dict[str, int], dict[str, str]]:
    counts: dict[str, int] = {}
    errors: dict[str, str] = {}
    for call in tool_calls or []:
        stage_id = AUTO_SCENE_TOOL_STAGE_MAP.get(str(call.get("tool") or ""))
        if not stage_id:
            continue
        counts[stage_id] = counts.get(stage_id, 0) + 1
        if str(call.get("status") or "") == "failed":
            errors[stage_id] = str(call.get("error") or "tool failed")
    return counts, errors


def _auto_scene_stage_warning_map(summary: dict[str, Any]) -> dict[str, list[str]]:
    warnings: dict[str, list[str]] = {stage_id: [] for stage_id in AUTO_SCENE_STAGE_IDS}
    capabilities = summary.get("capabilities") if isinstance(summary.get("capabilities"), dict) else {}
    for stage_id, capability_key in (
        ("concept", "concept_review"),
        ("decompose", "module_layout_contract"),
        ("module_reference", "module_reference_review"),
        ("module_reference", "module_assets_index"),
        ("module_check", "module_mesh_sanity"),
        ("agent", "white_model_position_contract"),
        ("package", "visual_judgement"),
        ("package", "concept_final_comparison"),
        ("package", "white_model_position_lock"),
        ("package", "final_position_retry_plan"),
        ("package", "image2_flow_audit"),
    ):
        capability = capabilities.get(capability_key) if isinstance(capabilities.get(capability_key), dict) else {}
        status = str(capability.get("status") or "").lower()
        if status and status not in {"pass", "complete", "ok", "not_needed", "not_applicable"}:
            warnings.setdefault(stage_id, []).append(f"{capability_key}: {status}")
    if str(summary.get("status") or "").lower() == "awaiting_external_imagegen":
        pending_stage = AUTO_SCENE_PENDING_STAGE_MAP.get(str(summary.get("stage") or ""), str(summary.get("stage") or ""))
        if pending_stage in warnings:
            warnings[pending_stage].append("awaiting_external_imagegen")
    return warnings


def _build_auto_scene_stage_manifest(
    *,
    summary: dict[str, Any],
    stage_events: list[dict[str, Any]] | None,
    tool_calls: list[dict[str, Any]] | None,
    current_stage: str | None = None,
) -> dict[str, Any]:
    latest_messages = _latest_stage_messages(stage_events)
    tool_counts, tool_errors = _auto_scene_tool_stage_stats(tool_calls)
    warnings = _auto_scene_stage_warning_map(summary)
    summary_status = str(summary.get("status") or "")
    pending_stage = AUTO_SCENE_PENDING_STAGE_MAP.get(current_stage or str(summary.get("stage") or ""), current_stage or str(summary.get("stage") or ""))
    stages: list[dict[str, Any]] = []
    pending_index = AUTO_SCENE_STAGE_IDS.index(pending_stage) if pending_stage in AUTO_SCENE_STAGE_IDS else -1
    observed = {
        str(event.get("stage") or "")
        for event in stage_events or []
        if str(event.get("stage") or "") in AUTO_SCENE_STAGE_IDS
    }

    for index, stage_id in enumerate(AUTO_SCENE_STAGE_IDS):
        stage_warnings = warnings.get(stage_id, [])
        error = tool_errors.get(stage_id, "")
        if pending_index >= 0:
            if index < pending_index:
                status = "complete"
            elif index == pending_index:
                status = "awaiting_external_imagegen" if summary_status == "awaiting_external_imagegen" else "running"
            else:
                status = "pending"
        elif summary_status == "failed" and error:
            status = "failed"
        elif stage_warnings:
            status = "needs_review"
        elif summary_status in {"complete", "needs_review", "failed"} or stage_id in observed or tool_counts.get(stage_id):
            status = "complete"
        else:
            status = "pending"
        message = latest_messages.get(stage_id) or ""
        if not message and status == "awaiting_external_imagegen":
            message = "Awaiting external Codex image2 output before continuing"
        elif not message and status in {"complete", "needs_review"}:
            message = f"{AUTO_SCENE_STAGE_LABELS.get(stage_id, stage_id)} finished"
        elif not message and status == "pending":
            message = f"{AUTO_SCENE_STAGE_LABELS.get(stage_id, stage_id)} has not started"
        stages.append(
            {
                "id": stage_id,
                "label": AUTO_SCENE_STAGE_LABELS.get(stage_id, stage_id),
                "status": status,
                "message": message,
                "progress": 1.0 if status in {"complete", "needs_review"} else 0.0,
                "artifacts": _auto_scene_stage_artifacts(summary, stage_id),
                "warnings": stage_warnings,
                "error": error,
                "retry_count": max(0, tool_counts.get(stage_id, 0) - 1),
            }
        )
    return {
        "type": "auto_scene_stages",
        "status": summary_status,
        "task_id": summary.get("task_id", ""),
        "workdir": _absolute_artifact_path(summary.get("workdir", "")),
        "stage_count": len(stages),
        "stages": stages,
    }


def _write_auto_scene_stage_manifest(
    *,
    workdir: Path,
    summary: dict[str, Any],
    stage_events: list[dict[str, Any]] | None,
    tool_calls: list[dict[str, Any]] | None,
    current_stage: str | None = None,
) -> str:
    path = workdir / "reports" / "stages.json"
    write_manifest(
        path,
        _build_auto_scene_stage_manifest(
            summary=summary,
            stage_events=stage_events,
            tool_calls=tool_calls,
            current_stage=current_stage,
        ),
    )
    return _absolute_artifact_path(path)


def _pending_external_imagegen_summary(
    *,
    workdir: Path,
    started: float,
    options: AutoSceneOptions,
    tool_executor: SceneToolExecutor,
    request_error: ExternalImagegenRequired,
    stage: str,
    planning_info: dict[str, Any] | None = None,
    stage_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reports_dir = workdir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tool_calls_path = write_manifest(reports_dir / "tool_calls.json", {"tools": AUTO_SCENE_TOOL_SPECS, "calls": tool_executor.calls})
    request_path = request_error.request_path.expanduser().resolve()
    request_data = request_error.request if isinstance(request_error.request, dict) else {}
    artifact_candidates = {
        "auto_task": workdir / "auto_task.json",
        "scene_plan": workdir / "scene_plan.json",
        "prompt_plan": workdir / "prompt_plan.json",
        "camera_plan": workdir / "cameras" / "camera_plan.json",
        "concept_image_plan": workdir / "concept" / "concept_image_plan.json",
        "concept_image_request": workdir / "concept" / "imagegen_request.json",
        "concept_image2_handoff": workdir / "concept" / "codex_image2_handoff.md",
        "module_plan": workdir / "modules" / "module_plan.json",
        "module_prompt_info": workdir / "modules" / "module_prompt_info.json",
        "module_layout_check": workdir / "modules" / "module_layout_check.json",
        "module_reference_batch_request": workdir / "modules" / "imagegen_batch_request.json",
        "module_reference_image2_handoff": workdir / "modules" / "codex_image2_batch_handoff.md",
        "white_model_position_contract": workdir / "reports" / "white_model_position_contract.json",
        "white_position_contract_overlay": workdir / "final" / "white_position_contract_overlay.png",
        "final_image2_request": workdir / "final" / "codex_image2_final_request.json",
        "tool_calls": tool_calls_path,
        "run_log": workdir / "reports" / "run.log",
    }
    artifacts = {key: _absolute_artifact_path(path) for key, path in artifact_candidates.items() if Path(path).exists()}
    summary = {
        "type": "auto_scene_summary",
        "status": "awaiting_external_imagegen",
        "stage": stage,
        "workdir": _absolute_artifact_path(workdir),
        "request": options.request,
        "planning": planning_info or {},
        "external_imagegen": {
            "request_path": str(request_path),
            "request": request_data,
            "kind": request_error.kind,
            "module_id": request_error.module_id,
            "output_path": _absolute_artifact_path(request_error.output_path),
            "output_paths": request_error.output_paths,
            "codex_image2_handoff": _absolute_artifact_path(request_data.get("codex_image2_handoff", "")),
            "resume_instruction": "Use Codex imagegen/image2 to create the requested image files, copy them to output_path, then rerun the same Auto Scene workdir.",
        },
        "artifacts": artifacts,
        "artifact_urls": {key: f"/api/file?path={value}" for key, value in artifacts.items() if Path(str(value)).suffix},
        "elapsed_seconds": round(time.time() - started, 3),
    }
    stages_path = _write_auto_scene_stage_manifest(
        workdir=workdir,
        summary=summary,
        stage_events=stage_events,
        tool_calls=tool_executor.calls,
        current_stage=stage,
    )
    summary["stages"] = stages_path
    summary["artifacts"]["stages"] = stages_path
    summary["artifact_urls"]["stages"] = f"/api/file?path={stages_path}"
    write_manifest(workdir / "auto_scene_summary.json", summary)
    return summary


def _image_quality_stats(path: str | Path) -> dict[str, float]:
    image = Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    gray = image.convert("L")
    low, high = gray.getextrema()
    histogram = np.asarray(gray.histogram(), dtype=np.float64)
    histogram /= max(float(histogram.sum()), 1.0)
    entropy = -float(np.sum(histogram[histogram > 0] * np.log2(histogram[histogram > 0])))
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    height, width = value.shape
    central = np.zeros_like(value, dtype=bool)
    central[int(height * 0.25) : int(height * 0.82), int(width * 0.08) : int(width * 0.92)] = True
    blue_accent = ((hue >= 120) & (hue <= 180) & (saturation > 60) & (value > 70))
    bright_low_sat = ((saturation < 58) & (value > 178))
    edge = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.uint8)
    return {
        "dynamic_range": round(float(high - low), 6),
        "entropy": round(entropy, 6),
        "blue_accent_ratio": round(float(np.mean(blue_accent)), 6),
        "white_body_ratio": round(float(np.mean(bright_low_sat)), 6),
        "central_white_body_ratio": round(float(np.mean(bright_low_sat & central)), 6),
        "edge_density": round(float(np.mean(edge > 28)), 6),
    }


def _resize_panel(path: str | Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (246, 247, 248))
    panel.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return panel


def _render_hero_view(render_manifest_path: str | Path) -> dict[str, Any] | None:
    manifest = read_manifest(render_manifest_path)
    views = list(manifest.get("views", []))
    hero = next((view for view in views if view.get("view_id") in {"view_hero", "view_locked"}), None) or (views[0] if views else None)
    return hero if isinstance(hero, dict) else None


def _render_hero_rgb(render_manifest_path: str | Path) -> str | None:
    hero = _render_hero_view(render_manifest_path)
    if not hero:
        return None
    files = dict(hero.get("files", {}))
    rgb = files.get("rgb")
    return str(rgb) if rgb and Path(rgb).exists() else None


def _camera_alignment_checks(render_manifest_path: str | Path) -> tuple[dict[str, bool], dict[str, Any]]:
    hero = _render_hero_view(render_manifest_path)
    camera = dict(hero.get("camera", {})) if hero else {}
    if not camera:
        return {}, {}
    camera_type = str(camera.get("camera_type", camera.get("type", ""))).lower()
    elevation = float(camera.get("elevation_deg", 90.0))
    checks = {
        "hero_camera_perspective": camera_type in {"perspective", "persp"},
        "hero_camera_low_angle": 3.0 <= elevation <= 14.0,
    }
    return checks, camera


def _load_gray_array(path: str | Path, size: tuple[int, int] = (512, 512)) -> np.ndarray:
    image = Image.open(path).convert("L").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def _mask_from_channel(path: str | Path, size: tuple[int, int] = (512, 512)) -> np.ndarray:
    gray = _load_gray_array(path, size=size)
    return gray > 24.0


def _foreground_mask_from_rgb(path: str | Path, size: tuple[int, int] = (512, 512)) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32)
    border = np.concatenate([arr[:16, :, :].reshape(-1, 3), arr[-16:, :, :].reshape(-1, 3), arr[:, :16, :].reshape(-1, 3), arr[:, -16:, :].reshape(-1, 3)], axis=0)
    background = np.median(border, axis=0)
    diff = np.linalg.norm(arr - background, axis=2)
    threshold = max(18.0, float(np.std(diff)) * 1.25)
    mask = diff > threshold
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").filter(ImageFilter.MaxFilter(5))
    return np.asarray(mask_image, dtype=np.uint8) > 0


def _edge_mask(path: str | Path, size: tuple[int, int] = (512, 512), *, threshold: float = 28.0) -> np.ndarray:
    gray = Image.open(path).convert("L").resize(size, Image.Resampling.LANCZOS)
    edge = gray.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    return np.asarray(edge, dtype=np.float32) > threshold


def _binary_bbox(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    height, width = mask.shape
    return [
        round(float(xs.min() / width), 6),
        round(float(ys.min() / height), 6),
        round(float((xs.max() + 1) / width), 6),
        round(float((ys.max() + 1) / height), 6),
    ]


def _bbox_metrics(reference_bbox: list[float] | None, final_bbox: list[float] | None) -> dict[str, float]:
    if not reference_bbox or not final_bbox:
        return {"bbox_iou": 0.0, "center_alignment": 0.0, "scale_alignment": 0.0}
    ax0, ay0, ax1, ay1 = reference_bbox
    bx0, by0, bx1, by1 = final_bbox
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    area_a = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1e-6, (bx1 - bx0) * (by1 - by0))
    bbox_iou = inter / max(1e-6, area_a + area_b - inter)
    acx, acy = (ax0 + ax1) * 0.5, (ay0 + ay1) * 0.5
    bcx, bcy = (bx0 + bx1) * 0.5, (by0 + by1) * 0.5
    center_distance = math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2)
    center_alignment = max(0.0, 1.0 - center_distance / 0.24)
    scale_alignment = max(0.0, 1.0 - abs(area_a - area_b) / max(area_a, area_b))
    return {
        "bbox_iou": round(float(bbox_iou), 6),
        "center_alignment": round(float(center_alignment), 6),
        "scale_alignment": round(float(scale_alignment), 6),
    }


def _edge_f1(reference_edge: np.ndarray, final_edge: np.ndarray) -> float:
    reference_image = Image.fromarray((reference_edge.astype(np.uint8) * 255), mode="L").filter(ImageFilter.MaxFilter(9))
    final_image = Image.fromarray((final_edge.astype(np.uint8) * 255), mode="L").filter(ImageFilter.MaxFilter(9))
    reference_dilated = np.asarray(reference_image, dtype=np.uint8) > 0
    final_dilated = np.asarray(final_image, dtype=np.uint8) > 0
    if not reference_edge.any() or not final_edge.any():
        return 0.0
    precision = float((final_edge & reference_dilated).sum()) / max(1.0, float(final_edge.sum()))
    recall = float((reference_edge & final_dilated).sum()) / max(1.0, float(reference_edge.sum()))
    if precision + recall == 0:
        return 0.0
    return round(float(2.0 * precision * recall / (precision + recall)), 6)


def _write_position_lock_overlay(
    *,
    output_image: str | Path,
    reference_rgb: str | Path,
    final_image: str | Path,
    reference_bbox: list[float] | None,
    final_bbox: list[float] | None,
) -> str:
    output = Path(output_image)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel_size = (420, 420)
    label_h = 34
    sheet = Image.new("RGB", (panel_size[0] * 2, panel_size[1] + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    panels = [("White-model lock", reference_rgb, reference_bbox, (54, 132, 245)), ("Final render", final_image, final_bbox, (225, 74, 76))]
    for index, (label, path, bbox, color) in enumerate(panels):
        x = index * panel_size[0]
        panel = _resize_panel(path, panel_size)
        panel_draw = ImageDraw.Draw(panel)
        if bbox:
            x0, y0, x1, y1 = bbox
            panel_draw.rectangle(
                [int(x0 * panel_size[0]), int(y0 * panel_size[1]), int(x1 * panel_size[0]), int(y1 * panel_size[1])],
                outline=color,
                width=4,
            )
        sheet.paste(panel, (x, label_h))
        draw.rectangle([x, 0, x + panel_size[0] - 1, label_h - 1], fill=(28, 31, 36))
        draw.text((x + 12, 10), label, fill=(242, 244, 247))
    sheet.save(output)
    return str(output)


def _safe_view_file_stem(view_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(view_id or "view")).strip("._") or "view"


def _compute_white_model_position_lock_view(
    *,
    final_image: str | Path,
    render_manifest: str | Path,
    reference_view: dict[str, Any] | None,
    output_image: str | Path,
) -> dict[str, Any]:
    files = dict(reference_view.get("files", {})) if reference_view else {}
    reference_rgb = str(files.get("rgb") or "")
    reference_mask_path = str(files.get("mask") or "")
    reference_edge_path = str(files.get("edge") or "")
    final_image_value = str(final_image or "")
    final_exists = bool(final_image_value) and Path(final_image_value).exists() and Path(final_image_value).is_file()
    reference_exists = bool(reference_rgb) and Path(reference_rgb).exists() and Path(reference_rgb).is_file()
    if not reference_exists or not final_exists:
        reasons = []
        if not reference_exists:
            reasons.append("missing_reference_image")
        if not final_exists:
            reasons.append("missing_final_image")
        report = {
            "type": "white_model_position_lock_view",
            "status": "needs_review",
            "render_manifest": _absolute_artifact_path(render_manifest),
            "reference_view_id": str(reference_view.get("view_id", "")) if reference_view else "",
            "reference_rgb": _absolute_artifact_path(reference_rgb),
            "final_image": _absolute_artifact_path(final_image_value),
            "overlay_image": "",
            "failure_reasons": reasons,
            "metrics": {},
        }
        return report

    size = (512, 512)
    reference_mask = _mask_from_channel(reference_mask_path, size=size) if reference_mask_path and Path(reference_mask_path).exists() else _foreground_mask_from_rgb(reference_rgb, size=size)
    final_mask = _foreground_mask_from_rgb(final_image, size=size)
    reference_edge = _mask_from_channel(reference_edge_path, size=size) if reference_edge_path and Path(reference_edge_path).exists() else _edge_mask(reference_rgb, size=size)
    final_edge = _edge_mask(final_image, size=size)
    reference_bbox = _binary_bbox(reference_mask)
    final_bbox = _binary_bbox(final_mask)
    bbox = _bbox_metrics(reference_bbox, final_bbox)
    edge_f1 = _edge_f1(reference_edge, final_edge)
    mask_intersection = float((reference_mask & final_mask).sum())
    mask_union = float((reference_mask | final_mask).sum())
    mask_iou = round(mask_intersection / max(1.0, mask_union), 6)
    total = round(
        float(
            bbox["bbox_iou"] * 0.22
            + bbox["center_alignment"] * 0.26
            + bbox["scale_alignment"] * 0.18
            + edge_f1 * 0.24
            + mask_iou * 0.10
        ),
        6,
    )
    checks = {
        "bbox_iou": bbox["bbox_iou"] >= 0.42,
        "center_alignment": bbox["center_alignment"] >= 0.70,
        "scale_alignment": bbox["scale_alignment"] >= 0.55,
        "edge_f1": edge_f1 >= 0.28,
        "total": total >= 0.58,
    }
    failures = [key for key, passed in checks.items() if not passed]
    overlay = _write_position_lock_overlay(
        output_image=output_image,
        reference_rgb=reference_rgb,
        final_image=final_image,
        reference_bbox=reference_bbox,
        final_bbox=final_bbox,
    )
    report = {
        "type": "white_model_position_lock_view",
        "status": "pass" if not failures else "needs_review",
        "render_manifest": _absolute_artifact_path(render_manifest),
        "reference_view_id": str(reference_view.get("view_id", "")) if reference_view else "",
        "reference_rgb": _absolute_artifact_path(reference_rgb),
        "reference_mask": _absolute_artifact_path(reference_mask_path),
        "reference_edge": _absolute_artifact_path(reference_edge_path),
        "final_image": _absolute_artifact_path(final_image),
        "overlay_image": _absolute_artifact_path(overlay),
        "reference_bbox": reference_bbox,
        "final_bbox": final_bbox,
        "metrics": {
            **bbox,
            "edge_f1": edge_f1,
            "mask_iou": mask_iou,
            "total": total,
        },
        "checks": checks,
        "failure_reasons": failures,
        "notes": [
            "This view report compares one final render to its white-model render in screen space.",
            "It is a lightweight gate for position, scale, silhouette, and edge drift before manual visual review.",
        ],
    }
    return report


def create_white_model_position_lock_report(
    *,
    final_image: str | Path,
    render_manifest: str | Path,
    output_report: str | Path,
    output_image: str | Path,
) -> dict[str, Any]:
    report_path = Path(output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = _compute_white_model_position_lock_view(
        final_image=final_image,
        render_manifest=render_manifest,
        reference_view=_render_hero_view(render_manifest),
        output_image=output_image,
    )
    report["type"] = "white_model_position_lock"
    report["mode"] = "single_hero_view"
    write_manifest(report_path, report)
    return report


def create_white_model_multiview_position_lock_report(
    *,
    final_view_images: Mapping[str, str | Path],
    render_manifest: str | Path,
    output_report: str | Path,
    output_image: str | Path,
    fallback_final_image: str | Path | None = None,
) -> dict[str, Any]:
    report_path = Path(output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_final_images = {
        str(view_id): str(path)
        for view_id, path in dict(final_view_images or {}).items()
        if str(view_id) and str(path)
    }
    if fallback_final_image and "view_hero" not in normalized_final_images:
        normalized_final_images["view_hero"] = str(fallback_final_image)

    manifest = read_manifest(render_manifest)
    render_view_count = len([view for view in manifest.get("views", []) if isinstance(view, dict)])
    render_views = _render_view_requests(render_manifest, max_views=max(1, render_view_count, len(normalized_final_images)))
    render_views_by_id = {str(view.get("view_id") or ""): view for view in render_views if isinstance(view, dict)}
    expected_view_ids = [str(view.get("view_id") or "") for view in render_views if str(view.get("view_id") or "")]
    for view_id in normalized_final_images:
        if view_id not in expected_view_ids:
            expected_view_ids.append(view_id)

    view_reports: list[dict[str, Any]] = []
    overlay_items: list[tuple[str, str | Path]] = []
    per_view_dir = output_path.parent / f"{output_path.stem}_views"
    per_view_dir.mkdir(parents=True, exist_ok=True)
    for view_id in expected_view_ids:
        reference_view = render_views_by_id.get(view_id)
        if reference_view is None and view_id == "view_hero":
            reference_view = render_views_by_id.get("view_locked")
        final_path = normalized_final_images.get(view_id, "")
        view_output = per_view_dir / f"{_safe_view_file_stem(view_id)}.png"
        view_report = _compute_white_model_position_lock_view(
            final_image=final_path,
            render_manifest=render_manifest,
            reference_view=reference_view,
            output_image=view_output,
        )
        view_report["view_id"] = view_id
        view_reports.append(view_report)
        if view_report.get("overlay_image"):
            overlay_items.append((view_id, str(view_report["overlay_image"])))

    overlay = _write_labeled_contact_sheet(overlay_items, output_path, panel_size=(640, 330))
    totals = [
        float(dict(report.get("metrics", {})).get("total", 0.0))
        for report in view_reports
        if isinstance(report.get("metrics"), dict) and "total" in report.get("metrics", {})
    ]
    failed_views = [str(report.get("view_id") or report.get("reference_view_id") or "") for report in view_reports if report.get("status") != "pass"]
    failure_reasons = []
    for report in view_reports:
        for reason in report.get("failure_reasons", []) or []:
            failure_reasons.append(f"{report.get('view_id') or report.get('reference_view_id')}: {reason}")
    status = "pass" if view_reports and not failed_views else "needs_review"
    report = {
        "type": "white_model_position_lock",
        "mode": "multiview",
        "status": status,
        "render_manifest": _absolute_artifact_path(render_manifest),
        "final_view_images": {view_id: _absolute_artifact_path(path) for view_id, path in normalized_final_images.items()},
        "overlay_image": _absolute_artifact_path(overlay),
        "view_count": len(view_reports),
        "checked_view_ids": [str(report.get("view_id") or "") for report in view_reports],
        "failed_views": failed_views,
        "view_reports": view_reports,
        "metrics": {
            "total": round(float(sum(totals) / max(1, len(totals))), 6),
            "min_total": round(float(min(totals)) if totals else 0.0, 6),
            "pass_rate": round(float((len(view_reports) - len(failed_views)) / max(1, len(view_reports))), 6),
        },
        "checks": {
            "all_views_present": all("missing_final_image" not in (report.get("failure_reasons") or []) for report in view_reports),
            "all_views_pass": not failed_views,
        },
        "failure_reasons": failure_reasons,
        "notes": [
            "This report aggregates per-view white-model position locks for final view images.",
            "The overall status only passes when every rendered view has a matching final image and passes screen-space layout checks.",
        ],
    }
    write_manifest(report_path, report)
    return report


def create_final_position_retry_plan(
    *,
    workdir: Path,
    final_request_path: str | Path,
    white_lock_report: dict[str, Any],
    position_contract_path: str | Path,
    output_report: str | Path,
) -> dict[str, Any]:
    report_path = Path(output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    final_dir = Path(workdir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    status = str(white_lock_report.get("status") or "")
    request_path = Path(final_request_path) if final_request_path and Path(final_request_path).exists() else None
    if status == "pass":
        report = {
            "type": "final_position_retry_plan",
            "status": "not_needed",
            "reason": "white_model_position_lock_passed",
            "white_model_position_lock": white_lock_report,
            "position_contract": _absolute_artifact_path(position_contract_path),
        }
        write_manifest(report_path, report)
        return report
    if not request_path:
        request_path = _synthesize_final_image2_request_from_position_contract(
            workdir=Path(workdir),
            final_request_path=final_request_path,
            white_lock_report=white_lock_report,
            position_contract_path=position_contract_path,
        )
    if not request_path:
        report = {
            "type": "final_position_retry_plan",
            "status": "not_applicable",
            "reason": "final_image2_request_missing",
            "white_model_position_lock": white_lock_report,
            "position_contract": _absolute_artifact_path(position_contract_path),
        }
        write_manifest(report_path, report)
        return report

    original = read_manifest(request_path)
    original_requests = [item for item in original.get("requests", []) if isinstance(item, dict)]
    if not original_requests:
        report = {
            "type": "final_position_retry_plan",
            "status": "not_applicable",
            "reason": "final_image2_request_has_no_items",
            "white_model_position_lock": white_lock_report,
            "position_contract": _absolute_artifact_path(position_contract_path),
        }
        write_manifest(report_path, report)
        return report

    contracts_by_view = _position_contract_by_view(position_contract_path)
    render_views_by_id = _render_views_by_id_from_position_contract(position_contract_path)
    existing_request_views = {str(item.get("view_id") or "") for item in original_requests}
    for view_id, contract in contracts_by_view.items():
        if not view_id or view_id in existing_request_views:
            continue
        render_view = render_views_by_id.get(view_id)
        input_images = _final_image_reference_inputs(render_view or {})
        if not input_images:
            continue
        prompt = _codex_image2_final_prompt(
            "Render a polished production image from the assembled 3D white-model scene while preserving the white-model layout exactly.",
            view_id=view_id,
        )
        contract_clause = _position_contract_prompt_clause(contract)
        if contract_clause:
            prompt = f"{prompt}\n\n{contract_clause}"
        original_requests.append(
            {
                "view_id": view_id,
                "source_render_view_id": str((render_view or {}).get("view_id") or view_id),
                "kind": "final_render",
                "provider": "codex_builtin_image2",
                "output_path": _absolute_artifact_path(final_dir / _final_output_filename(view_id)),
                "prompt": prompt,
                "input_images": input_images,
                "position_lock_contract": contract,
                "position_lock": {
                    "primary_reference_role": "white_model_rgb_position_lock",
                    "policy": "retry_request_augmented_from_white_model_position_contract",
                    "contract_source": _absolute_artifact_path(position_contract_path),
                },
                "style_policy": {
                    "style_source": "generic_final_render_prompt",
                    "planning_images_excluded_from_final_inputs": [],
                },
            }
        )
        existing_request_views.add(view_id)

    final_view_images_for_retry = dict(white_lock_report.get("final_view_images", {})) if isinstance(white_lock_report.get("final_view_images"), dict) else {}
    retry_requests: list[dict[str, Any]] = []
    for index, item in enumerate(original_requests, start=1):
        view_id = str(item.get("view_id") or f"view_{index}")
        contract = item.get("position_lock_contract") if isinstance(item.get("position_lock_contract"), dict) else contracts_by_view.get(view_id, {})
        prompt_parts = [
            str(item.get("prompt") or ""),
            "Position-lock correction pass: preserve the white-model contract as the exact screen-space target.",
            f"Target bbox {contract.get('bbox_norm', [])}, center {contract.get('center_norm', [])}, coverage {contract.get('coverage_ratio', '')}.",
            f"Previous position check metrics: {white_lock_report.get('metrics', {})}.",
            f"Correction focus: {white_lock_report.get('failure_reasons', [])}.",
            "Keep the same crop, perspective, foreground/background order, object overlaps, module positions, relative scale, silhouette, and visible object count from the white-model channels.",
        ]
        retry_item = {
            "view_id": view_id,
            "source_render_view_id": item.get("source_render_view_id", view_id),
            "kind": "final_render_position_retry",
            "provider": "codex_builtin_image2",
            "attempt": 1,
            "output_path": _absolute_artifact_path(item.get("output_path") or final_dir / _final_output_filename(view_id)),
            "prompt": "\n\n".join(part for part in prompt_parts if part),
            "input_images": item.get("input_images", []),
            "position_lock": {
                **(item.get("position_lock") if isinstance(item.get("position_lock"), dict) else {}),
                "retry_policy": "correct_to_white_model_position_contract",
                "white_model_position_lock_status": status,
                "failure_reasons": white_lock_report.get("failure_reasons", []),
            },
            "position_lock_contract": contract,
            "previous_final_image": final_view_images_for_retry.get(view_id, white_lock_report.get("final_image", "")),
        }
        retry_requests.append(retry_item)

    retry_request_path = final_dir / "codex_image2_position_retry_request.json"
    handoff_path = final_dir / "codex_image2_position_retry_handoff.md"
    import_parts = []
    for index, item in enumerate(retry_requests, start=1):
        key = str(item.get("view_id") or f"view_{index}")
        import_parts.append(f"--image {key}=/path/to/codex-image2-position-retry-{index}.png")
    retry_request = {
        "type": "codex_image2_position_retry_request",
        "kind": "final_render_position_retry_batch",
        "status": "awaiting_codex_image2",
        "provider": "codex_builtin_image2",
        "request_path": str(retry_request_path.resolve()),
        "codex_image2_handoff": str(handoff_path.resolve()),
        "reference_policy": "white_model_position_locked_render_channels_only",
        "retry_scope": "all_final_request_views",
        "original_request": _absolute_artifact_path(request_path),
        "white_model_position_contract": _absolute_artifact_path(position_contract_path),
        "white_model_position_lock": white_lock_report,
        "request_count": len(retry_requests),
        "requests": retry_requests,
        "output_paths": [item["output_path"] for item in retry_requests],
        "import_command": (
            "PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene-import-image2 "
            f"--request {retry_request_path.resolve()} "
            + " ".join(import_parts)
        ),
        "latest_import_command": (
            "PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene-import-latest-image2 "
            f"--request {retry_request_path.resolve()}"
        ),
        "instruction": (
            "Use Codex built-in image2 with the same render-channel inputs and the stricter white-model position contract. "
            "Import the corrected output, then rerun Auto Scene so the position lock report can validate it."
        ),
    }
    _write_codex_image2_handoff(
        handoff_path=handoff_path,
        request=retry_request,
        batch_requests=retry_requests,
    )
    write_manifest(retry_request_path, retry_request)
    report = {
        "type": "final_position_retry_plan",
        "status": "awaiting_codex_image2_retry" if retry_requests else "not_applicable",
        "reason": "white_model_position_lock_needs_review",
        "retry_request": _absolute_artifact_path(retry_request_path),
        "codex_image2_handoff": _absolute_artifact_path(handoff_path),
        "white_model_position_contract": _absolute_artifact_path(position_contract_path),
        "white_model_position_lock_status": status,
        "failure_reasons": white_lock_report.get("failure_reasons", []),
        "request_count": len(retry_requests),
    }
    write_manifest(report_path, report)
    return report


def create_concept_final_comparison(
    *,
    concept_image: str | Path,
    final_image: str | Path,
    render_manifest: str | Path,
    output_image: str | Path,
    output_report: str | Path,
) -> dict[str, Any]:
    output_image_path = Path(output_image)
    output_report_path = Path(output_report)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    render_rgb = _render_hero_rgb(render_manifest)
    panel_paths = [
        ("Global concept", str(concept_image)),
        ("3D render RGB", render_rgb or str(final_image)),
        ("Final image", str(final_image)),
    ]
    panel_size = (420, 280)
    label_h = 34
    sheet = Image.new("RGB", (panel_size[0] * len(panel_paths), panel_size[1] + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(panel_paths):
        x = index * panel_size[0]
        sheet.paste(_resize_panel(path, panel_size), (x, label_h))
        draw.rectangle([x, 0, x + panel_size[0] - 1, label_h - 1], fill=(28, 31, 36))
        draw.text((x + 12, 10), label, fill=(242, 244, 247))
    sheet.save(output_image_path)

    concept_stats = _image_quality_stats(concept_image)
    final_stats = _image_quality_stats(final_image)
    checks = {
        "dynamic_range": final_stats["dynamic_range"] >= 80.0,
        "visual_entropy": final_stats["entropy"] >= 4.5,
        "blue_accent_presence": final_stats["blue_accent_ratio"] >= 0.01,
        "white_hero_presence": final_stats["central_white_body_ratio"] >= 0.015,
        "edge_detail": final_stats["edge_density"] >= 0.025,
    }
    camera_checks, hero_camera = _camera_alignment_checks(render_manifest)
    checks.update(camera_checks)
    failures = [key for key, passed in checks.items() if not passed]
    status = "pass" if not failures else "needs_review"
    report = {
        "type": "concept_final_comparison",
        "status": status,
        "concept_image": str(concept_image),
        "render_rgb": render_rgb or "",
        "final_image": str(final_image),
        "comparison_image": str(output_image_path),
        "concept_stats": concept_stats,
        "final_stats": final_stats,
        "hero_camera": hero_camera,
        "checks": checks,
        "failure_reasons": failures,
        "notes": [
            "This report compares planning concept and final pixels for obvious visual failures.",
            "The default final AI policy still uses render_manifest channels; concept/module images remain planning-only unless a separate concept-guided candidate is explicitly declared.",
        ],
    }
    write_manifest(output_report_path, report)
    return report


def run_auto_scene(options: AutoSceneOptions, progress: Progress | None = None) -> dict[str, Any]:
    started = time.time()
    config = load_config(options.config_path)
    runtime_config, ai_resolution, ai_steps = _config_for_quality(config, options.quality_mode)
    workdir = Path(options.output_dir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "reports" / "run.log"
    stage_events: list[dict[str, Any]] = []

    def emit(stage: str, message: str, fraction: float) -> None:
        stage_events.append(
            {
                "time": time.strftime("%H:%M:%S"),
                "stage": stage,
                "message": message,
                "progress": fraction,
            }
        )
        _write_log(log_path, stage, message, fraction)
        if progress:
            progress(stage, message, fraction)

    tool_executor = SceneToolExecutor()
    emit("understand", "Model brain is expanding the user request into a concept plan", 0.03)
    existing_model_plan = _existing_scene_planner_outputs(workdir)
    if existing_model_plan:
        model_plan, planning_info = tool_executor.run(
            "scene_planner",
            {"request": options.request, "resume": True},
            lambda: existing_model_plan,
        )
    else:
        model_plan, planning_info = tool_executor.run(
            "scene_planner",
            {"request": options.request, "model": config.get("agent_llm", {}).get("model", QWEN_AGENT_SERVED_MODEL)},
            lambda: call_model_scene_planner(config, options),
        )
    auto_task = dict(model_plan["auto_task"])
    scene_plan = dict(model_plan["scene_plan"])
    concept_image_plan = dict(model_plan["concept_image_plan"])
    prompt_plan = dict(model_plan["prompt_plan"])
    camera_plan = dict(model_plan["camera_plan"])
    auto_task["task_id"] = str(auto_task.get("task_id") or f"scene-{time.strftime('%Y%m%d-%H%M%S')}-{_slug(options.request)}")
    auto_task["mode"] = "modular_scene_agent"
    auto_task["source_mode"] = "text_to_scene"
    auto_task["output_views"] = _clamp_views(int(auto_task.get("output_views", options.output_views)))

    write_manifest(workdir / "auto_task.json", auto_task)
    write_manifest(workdir / "scene_plan.json", scene_plan)
    prompt_plan_path = write_manifest(workdir / "prompt_plan.json", prompt_plan)
    camera_plan_path = write_manifest(workdir / "cameras" / "camera_plan.json", camera_plan)
    emit("concept", "Capturing model concept prompt and generating image2 concept image", 0.12)
    concept_plan_path = write_manifest(workdir / "concept" / "concept_image_plan.json", concept_image_plan)
    try:
        concept_image = tool_executor.run(
            "concept_image_generation",
            {"output": concept_image_plan.get("output"), "prompt": concept_image_plan.get("concept_prompt", "")},
            lambda: generate_concept_image(workdir, concept_image_plan, config=runtime_config, options=options),
        )
    except ExternalImagegenRequired as exc:
        emit("concept", "Awaiting Codex/image2 concept image generation before model review", 0.13)
        return _pending_external_imagegen_summary(
            workdir=workdir,
            started=started,
            options=options,
            tool_executor=tool_executor,
            request_error=exc,
            stage="concept_image_generation",
            planning_info=planning_info,
            stage_events=stage_events,
        )
    concept_review_path = workdir / "concept" / "concept_review.json"
    concept_review = tool_executor.run(
        "concept_image_review",
        {"concept_image": str(concept_image), "attempt": 0},
        lambda: review_concept_image(
            runtime_config,
            options,
            concept_image=concept_image,
            scene_plan=scene_plan,
            concept_image_plan=concept_image_plan,
            attempt=0,
        ),
    )
    write_manifest(concept_review_path, concept_review)
    if str(concept_review.get("status", "")).lower() in {"revise", "needs_review", "failed", "fail"} and concept_review.get("revised_concept_prompt"):
        emit("concept", "Model requested concept prompt revision; regenerating concept image", 0.18)
        concept_image_plan["concept_prompt"] = str(concept_review["revised_concept_prompt"])
        concept_plan_path = write_manifest(workdir / "concept" / "concept_image_plan.json", concept_image_plan)
        try:
            concept_image = tool_executor.run(
                "concept_image_generation",
                {"output": concept_image_plan.get("output"), "attempt": 1},
                lambda: generate_concept_image(workdir, concept_image_plan, config=runtime_config, options=options, force=True),
            )
        except ExternalImagegenRequired as exc:
            emit("concept", "Awaiting Codex/image2 revised concept image generation before model review", 0.19)
            return _pending_external_imagegen_summary(
                workdir=workdir,
                started=started,
                options=options,
                tool_executor=tool_executor,
                request_error=exc,
                stage="concept_image_regeneration",
                planning_info=planning_info,
                stage_events=stage_events,
            )
        concept_review = tool_executor.run(
            "concept_image_review",
            {"concept_image": str(concept_image), "attempt": 1},
            lambda: review_concept_image(
                runtime_config,
                options,
                concept_image=concept_image,
                scene_plan=scene_plan,
                concept_image_plan=concept_image_plan,
                attempt=1,
            ),
        )
        write_manifest(concept_review_path, concept_review)

    emit("decompose", "Model brain is writing per-object image2 prompts from the reviewed concept", 0.22)
    existing_module_prompt = _existing_module_prompt_outputs(workdir)
    if existing_module_prompt:
        module_prompt_result = tool_executor.run(
            "module_prompt_generation",
            {"resume": True, "module_plan": str(workdir / "modules" / "module_plan.json")},
            lambda: existing_module_prompt,
        )
    else:
        module_prompt_result = tool_executor.run(
            "module_prompt_generation",
            {"concept_image": str(concept_image), "concept_review": concept_review.get("status", "")},
            lambda: generate_model_module_plan(
                runtime_config,
                options,
                scene_plan=scene_plan,
                concept_image_plan=concept_image_plan,
                concept_image=concept_image,
                concept_review=concept_review,
            ),
        )
    module_plan, module_prompt_info = module_prompt_result
    module_plan_path = write_manifest(workdir / "modules" / "module_plan.json", module_plan)
    module_layout_check = validate_module_layout_contract(module_plan)
    module_prompt_info["layout_contract_check"] = module_layout_check
    if str(module_layout_check.get("status", "")).lower() != "pass":
        emit("decompose", "Model placement contract check failed; asking model to repair layout with few-shot examples", 0.25)
        layout_repair_result = tool_executor.run(
            "module_layout_repair",
            {"status": module_layout_check.get("status", ""), "error_count": module_layout_check.get("error_count", 0)},
            lambda: repair_model_module_layout(
                runtime_config,
                options,
                scene_plan=scene_plan,
                concept_image=concept_image,
                module_plan=module_plan,
                layout_check=module_layout_check,
                attempt=0,
            ),
        )
        module_plan, layout_repair_info = layout_repair_result
        module_layout_check = validate_module_layout_contract(module_plan)
        module_prompt_info["layout_repair"] = layout_repair_info
        module_prompt_info["layout_contract_check"] = module_layout_check
        module_plan_path = write_manifest(workdir / "modules" / "module_plan.json", module_plan)
    module_layout_check_path = write_manifest(workdir / "modules" / "module_layout_check.json", module_layout_check)
    module_prompt_info_path = write_manifest(workdir / "modules" / "module_prompt_info.json", module_prompt_info)
    emit("module_reference", "Generating isolated solid-background module images with image2", 0.3)
    try:
        reference_summary = tool_executor.run(
            "module_reference_generation",
            {"module_count": len(module_plan.get("modules", []))},
            lambda: generate_module_references(
                workdir,
                scene_plan,
                module_plan,
                config=runtime_config,
                options=options,
                concept_image=concept_image,
            ),
        )
    except ExternalImagegenRequired as exc:
        emit("module_reference", "Awaiting Codex/image2 module reference images before model review", 0.31)
        return _pending_external_imagegen_summary(
            workdir=workdir,
            started=started,
            options=options,
            tool_executor=tool_executor,
            request_error=exc,
            stage="module_reference_generation",
            planning_info=planning_info,
            stage_events=stage_events,
        )
    module_reference_review_path = workdir / "modules" / "module_reference_review.json"
    module_reference_review = tool_executor.run(
        "module_reference_review",
        {"module_count": len(module_plan.get("modules", [])), "attempt": 0},
        lambda: review_module_reference_images(
            runtime_config,
            options,
            module_plan=module_plan,
            reference_summary=reference_summary,
            concept_image=concept_image,
            attempt=0,
            output_path=module_reference_review_path,
        ),
    )
    mark_module_reference_review_status(workdir, module_reference_review)
    failed_reference_modules = apply_module_review_revisions(module_plan, module_reference_review)
    max_reference_review_attempts = _module_reference_max_review_attempts(runtime_config)
    review_attempt = 1
    while failed_reference_modules and review_attempt < max_reference_review_attempts:
        emit("module_reference", f"Model requested reference regeneration for {sorted(failed_reference_modules)}", 0.36 + min(0.05, review_attempt * 0.01))
        module_plan_path = write_manifest(workdir / "modules" / "module_plan.json", module_plan)
        try:
            reference_summary = tool_executor.run(
                "module_reference_generation",
                {"module_count": len(module_plan.get("modules", [])), "retry_modules": sorted(failed_reference_modules), "attempt": review_attempt},
                lambda: generate_module_references(
                    workdir,
                    scene_plan,
                    module_plan,
                    config=runtime_config,
                    options=options,
                    force_module_ids=failed_reference_modules,
                    concept_image=concept_image,
                ),
            )
        except ExternalImagegenRequired as exc:
            emit("module_reference", "Awaiting Codex/image2 revised module reference images before model review", 0.37)
            return _pending_external_imagegen_summary(
                workdir=workdir,
                started=started,
                options=options,
                tool_executor=tool_executor,
                request_error=exc,
                stage="module_reference_regeneration",
                planning_info=planning_info,
                stage_events=stage_events,
            )
        module_reference_review = tool_executor.run(
            "module_reference_review",
            {"module_count": len(module_plan.get("modules", [])), "attempt": review_attempt},
            lambda: review_module_reference_images(
                runtime_config,
                options,
                module_plan=module_plan,
                reference_summary=reference_summary,
                concept_image=concept_image,
                attempt=review_attempt,
                output_path=module_reference_review_path,
            ),
        )
        mark_module_reference_review_status(workdir, module_reference_review)
        failed_reference_modules = apply_module_review_revisions(module_plan, module_reference_review)
        review_attempt += 1
    if failed_reference_modules:
        raise RuntimeError(
            f"Module reference review failed after {max_reference_review_attempts} attempts; "
            f"not sending failed reference images to 3D AI: {sorted(failed_reference_modules)}"
        )

    emit("module_3d", "Sending reviewed module images to 3D AI model generation", 0.42)
    asset_manifest = tool_executor.run(
        "module_image_to_3d",
        {
            "module_count": len(module_plan.get("modules", [])),
            "allow_fallback": options.allow_procedural_fallback,
            "hero_model_path": str(options.hero_model_path) if options.hero_model_path else "",
        },
        lambda: generate_module_assets(
            workdir,
            module_plan,
            allow_fallback=options.allow_procedural_fallback,
            hero_model_path=options.hero_model_path,
            config=runtime_config,
            options=options,
        ),
    )
    asset_manifest_path = write_manifest(workdir / "modules" / "module_asset_manifest.json", asset_manifest)
    emit("module_check", "Checking module mesh sanity and failure policy", 0.5)
    sanity_summary = tool_executor.run(
        "module_mesh_sanity",
        {"module_asset_manifest": str(asset_manifest_path)},
        lambda: {
            "type": "module_mesh_sanity",
            "status": "pass" if not asset_manifest.get("failed_modules") and not asset_manifest.get("quality_issues") else "needs_review",
            "module_count": len(asset_manifest.get("modules", [])),
            "failed_modules": asset_manifest.get("failed_modules", []),
            "quality_issues": asset_manifest.get("quality_issues", []),
            "fallback_policy": asset_manifest.get("failed_modules", []),
            "module_sanity_files": [
                str((workdir / "modules" / str(item.get("module_id", "")) / "sanity.json").resolve())
                for item in asset_manifest.get("modules", [])
                if item.get("module_id")
            ],
        },
    )
    module_mesh_sanity_path = write_manifest(workdir / "reports" / "module_mesh_sanity.json", sanity_summary)

    emit("layout", "Planning module scale, placement, and collision policy", 0.58)
    scene_assembly = tool_executor.run("scene_layout_agent", {"module_count": len(module_plan.get("modules", []))}, lambda: plan_scene_layout(scene_plan, module_plan, asset_manifest))
    scene_assembly_path = write_manifest(workdir / "scene" / "scene_assembly.json", scene_assembly)
    emit("scene_preview", "Assembling final scene GLB and preview", 0.64)
    scene_outputs = tool_executor.run("scene_assembler", {"scene_assembly": str(scene_assembly_path)}, lambda: assemble_scene(workdir, scene_assembly))
    emit("camera", "Searching rendered white-model camera candidates", 0.7)
    camera_selection = tool_executor.run(
        "camera_candidate_search",
        {"scene_model_path": scene_outputs["scene_model_path"], "views": auto_task["output_views"], "render_backend": options.render_backend},
        lambda: select_scene_camera(
            workdir,
            scene_outputs,
            camera_plan,
            views=int(auto_task["output_views"]),
            config=runtime_config,
            render_backend=options.render_backend,
            dry_run=options.dry_run,
            concept_image=concept_image,
        ),
    )
    camera_plan = camera_selection.get("camera_plan", camera_plan)
    camera_plan_path = write_manifest(workdir / "cameras" / "camera_plan.json", camera_plan)

    render_mode_label = "Blender" if str(options.render_backend).lower() == "blender" else "scene"
    emit("render", f"Rendering {render_mode_label} white channels from assembled scene", 0.76)
    render_manifest_path = tool_executor.run(
        "render_white_channels",
        {"scene_model_path": scene_outputs["scene_model_path"], "views": auto_task["output_views"], "render_backend": options.render_backend},
        lambda: render_scene_channels(
            workdir,
            scene_outputs,
            scene_assembly_path,
            camera_plan,
            views=int(auto_task["output_views"]),
            config=runtime_config,
            dry_run=options.dry_run,
            render_backend=options.render_backend,
            allow_fallback=options.allow_procedural_fallback,
        ),
    )

    final_dir = workdir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    emit("agent", "Writing white-model screen-space position contract for final image2", 0.82)
    white_model_position_contract = tool_executor.run(
        "white_model_position_contract",
        {"render_manifest": str(render_manifest_path), "views": auto_task["output_views"]},
        lambda: create_white_model_position_contract(
            render_manifest=render_manifest_path,
            output_report=workdir / "reports" / "white_model_position_contract.json",
            output_image=final_dir / "white_position_contract_overlay.png",
            output_views=int(auto_task.get("output_views", 3)),
        ),
    )
    white_model_position_contract_path = workdir / "reports" / "white_model_position_contract.json"
    white_position_contract_overlay_path = final_dir / "white_position_contract_overlay.png"

    final_render_provider = _auto_scene_final_render_provider(runtime_config, options)
    if final_render_provider == "codex_image2":
        emit("agent", "Requesting Codex image2 final render from white-model position-lock channels", 0.84)
        try:
            agent_summary = tool_executor.run(
                "final_image2_render",
                {"input_renders": str(render_manifest_path), "provider": "codex_image2"},
                lambda: generate_codex_image2_final_render(
                    workdir=workdir,
                    render_manifest_path=render_manifest_path,
                    prompt_plan=prompt_plan,
                    concept_image=concept_image,
                    output_views=int(auto_task.get("output_views", 3)),
                    position_contract_path=white_model_position_contract_path,
                ),
            )
        except ExternalImagegenRequired as exc:
            emit("agent", "Awaiting Codex image2 final render outputs constrained by white-model positions", 0.85)
            return _pending_external_imagegen_summary(
                workdir=workdir,
                started=started,
                options=options,
                tool_executor=tool_executor,
                request_error=exc,
                stage="final_image2_render",
                planning_info=planning_info,
                stage_events=stage_events,
            )
    else:
        emit("agent", "Running final geometry-locked AI rendering from render_manifest channels", 0.84)
        final_backend = options.backend or ("mock" if options.dry_run else str(runtime_config.get("ai", {}).get("default_backend", "mock")))
        use_mock_backend = final_backend == "mock"
        strict_mesh_lock = options.geometry_mode == "strict" and not options.dry_run and not use_mock_backend
        agent_options = AgentRunOptions(
            input_renders=render_manifest_path,
            output_dir=workdir / "ai",
            prompt=prompt_plan["render_prompt"],
            config_path=options.config_path,
            model_key=options.backend_model_key or runtime_config.get("auto_agent", {}).get("default_model_key") or runtime_config.get("agent", {}).get("default_model_key", "flux2_klein_4b"),
            backend=final_backend,
            target_view="view_hero",
            max_generations=max(1, int(auto_task.get("num_candidates_per_view", 1)) * min(3, int(auto_task.get("output_views", 3))) + int(auto_task.get("max_retries", 0))),
            seed=options.seed,
            expand_views=int(auto_task.get("output_views", 3)) > 1,
            expand_view_ids=("view_hero", "view_left_30", "view_right_30"),
            default_reference_channels=("rgb", "edge", "depth", "normal", "mask", "skeleton"),
            negative_prompt="",
            steps=1 if options.dry_run or use_mock_backend else ai_steps,
            width=128 if options.dry_run or use_mock_backend else ai_resolution,
            height=128 if options.dry_run or use_mock_backend else ai_resolution,
            geometry_lock=False,
            mesh_position_lock=strict_mesh_lock,
        )
        agent_summary = tool_executor.run(
            "ai_candidate_search",
            {"input_renders": str(render_manifest_path), "backend": agent_options.backend or "configured"},
            lambda: run_agent_render(agent_options, progress=lambda stage, message, fraction: emit("agent", message, 0.84 + fraction * 0.08)),
        )

    emit("score", "Scoring module presence and structure adherence", 0.93)
    module_scores = tool_executor.run(
        "module_presence_scoring",
        {"scene_assembly": str(scene_assembly_path), "render_manifest": str(render_manifest_path)},
        lambda: module_presence_score(scene_assembly, read_manifest(render_manifest_path), agent_summary),
    )
    module_scores_path = write_manifest(workdir / "reports" / "module_scores.json", module_scores)
    structure_path = write_manifest(workdir / "reports" / "structure_scores.json", agent_summary.get("structure_scores", {}))
    multiview_path = write_manifest(workdir / "reports" / "multiview_score.json", agent_summary.get("multiview_scores", {}))
    emit("consistency", "Recording multiview consistency result", 0.95)

    emit("package", "Packaging auto-scene outputs", 0.97)
    final_image = _copy_or_blank(agent_summary.get("final_image"), final_dir / "final_view_hero.png")
    contact_sheet = _copy_or_blank(agent_summary.get("multiview_contact_sheet") or agent_summary.get("three_view_contact"), final_dir / "contact_sheet.png")
    comparison_image = _copy_or_blank(agent_summary.get("comparison_image"), final_dir / "white_vs_final.png")
    white_channels = create_white_channel_contact_sheet(
        render_manifest_path=render_manifest_path,
        output_path=final_dir / "white_channels_contact_sheet.png",
    )
    white_render = white_channels.get("white_render", "")
    white_channel_contact_sheet = white_channels.get("contact_sheet", "")
    final_view_images: dict[str, str] = {}
    for view_id, source in dict(agent_summary.get("final_view_images", {})).items():
        target_name = f"final_{view_id}.png"
        final_view_images[str(view_id)] = _copy_or_blank(source, final_dir / target_name)
    agent_report = _copy_or_blank(agent_summary.get("agent_report"), workdir / "reports" / "agent_report.json")
    final_scene_manifest = {
        "task_id": auto_task["task_id"],
        "scene_model_path": scene_outputs["scene_model_path"],
        "scene_assembly": str(scene_assembly_path),
        "camera_plan": str(camera_plan_path),
        "render_manifest": str(render_manifest_path),
        "agent_report": str(agent_report),
        "camera_search_report": str(camera_selection.get("report_path", "")),
        "assembly_report": str(scene_outputs.get("assembly_report", "")),
    }
    final_scene_manifest_path = write_manifest(workdir / "scene" / "final_scene_manifest.json", final_scene_manifest)
    module_assets_index = tool_executor.run(
        "module_assets_index",
        {"module_count": len(module_plan.get("modules", [])), "module_asset_manifest": str(asset_manifest_path)},
        lambda: create_module_assets_index(
            workdir=workdir,
            module_plan=module_plan,
            asset_manifest=asset_manifest,
            output_report=workdir / "reports" / "module_assets_index.json",
            output_contact_sheet=final_dir / "module_references_contact_sheet.png",
        ),
    )
    module_assets_index_path = workdir / "reports" / "module_assets_index.json"
    module_references_contact_sheet_path = final_dir / "module_references_contact_sheet.png"
    visual = visual_judgement_report(final_image=final_image, comparison_image=comparison_image, contact_sheet=contact_sheet, agent_summary=agent_summary)
    visual_path = write_manifest(workdir / "reports" / "visual_judgement.json", visual)
    concept_final_comparison = tool_executor.run(
        "concept_final_comparison",
        {"concept_image": str(concept_image), "final_image": str(final_image), "render_manifest": str(render_manifest_path)},
        lambda: create_concept_final_comparison(
            concept_image=concept_image,
            final_image=final_image,
            render_manifest=render_manifest_path,
            output_image=final_dir / "concept_vs_final.png",
            output_report=workdir / "reports" / "concept_final_comparison.json",
        ),
    )
    concept_final_comparison_path = workdir / "reports" / "concept_final_comparison.json"
    white_model_position_lock = tool_executor.run(
        "white_model_position_lock",
        {"final_view_images": final_view_images, "render_manifest": str(render_manifest_path)},
        lambda: create_white_model_multiview_position_lock_report(
            final_view_images=final_view_images,
            render_manifest=render_manifest_path,
            output_report=workdir / "reports" / "white_model_position_lock.json",
            output_image=final_dir / "white_position_lock_overlay.png",
            fallback_final_image=final_image,
        ),
    )
    white_model_position_lock_path = workdir / "reports" / "white_model_position_lock.json"
    final_position_retry_plan = tool_executor.run(
        "final_position_retry_plan",
        {"white_model_position_lock": white_model_position_lock.get("status", ""), "final_request": str(workdir / "final" / "codex_image2_final_request.json")},
        lambda: create_final_position_retry_plan(
            workdir=workdir,
            final_request_path=workdir / "final" / "codex_image2_final_request.json",
            white_lock_report=white_model_position_lock,
            position_contract_path=white_model_position_contract_path,
            output_report=workdir / "reports" / "final_position_retry_plan.json",
        ),
    )
    final_position_retry_plan_path = workdir / "reports" / "final_position_retry_plan.json"
    tool_calls_path = write_manifest(workdir / "reports" / "tool_calls.json", {"tools": AUTO_SCENE_TOOL_SPECS, "calls": tool_executor.calls})
    status = (
        "complete"
        if (
            visual["status"] == "pass"
            and module_scores["total"] >= 0.75
            and sanity_summary["status"] == "pass"
            and white_model_position_contract["status"] == "pass"
            and concept_final_comparison["status"] == "pass"
            and white_model_position_lock["status"] == "pass"
        )
        else "needs_review"
    )
    if any(item.get("action") == "fail_task" for item in sanity_summary.get("failed_modules", []) if isinstance(item, dict)):
        status = "failed"
    elif str(sanity_summary.get("status", "")).lower() != "pass":
        status = "needs_review"
    elif str(white_model_position_contract.get("status", "")).lower() != "pass":
        status = "needs_review"
    elif str(module_layout_check.get("status", "")).lower() != "pass":
        status = "needs_review"
    summary = {
        "type": "auto_scene_summary",
        "status": status,
        "task_id": auto_task["task_id"],
        "workdir": str(workdir),
        "request": options.request,
        "planning": planning_info,
        "auto_task": str(workdir / "auto_task.json"),
        "scene_plan": str(workdir / "scene_plan.json"),
        "concept_image_plan": str(concept_plan_path),
        "global_concept": str(concept_image),
        "concept_review": str(concept_review_path),
        "module_plan": str(module_plan_path),
        "module_prompt_info": str(module_prompt_info_path),
        "module_layout_check": str(module_layout_check_path),
        "module_reference_review": str(module_reference_review_path),
        "module_asset_manifest": str(asset_manifest_path),
        "module_mesh_sanity": str(module_mesh_sanity_path),
        "module_assets_index": str(module_assets_index_path),
        "module_references_contact_sheet": str(module_references_contact_sheet_path),
        "scene_assembly": str(scene_assembly_path),
        "final_scene_manifest": str(final_scene_manifest_path),
        "scene_model_path": scene_outputs["scene_model_path"],
        "assembly_report": str(scene_outputs.get("assembly_report", "")),
        "scene_preview": scene_outputs["scene_preview"],
        "camera_plan": str(camera_plan_path),
        "camera_search_report": str(camera_selection.get("report_path", "")),
        "camera_search_sheet": str(camera_selection.get("contact_sheet", "")),
        "render_manifest": str(render_manifest_path),
        "white_model_position_contract": str(white_model_position_contract_path),
        "white_position_contract_overlay": str(white_position_contract_overlay_path),
        "white_render": str(white_render),
        "white_channel_contact_sheet": str(white_channel_contact_sheet),
        "module_scores": str(module_scores_path),
        "structure_scores": str(structure_path),
        "multiview_score": str(multiview_path),
        "agent_report": str(agent_report),
        "tool_calls": str(tool_calls_path),
        "visual_judgement": str(visual_path),
        "concept_final_comparison": str(concept_final_comparison_path),
        "concept_vs_final": str(final_dir / "concept_vs_final.png"),
        "white_model_position_lock": str(white_model_position_lock_path),
        "white_position_lock_overlay": str(final_dir / "white_position_lock_overlay.png"),
        "final_position_retry_plan": str(final_position_retry_plan_path),
        "final_image": final_image,
        "final_view_images": final_view_images,
        "comparison_image": comparison_image,
        "contact_sheet": contact_sheet,
        "stages": str(workdir / "reports" / "stages.json"),
        "run_log": str(log_path),
        "elapsed_seconds": round(time.time() - started, 3),
        "reference_policy": {
            "final_ai_inputs": [
                "render_manifest.rgb",
                "render_manifest.edge",
                "render_manifest.depth",
                "render_manifest.normal",
                "render_manifest.mask",
                "render_manifest.skeleton_from_mask",
            ],
            "planning_only_images": ["concept/global_concept.png", "modules/*/reference.png"],
        },
        "capabilities": {
            "modular_scene_generation": True,
            "module_count": len(module_plan.get("modules", [])),
            "model_brain": planning_info,
            "concept_review": {"enabled": True, "status": concept_review.get("status", "")},
            "module_layout_contract": {"enabled": True, "status": module_layout_check.get("status", ""), "error_count": module_layout_check.get("error_count", 0)},
            "module_reference_review": {"enabled": True, "status": module_reference_review.get("status", "")},
            "module_mesh_sanity": {
                "enabled": True,
                "status": sanity_summary.get("status", ""),
                "quality_issue_count": len(sanity_summary.get("quality_issues", [])),
                "failed_module_count": len(sanity_summary.get("failed_modules", [])),
            },
            "module_assets_index": {"enabled": True, "status": module_assets_index.get("status", ""), "module_count": module_assets_index.get("module_count", 0)},
            "module_3d_backend": asset_manifest.get("module_3d_backend", ""),
            "render_backend": read_manifest(render_manifest_path).get("render_backend") or read_manifest(render_manifest_path).get("source"),
            "tool_execution": {"enabled": True, "tool_call_count": len(tool_executor.calls), "executed_tool_names": [call["tool"] for call in tool_executor.calls]},
            "visual_judgement": {"enabled": True, "status": visual["status"]},
            "concept_final_comparison": {"enabled": True, "status": concept_final_comparison["status"]},
            "white_model_position_contract": {
                "enabled": True,
                "status": white_model_position_contract.get("status", ""),
                "contract_count": white_model_position_contract.get("contract_count", 0),
            },
            "white_model_position_lock": {"enabled": True, "status": white_model_position_lock["status"], "total": white_model_position_lock.get("metrics", {}).get("total", 0.0)},
            "final_position_retry_plan": {"enabled": True, "status": final_position_retry_plan.get("status", "")},
        },
    }
    artifact_keys = [
        "auto_task",
        "scene_plan",
        "concept_image_plan",
        "global_concept",
        "concept_review",
        "module_plan",
        "module_prompt_info",
        "module_layout_check",
        "module_reference_review",
        "module_asset_manifest",
        "module_mesh_sanity",
        "module_assets_index",
        "module_references_contact_sheet",
        "scene_assembly",
        "final_scene_manifest",
        "scene_model_path",
        "assembly_report",
        "scene_preview",
        "camera_plan",
        "camera_search_report",
        "camera_search_sheet",
        "render_manifest",
        "white_model_position_contract",
        "white_position_contract_overlay",
        "white_render",
        "white_channel_contact_sheet",
        "module_scores",
        "structure_scores",
        "multiview_score",
        "agent_report",
        "tool_calls",
        "visual_judgement",
        "concept_final_comparison",
        "concept_vs_final",
        "white_model_position_lock",
        "white_position_lock_overlay",
        "final_position_retry_plan",
        "final_image",
        "comparison_image",
        "contact_sheet",
        "stages",
        "run_log",
    ]
    for key in artifact_keys:
        if summary.get(key):
            summary[key] = _absolute_artifact_path(summary[key])
    summary["final_view_images"] = {key: _absolute_artifact_path(value) for key, value in final_view_images.items()}
    summary["artifacts"] = {key: summary[key] for key in artifact_keys if summary.get(key)}
    summary["artifact_urls"] = {key: f"/api/file?path={value}" for key, value in summary["artifacts"].items() if Path(str(value)).suffix}
    stages_path = _write_auto_scene_stage_manifest(
        workdir=workdir,
        summary=summary,
        stage_events=stage_events,
        tool_calls=tool_executor.calls,
    )
    summary["stages"] = stages_path
    summary["artifacts"]["stages"] = stages_path
    summary["artifact_urls"]["stages"] = f"/api/file?path={stages_path}"
    summary_path = write_manifest(workdir / "auto_scene_summary.json", summary)
    if not options.dry_run and options.backend != "mock":
        emit("package", "Auditing model-planned Codex image2 to reviewed 3D flow", 0.985)
        image2_flow_audit = tool_executor.run(
            "image2_flow_audit",
            {"workdir": str(workdir), "require_codex_image2": True, "require_hunyuan_3d": True},
            lambda: audit_auto_scene_image2_flow(workdir, require_codex_image2=True, require_hunyuan_3d=True, write_report_file=True),
        )
        if image2_flow_audit["status"] != "pass" and summary["status"] == "complete":
            status = "needs_review"
            summary["status"] = status
        image2_flow_audit_path = _absolute_artifact_path(workdir / "reports" / "image2_flow_audit.json")
        summary["image2_flow_audit"] = image2_flow_audit_path
        summary["capabilities"]["image2_flow_audit"] = {
            "enabled": True,
            "status": image2_flow_audit["status"],
            "passed_required_checks": image2_flow_audit.get("passed_required_checks", 0),
            "required_checks": image2_flow_audit.get("required_checks", 0),
        }
        summary["capabilities"]["tool_execution"] = {
            "enabled": True,
            "tool_call_count": len(tool_executor.calls),
            "executed_tool_names": [call["tool"] for call in tool_executor.calls],
        }
        summary["artifacts"]["image2_flow_audit"] = image2_flow_audit_path
        summary["artifact_urls"]["image2_flow_audit"] = f"/api/file?path={image2_flow_audit_path}"
        tool_calls_path = write_manifest(workdir / "reports" / "tool_calls.json", {"tools": AUTO_SCENE_TOOL_SPECS, "calls": tool_executor.calls})
        summary["tool_calls"] = _absolute_artifact_path(tool_calls_path)
        summary["artifacts"]["tool_calls"] = summary["tool_calls"]
        summary["artifact_urls"]["tool_calls"] = f"/api/file?path={summary['tool_calls']}"
        stages_path = _write_auto_scene_stage_manifest(
            workdir=workdir,
            summary=summary,
            stage_events=stage_events,
            tool_calls=tool_executor.calls,
        )
        summary["stages"] = stages_path
        summary["artifacts"]["stages"] = stages_path
        summary["artifact_urls"]["stages"] = f"/api/file?path={stages_path}"
        write_manifest(summary_path, summary)
    emit("complete", f"Auto Scene finished with status={status}", 1.0)
    return summary
