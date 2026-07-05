from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default as email_policy
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parents[2]


def _maybe_reexec_configured_python(argv: list[str] | None = None) -> None:
    if os.environ.get("LOCAL3DAI_NO_PYTHON_REEXEC") == "1":
        return
    try:
        config_path = ROOT / "configs/local.json"
        if not config_path.exists():
            return
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configured = str(config.get("system", {}).get("python") or "")
        if not configured:
            return
        configured_python_entry = Path(configured)
        configured_python = configured_python_entry.resolve()
        current_python = Path(sys.executable).resolve()
        if configured_python.exists() and configured_python != current_python:
            env = os.environ.copy()
            env["LOCAL3DAI_NO_PYTHON_REEXEC"] = "1"
            print(f"[web] Re-executing with configured Python: {configured_python_entry}", file=sys.stderr, flush=True)
            os.execve(
                str(configured_python_entry),
                [str(configured_python_entry), "-m", "local3dai.webapp", *(argv or sys.argv[1:])],
                env,
            )
    except Exception as exc:
        print(f"[web] Python re-exec skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    _maybe_reexec_configured_python()

from .agent import AgentRunOptions, run_agent_render
from .auto_agent import AUTO_STAGE_IDS, AutoRunOptions, qwen_runtime_status, run_auto_agent
from .auto_scene import AUTO_SCENE_STAGE_IDS, AutoSceneOptions, run_auto_scene
from .ai.backends import build_backend
from .ai.geometry import create_comparison_image
from .camera import CameraState
from .config import load_config, resolve_path
from .manifest import read_manifest
from .modelgen import generate_3d_model
from .rendering.blender import render_model_with_blender
from .scoring import score_candidates, summarize_report
from .stages import planned_stage_ids, stage_definition
from .workflow import WorkflowOptions, _direct_download_env, _run_hunyuan_shape, run_workflow


WEB_ROOT = ROOT / "web"
OUTPUTS_ROOT = ROOT / "outputs"
UPLOAD_ROOT = OUTPUTS_ROOT / "web-uploads"
STAGE_ROOT = OUTPUTS_ROOT / "stage-workbench"
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


class StageError(RuntimeError):
    def __init__(self, message: str, *, status: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR, **payload: Any) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


def _now() -> float:
    return time.time()


def _stage_state(stage_id: str) -> dict[str, Any]:
    definition = stage_definition(stage_id)
    return {
        **definition,
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "progress": 0.0,
        "message": "",
        "logs": [],
        "artifacts": {},
        "warnings": [],
        "error": "",
        "retry_count": 0,
    }


def _init_stage_states(*, agent_render: bool) -> list[dict[str, Any]]:
    return [_stage_state(stage_id) for stage_id in planned_stage_ids(agent_render=agent_render)]


def _stage_map(job: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stages = job.setdefault("stages", [])
    return {stage["id"]: stage for stage in stages}


def _finish_stage(stage: dict[str, Any], status: str, timestamp: float) -> None:
    if stage.get("status") in {"complete", "failed", "skipped"}:
        return
    stage["status"] = status
    stage["finished_at"] = timestamp
    started = stage.get("started_at")
    if started is not None:
        stage["duration_seconds"] = round(timestamp - float(started), 3)


def _ensure_stage_running(stage: dict[str, Any], timestamp: float) -> None:
    if stage.get("started_at") is None:
        stage["started_at"] = timestamp
    if stage.get("status") not in {"complete", "failed", "skipped"}:
        stage["status"] = "running"


def _mark_stage_artifacts(job: dict[str, Any], summary: dict[str, Any]) -> None:
    stage_artifacts: dict[str, dict[str, str]] = {
        "source": {"model_path": summary.get("model_path", "")},
        "render": {
            "render_manifest": summary.get("render_manifest", ""),
            "white_image": summary.get("white_image", ""),
        },
        "agent": {
            "agent_report": summary.get("agent_report", ""),
            "final_image": summary.get("final_image", ""),
            "comparison_image": summary.get("comparison_image", ""),
            "three_view_contact": summary.get("three_view_contact", ""),
            "multiview_contact_sheet": summary.get("multiview_contact_sheet", ""),
        },
        "ai": {"ai_manifest": summary.get("ai_manifest", "")},
        "score": {
            "score_report": summary.get("score_report", ""),
            "final_image": summary.get("final_image", ""),
            "comparison_image": summary.get("comparison_image", ""),
        },
        "package": {
            "workdir": summary.get("workdir", ""),
            "run_summary": str(Path(summary.get("workdir", "")) / "run_summary.json") if summary.get("workdir") else "",
        },
    }
    stages = _stage_map(job)
    for stage_id, artifacts in stage_artifacts.items():
        if stage_id not in stages:
            continue
        clean = {key: value for key, value in artifacts.items() if value}
        clean.update(
            {
                f"{key}_url": _file_url(value)
                for key, value in clean.items()
                if Path(str(value)).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".json"}
            }
        )
        stages[stage_id]["artifacts"].update(clean)


def _mark_auto_stage_artifacts(job: dict[str, Any], summary: dict[str, Any]) -> None:
    stage_artifacts: dict[str, dict[str, str]] = {
        "understand": {
            "auto_summary": str(Path(summary.get("workdir", "")) / "auto_summary.json") if summary.get("workdir") else "",
            "tool_calls": summary.get("tool_calls", ""),
        },
        "expand": {"auto_task": summary.get("auto_task", ""), "prompt_plan": summary.get("prompt_plan", "")},
        "plan": {"camera_plan": summary.get("camera_plan", "")},
        "source": {"model_path": summary.get("model_path", ""), "mesh_metadata": summary.get("mesh_metadata", "")},
        "mesh_check": {"mesh_sanity": summary.get("mesh_sanity", "")},
        "camera": {"camera_plan": summary.get("camera_plan", "")},
        "render": {
            "render_manifest": summary.get("render_manifest", ""),
            "white_render": summary.get("white_render", ""),
            "white_channel_contact_sheet": summary.get("white_channel_contact_sheet", ""),
        },
        "agent": {
            "agent_report": summary.get("agent_report", ""),
            "final_image": summary.get("final_image", ""),
            "comparison_image": summary.get("comparison_image", ""),
            "contact_sheet": summary.get("contact_sheet", ""),
        },
        "score": {"scores": summary.get("scores", ""), "multiview_score": summary.get("multiview_score", "")},
        "retry": {"tool_calls": summary.get("tool_calls", "")},
        "package": {
            "final_image": summary.get("final_image", ""),
            "comparison_image": summary.get("comparison_image", ""),
            "contact_sheet": summary.get("contact_sheet", ""),
            "visual_judgement": summary.get("visual_judgement", ""),
            "run_log": summary.get("run_log", ""),
        },
        "complete": {
            "auto_summary": str(Path(summary.get("workdir", "")) / "auto_summary.json") if summary.get("workdir") else "",
            "visual_judgement": summary.get("visual_judgement", ""),
        },
    }
    stages = _stage_map(job)
    for stage_id, artifacts in stage_artifacts.items():
        if stage_id not in stages:
            continue
        clean = {key: value for key, value in artifacts.items() if value}
        clean.update(
            {
                f"{key}_url": _file_url(value)
                for key, value in clean.items()
                if Path(str(value)).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".json"}
            }
        )
        stages[stage_id]["artifacts"].update(clean)


def _mark_auto_scene_stage_artifacts(job: dict[str, Any], summary: dict[str, Any]) -> None:
    stage_artifacts: dict[str, dict[str, str]] = {
        "understand": {"auto_task": summary.get("auto_task", ""), "tool_calls": summary.get("tool_calls", "")},
        "concept": {
            "concept_image_plan": summary.get("concept_image_plan", ""),
            "global_concept": summary.get("global_concept", ""),
        },
        "decompose": {"scene_plan": summary.get("scene_plan", ""), "module_plan": summary.get("module_plan", "")},
        "module_reference": {
            "module_asset_manifest": summary.get("module_asset_manifest", ""),
            "module_assets_index": summary.get("module_assets_index", ""),
            "module_references_contact_sheet": summary.get("module_references_contact_sheet", ""),
        },
        "module_3d": {"module_asset_manifest": summary.get("module_asset_manifest", ""), "module_assets_index": summary.get("module_assets_index", "")},
        "module_check": {
            "module_asset_manifest": summary.get("module_asset_manifest", ""),
            "module_mesh_sanity": summary.get("module_mesh_sanity", ""),
            "module_assets_index": summary.get("module_assets_index", ""),
        },
        "layout": {"scene_assembly": summary.get("scene_assembly", "")},
        "scene_preview": {
            "scene_model_path": summary.get("scene_model_path", ""),
            "scene_preview": summary.get("scene_preview", ""),
            "final_scene_manifest": summary.get("final_scene_manifest", ""),
            "assembly_report": summary.get("assembly_report", ""),
        },
        "camera": {"camera_plan": summary.get("camera_plan", "")},
        "render": {"render_manifest": summary.get("render_manifest", "")},
        "agent": {
            "white_model_position_contract": summary.get("white_model_position_contract", ""),
            "white_position_contract_overlay": summary.get("white_position_contract_overlay", ""),
            "agent_report": summary.get("agent_report", ""),
        },
        "score": {
            "module_scores": summary.get("module_scores", ""),
            "structure_scores": summary.get("structure_scores", ""),
            "multiview_score": summary.get("multiview_score", ""),
        },
        "consistency": {"multiview_score": summary.get("multiview_score", "")},
        "package": {
            "final_image": summary.get("final_image", ""),
            "comparison_image": summary.get("comparison_image", ""),
            "contact_sheet": summary.get("contact_sheet", ""),
            "visual_judgement": summary.get("visual_judgement", ""),
            "white_model_position_lock": summary.get("white_model_position_lock", ""),
            "white_position_lock_overlay": summary.get("white_position_lock_overlay", ""),
            "final_position_retry_plan": summary.get("final_position_retry_plan", ""),
            "image2_flow_audit": summary.get("image2_flow_audit", ""),
            "stages": summary.get("stages", ""),
            "run_log": summary.get("run_log", ""),
        },
        "complete": {
            "auto_scene_summary": str(Path(summary.get("workdir", "")) / "auto_scene_summary.json") if summary.get("workdir") else "",
            "visual_judgement": summary.get("visual_judgement", ""),
            "contact_sheet": summary.get("contact_sheet", ""),
            "stages": summary.get("stages", ""),
        },
    }
    stages = _stage_map(job)
    for stage_id, artifacts in stage_artifacts.items():
        if stage_id not in stages:
            continue
        clean = {key: value for key, value in artifacts.items() if value}
        clean.update(
            {
                f"{key}_url": _file_url(value)
                for key, value in clean.items()
                if Path(str(value)).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".json", ".glb"}
            }
        )
        stages[stage_id]["artifacts"].update(clean)


def _json_response(handler: BaseHTTPRequestHandler, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _read_multipart_upload(handler: BaseHTTPRequestHandler, field_name: str = "file") -> tuple[str, bytes] | None:
    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", "0") or "0")
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=email_policy).parsebytes(header + handler.rfile.read(length))
    if message.get_content_type() != "multipart/form-data" or not message.get_boundary():
        raise StageError("multipart/form-data required", status=HTTPStatus.BAD_REQUEST)
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        if part.get_param("name", header="content-disposition") != field_name:
            continue
        filename = part.get_filename()
        if not filename:
            return None
        return filename, part.get_payload(decode=True) or b""
    return None


def _is_safe_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def _file_url(path: str | Path) -> str:
    return f"/api/file?path={str(path)}"


def _file_payload(path: str | Path) -> dict[str, str]:
    value = str(path)
    return {"path": value, "url": _file_url(value)}


def _read_json_file(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_channels(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value)
    if not text.strip():
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _timestamped_dir(prefix: str) -> Path:
    path = STAGE_ROOT / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _memory_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            parts = raw_value.strip().split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0])

    def mib(key: str) -> int:
        return round(values.get(key, 0) / 1024)

    return {
        "mem_total_mib": mib("MemTotal"),
        "mem_available_mib": mib("MemAvailable"),
        "swap_total_mib": mib("SwapTotal"),
        "swap_free_mib": mib("SwapFree"),
        "available_plus_swap_mib": mib("MemAvailable") + mib("SwapFree"),
    }


def _stage_exception_payload(exc: Exception) -> dict[str, Any]:
    data: dict[str, Any] = {"error": str(exc), "traceback": traceback.format_exc()}
    payload = getattr(exc, "payload", None)
    if isinstance(payload, dict):
        data.update(payload)
    return data


def _hunyuan_shape_profile(config: dict[str, Any], quality: str | None) -> tuple[str, dict[str, Any]]:
    hunyuan = _model_cfg(config, "hunyuan3d_2_1_shape")
    profiles = hunyuan.get("profiles") or {}
    selected = (quality or hunyuan.get("default_profile") or "stable").strip().lower()
    if selected not in profiles:
        selected = "stable" if "stable" in profiles else next(iter(profiles), "")
    return selected or "config", dict(profiles.get(selected, {}))


def _hunyuan_attempt_profiles(config: dict[str, Any], quality: str | None) -> list[tuple[str, dict[str, Any]]]:
    selected_name, selected = _hunyuan_shape_profile(config, quality)
    attempts = [(selected_name, selected)]
    if selected_name != "stable":
        stable = (config.get("models", {}).get("hunyuan3d_2_1_shape", {}).get("profiles") or {}).get("stable")
        if stable:
            attempts.append(("stable", dict(stable)))
    return attempts


def _model_cfg(config: dict[str, Any], key: str) -> dict[str, Any]:
    model = config.get("models", {}).get(key)
    if not model:
        raise RuntimeError(f"Unknown model key: {key}")
    return model


def _materialize_model_for_web(model_path: Path, workdir: Path) -> Path:
    model_path = model_path.expanduser().resolve()
    if not model_path.exists():
        raise RuntimeError(f"Model path does not exist: {model_path}")
    if _is_safe_path(model_path):
        return model_path
    target = workdir / f"source_model{model_path.suffix.lower()}"
    shutil.copy2(model_path, target)
    return target


def _generate_hidream_reference(
    *,
    config: dict[str, Any],
    prompt: str,
    output: Path,
    seed: int,
    width: int = 1024,
    height: int = 1024,
) -> Path:
    model = _model_cfg(config, "hidream_o1_image_full")
    repo_root = Path(model["repo_root"])
    model_path = Path(model["local_path"])
    if not repo_root.exists():
        raise RuntimeError(f"HiDream repo_root does not exist: {repo_root}")
    if not model_path.exists():
        raise RuntimeError(f"HiDream model path does not exist: {model_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(Path(config["system"].get("python") or ".venv/bin/python")),
        str(repo_root / "inference.py"),
        "--model_path",
        str(model_path),
        "--prompt",
        prompt,
        "--output_image",
        str(output),
        "--height",
        str(height),
        "--width",
        str(width),
        "--model_type",
        str(model.get("model_type", "full")),
        "--seed",
        str(seed),
        "--shift",
        str(float(model.get("shift", 3.0))),
        "--guidance_scale",
        str(float(model.get("guidance_scale", 2.2))),
    ]
    subprocess.run(command, cwd=repo_root, env=_direct_download_env(), check=True)
    if not output.exists():
        raise RuntimeError(f"HiDream did not write the expected reference image: {output}")
    return output


def _generate_prompt_reference(
    *,
    config: dict[str, Any],
    prompt: str,
    output: Path,
    seed: int,
    width: int = 1024,
    height: int = 1024,
    model_key: str | None = None,
) -> Path:
    ref_cfg = config.get("reference_generation", {})
    selected_key = model_key or ref_cfg.get("default_model_key") or "flux2_klein_4b"
    model = dict(_model_cfg(config, selected_key))
    backend_name = model.get("backend", "")
    if backend_name == "hidream-o1-image":
        if not model.get("enabled", False):
            raise RuntimeError(f"Prompt reference model is disabled: {selected_key}")
        return _generate_hidream_reference(
            config=config,
            prompt=prompt,
            output=output,
            seed=seed,
            width=width,
            height=height,
        )
    backend = build_backend(backend_name)
    generate_reference = getattr(backend, "generate_reference", None)
    if not callable(generate_reference):
        raise RuntimeError(f"Backend {backend_name!r} cannot generate prompt reference images.")
    return generate_reference(
        output,
        prompt=prompt,
        seed=seed,
        model_ref=model.get("local_path") or model.get("model_path") or model.get("model_id", ""),
        negative_prompt=model.get("negative_prompt", ""),
        device=config.get("ai", {}).get("device", "cuda:0"),
        dtype=model.get("dtype") or config.get("ai", {}).get("dtype", "bfloat16"),
        variant=model.get("variant") or config.get("ai", {}).get("variant"),
        steps=int(model.get("steps", 4)),
        guidance_scale=float(model.get("guidance_scale", 0.0)),
        width=width or int(model.get("width", 1024)),
        height=height or int(model.get("height", 1024)),
    )


def _run_hunyuan_shape_for_stage(
    *,
    config: dict[str, Any],
    reference_image: Path,
    output_model: Path,
    metadata: Path,
    seed: int,
    source_mode: str,
    workdir: Path,
    profiles: list[tuple[str, dict[str, Any]]],
) -> Path:
    attempts: list[dict[str, Any]] = []
    for index, (profile_name, overrides) in enumerate(profiles or [("config", {})]):
        try:
            return _run_hunyuan_shape(
                config=config,
                reference_image=reference_image,
                output_model=output_model,
                metadata=metadata,
                seed=seed,
                progress=lambda *_: None,
                shape_overrides=overrides,
            )
        except subprocess.CalledProcessError as exc:
            oom_like = exc.returncode in {-9, 137}
            attempts.append(
                {
                    "profile": profile_name,
                    "returncode": exc.returncode,
                    "error_type": "hunyuan_oom" if oom_like else "hunyuan_shape_failed",
                    "parameters": overrides,
                    "memory": _memory_snapshot(),
                }
            )
            if oom_like and index + 1 < len(profiles):
                continue
            error_type = "hunyuan_oom" if oom_like else "hunyuan_shape_failed"
            message = (
                "Hunyuan3D 2.1 已启动，但进程被系统终止，通常是 WSL 内存或 Swap 不足。"
                if oom_like
                else f"Hunyuan3D 2.1 shape generation failed with exit code {exc.returncode}."
            )
            raise StageError(
                message,
                stage="3d",
                source_mode=source_mode,
                error_type=error_type,
                returncode=exc.returncode,
                workdir=str(workdir),
                reference_image=str(reference_image),
                reference_image_url=_file_url(reference_image),
                preprocessed_image=str(output_model.with_name("shape_input_preprocessed.png")),
                preprocessed_image_url=_file_url(output_model.with_name("shape_input_preprocessed.png")),
                hunyuan_attempts=attempts,
                memory=_memory_snapshot(),
                next_action=(
                    "参考图已保留。请关闭占用内存的进程或提高 WSL memory/swap 后重试；"
                    "也可以临时选择“已有模型路径”验证后续白模渲染和 AI 渲染阶段。"
                ),
            ) from exc
        except Exception as exc:
            attempts.append({"profile": profile_name, "error_type": "hunyuan_shape_failed", "parameters": overrides})
            raise StageError(
                f"Hunyuan3D 2.1 shape generation failed: {exc}",
                stage="3d",
                source_mode=source_mode,
                error_type="hunyuan_shape_failed",
                workdir=str(workdir),
                reference_image=str(reference_image),
                reference_image_url=_file_url(reference_image),
                preprocessed_image=str(output_model.with_name("shape_input_preprocessed.png")),
                preprocessed_image_url=_file_url(output_model.with_name("shape_input_preprocessed.png")),
                hunyuan_attempts=attempts,
                memory=_memory_snapshot(),
            ) from exc
    raise StageError("Hunyuan3D 2.1 did not run because no shape profile was available.", stage="3d", source_mode=source_mode)


def _stage_generate_3d(payload: dict[str, Any]) -> dict[str, Any]:
    config = load_config(ROOT / "configs/local.json")
    workdir = _timestamped_dir("3d")
    source_mode = payload.get("source_mode") or "model_path"
    seed = int(payload.get("seed") or config["ai"].get("seed", 20260610))
    prompt = payload.get("prompt") or "high quality concept object reference, centered product image"
    shape_quality = payload.get("shape_quality") or config["models"]["hunyuan3d_2_1_shape"].get("default_profile", "stable")
    shape_profiles = _hunyuan_attempt_profiles(config, shape_quality)
    reference_image = ""
    metadata = ""

    if source_mode in {"model_path", "existing_mesh"}:
        if not payload.get("model_path"):
            raise RuntimeError("model_path is required.")
        model_path = _materialize_model_for_web(Path(payload["model_path"]), workdir)
    elif source_mode in {"image_3d", "hunyuan_reference"}:
        if not payload.get("reference_image"):
            raise RuntimeError("reference_image is required for image-to-3D.")
        reference_image = str(Path(payload["reference_image"]).resolve())
        metadata_path = workdir / "hunyuan_shape_metadata.json"
        model_path = _run_hunyuan_shape_for_stage(
            config=config,
            reference_image=Path(reference_image),
            output_model=workdir / "white_mesh.glb",
            metadata=metadata_path,
            seed=seed,
            source_mode=source_mode,
            workdir=workdir,
            profiles=shape_profiles,
        )
        metadata = str(metadata_path)
    elif source_mode == "prompt_3d":
        try:
            reference_path = _generate_prompt_reference(
                config=config,
                prompt=prompt,
                output=workdir / "prompt_reference.png",
                seed=seed,
                width=int(payload.get("reference_width") or 1024),
                height=int(payload.get("reference_height") or 1024),
                model_key=payload.get("reference_model_key"),
            )
        except subprocess.CalledProcessError as exc:
            raise StageError(
                f"Prompt reference image generation failed with exit code {exc.returncode}.",
                stage="3d",
                source_mode=source_mode,
                error_type="prompt_reference_failed",
                returncode=exc.returncode,
                workdir=str(workdir),
                memory=_memory_snapshot(),
            ) from exc
        except Exception as exc:
            raise StageError(
                f"Prompt reference image generation failed: {exc}",
                stage="3d",
                source_mode=source_mode,
                error_type="prompt_reference_failed",
                workdir=str(workdir),
                memory=_memory_snapshot(),
            ) from exc
        reference_image = str(reference_path)
        metadata_path = workdir / "hunyuan_shape_metadata.json"
        model_path = _run_hunyuan_shape_for_stage(
            config=config,
            reference_image=reference_path,
            output_model=workdir / "white_mesh.glb",
            metadata=metadata_path,
            seed=seed,
            source_mode=source_mode,
            workdir=workdir,
            profiles=shape_profiles,
        )
        metadata = str(metadata_path)
    elif source_mode == "procedural":
        model_path = generate_3d_model(
            prompt=prompt,
            output=workdir / "generated_model.obj",
            backend=payload.get("model_backend") or "procedural-crystal",
        )
    else:
        raise RuntimeError(f"Unknown stage source_mode: {source_mode}")

    result = {
        "status": "complete",
        "stage": "3d",
        "source_mode": source_mode,
        "workdir": str(workdir),
        "model_path": str(model_path),
        "model_url": _file_url(model_path),
        "metadata": metadata,
        "shape_quality": shape_quality,
        "hunyuan_profile": shape_profiles[0][0] if metadata and shape_profiles else "",
        "hunyuan_parameters": shape_profiles[0][1] if metadata and shape_profiles else {},
    }
    metadata_data = _read_json_file(metadata) if metadata else {}
    if metadata_data.get("preprocessed_image"):
        result["preprocessed_image"] = metadata_data["preprocessed_image"]
        result["preprocessed_image_url"] = _file_url(metadata_data["preprocessed_image"])
    if metadata_data.get("mesh_sanity"):
        result["mesh_sanity"] = metadata_data["mesh_sanity"]
        if metadata_data["mesh_sanity"].get("status") == "needs_review":
            result["status"] = "needs_review"
    if reference_image:
        result["reference_image"] = reference_image
        result["reference_image_url"] = _file_url(reference_image)
    return result


def _stage_white_render(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload.get("model_path"):
        raise RuntimeError("model_path is required.")
    config = load_config(ROOT / "configs/local.json")
    workdir = _timestamped_dir("white-render")
    model_path = str(payload["model_path"])
    browser_camera_state = CameraState.from_payload(payload.get("camera"))
    camera_state = browser_camera_state.to_blender(model_path=model_path)
    render_cfg = config["render"]
    resolution = int(payload.get("resolution") or render_cfg.get("resolution", 1024))
    viewport_aspect = max(0.2, min(4.0, float(browser_camera_state.viewport_aspect or 1.0)))
    if viewport_aspect >= 1.0:
        resolution_x = resolution
        resolution_y = max(64, round(resolution / viewport_aspect))
    else:
        resolution_x = max(64, round(resolution * viewport_aspect))
        resolution_y = resolution
    render_manifest = render_model_with_blender(
        model_path=Path(model_path),
        output_dir=workdir / "renders",
        blender_script=resolve_path(config, config["paths"]["blender_script"]),
        blender_path=config["system"].get("blender_path") or None,
        views=1,
        resolution=resolution,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        engine=payload.get("engine") or render_cfg.get("engine", "CYCLES"),
        samples=int(payload.get("samples") or render_cfg.get("samples", 64)),
        camera_distance=float(render_cfg.get("camera_distance", 3.2)),
        camera=camera_state.to_dict(),
    )
    manifest = read_manifest(render_manifest)
    view = manifest["views"][0]
    files = view["files"]
    channel_urls = {channel: _file_url(path) for channel, path in files.items()}
    return {
        "status": "complete",
        "stage": "white-render",
        "workdir": str(workdir),
        "model_path": model_path,
        "render_manifest": str(render_manifest),
        "render_manifest_url": _file_url(render_manifest),
        "white_image": files["rgb"],
        "white_image_url": _file_url(files["rgb"]),
        "channel_urls": channel_urls,
        "camera": browser_camera_state.to_dict(),
        "blender_camera": camera_state.to_dict(),
        "resolution": {"width": resolution_x, "height": resolution_y},
    }


def _best_ranked_image(report_path: Path) -> Path:
    report = read_manifest(report_path)
    ranked = report.get("ranked", [])
    if not ranked:
        raise RuntimeError("No AI candidates were produced or ranked.")
    return Path(ranked[0].get("ranked_copy") or ranked[0]["file"])


def _stage_ai_render(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload.get("render_manifest"):
        raise RuntimeError("render_manifest is required.")
    forbidden = sorted(set(payload) & {"reference_image", "input_image", "source_image", "preprocessed_image"})
    if forbidden:
        raise StageError(
            "AI render only accepts a Blender render_manifest; raw 3D input images are not valid AI references.",
            status=HTTPStatus.BAD_REQUEST,
            stage="ai-render",
            forbidden_fields=forbidden,
        )
    config = load_config(ROOT / "configs/local.json")
    workdir = _timestamped_dir("ai-render")
    render_manifest = Path(payload["render_manifest"])
    render_data = read_manifest(render_manifest)
    source_model_path = str(render_data.get("source", ""))
    prompt = payload.get("prompt") or "high quality clean product render, smooth studio lighting"
    negative_prompt = payload.get("negative_prompt") or ""
    agent_render = bool(payload.get("agent_render", True))
    seed = int(payload.get("seed") or config["ai"].get("seed", 20260610))

    if agent_render:
        agent_cfg = config.get("agent", {})
        model_key = payload.get("model_key") or agent_cfg.get("default_model_key") or config["ai"].get("default_model_key", "flux2_klein_4b")
        model_cfg = dict(_model_cfg(config, model_key))
        summary = run_agent_render(
            AgentRunOptions(
                input_renders=render_manifest,
                output_dir=workdir / "agent",
                prompt=prompt,
                config_path=ROOT / "configs/local.json",
                model_key=model_key,
                backend=payload.get("backend") or None,
                target_view="view_locked",
                max_generations=int(payload.get("max_generations") or agent_cfg.get("max_generations", 10)),
                seed=seed,
                expand_views=bool(payload.get("expand_views", True)),
                expand_view_ids=tuple(agent_cfg.get("expand_view_ids", ["view_locked", "view_left_30", "view_right_30"])),
                default_reference_channels=tuple(
                    _parse_channels(payload.get("reference_channels"))
                    if "reference_channels" in payload
                    else agent_cfg.get("default_reference_channels", ["rgb", "edge", "depth", "normal", "mask", "skeleton"])
                ),
                experimental_reference_channels=tuple(agent_cfg.get("experimental_reference_channels", [])),
                pass_threshold=float(agent_cfg.get("pass_threshold", 0.62)),
                roughness_weight=float(agent_cfg.get("roughness_weight", 0.25)),
                edge_weight=float(agent_cfg.get("edge_weight", 0.35)),
                mask_weight=float(agent_cfg.get("mask_weight", 0.25)),
                background_weight=float(agent_cfg.get("background_weight", 0.15)),
                negative_prompt=negative_prompt,
                steps=int(payload.get("steps") or model_cfg.get("steps", config["ai"].get("steps", 4))),
                width=int(payload.get("width") or model_cfg.get("width", config["ai"].get("width", 1024))),
                height=int(payload.get("height") or model_cfg.get("height", config["ai"].get("height", 1024))),
            )
        )
        return {
            "status": summary["status"],
            "stage": "ai-render",
            "workdir": str(workdir),
            "render_manifest": str(render_manifest),
            "source_model_path": source_model_path,
            "final_image": summary["final_image"],
            "final_image_url": _file_url(summary["final_image"]),
            "comparison_image": summary["comparison_image"],
            "comparison_image_url": _file_url(summary["comparison_image"]),
            "agent_report": summary["agent_report"],
            "agent_report_url": _file_url(summary["agent_report"]),
            "multiview_contact_sheet": summary.get("multiview_contact_sheet", ""),
            "multiview_contact_sheet_url": _file_url(summary["multiview_contact_sheet"]) if summary.get("multiview_contact_sheet") else "",
            "final_view_images": summary.get("final_view_images", {}),
            "summary": summary,
        }

    model_key = payload.get("model_key") or config["ai"].get("default_model_key", "flux2_klein_4b")
    model_cfg = dict(_model_cfg(config, model_key))
    backend_name = payload.get("backend") or model_cfg.get("backend") or config["ai"]["default_backend"]
    ai_manifest = build_backend(backend_name).generate(
        render_manifest,
        workdir / "candidates",
        prompt=prompt,
        negative_prompt=negative_prompt or model_cfg.get("negative_prompt", ""),
        candidates_per_view=int(payload.get("candidates") or 1),
        seed=seed,
        model_ref=model_cfg.get("local_path") or model_cfg.get("model_path") or "",
        model_config=model_cfg,
        device=config["ai"].get("device", "cuda:0"),
        dtype=model_cfg.get("dtype") or config["ai"].get("dtype", "bfloat16"),
        variant=model_cfg.get("variant") or config["ai"].get("variant"),
        steps=int(payload.get("steps") or model_cfg.get("steps", config["ai"].get("steps", 4))),
        guidance_scale=float(payload.get("guidance_scale") or model_cfg.get("guidance_scale", config["ai"].get("guidance_scale", 1.0))),
        strength=float(model_cfg.get("strength", config["ai"].get("strength", 0.68))),
        width=int(payload.get("width") or model_cfg.get("width", config["ai"].get("width", 1024))),
        height=int(payload.get("height") or model_cfg.get("height", config["ai"].get("height", 1024))),
        canny_scale=float(model_cfg.get("canny_scale", 2.85)),
        depth_scale=float(model_cfg.get("depth_scale", 0.55)),
        control_channels=list(model_cfg.get("control_channels", ["canny", "depth"])),
        reference_channels=(
            _parse_channels(payload.get("reference_channels"))
            if "reference_channels" in payload
            else list(model_cfg.get("reference_channels", []))
        ),
        control_only=bool(model_cfg.get("control_only", True)),
        geometry_lock=bool(model_cfg.get("geometry_lock", True)),
    )
    score_cfg = config["score"]
    score_report = score_candidates(
        render_manifest_path=render_manifest,
        ai_manifest_path=ai_manifest,
        output_dir=workdir / "score",
        edge_weight=float(score_cfg["edge_weight"]),
        mask_weight=float(score_cfg["mask_weight"]),
        prompt_weight=float(score_cfg["prompt_weight"]),
        copy_top_k=int(score_cfg["copy_top_k"]),
        version=str(score_cfg.get("version", "legacy")),
        structure_weights=dict(score_cfg.get("structure_v2", {})),
    )
    final_image = workdir / "final.png"
    shutil.copy2(_best_ranked_image(score_report), final_image)
    white_image = render_data["views"][0]["files"]["rgb"]
    comparison = create_comparison_image(white_image=white_image, final_image=final_image, output=workdir / "white_vs_final.png")
    return {
        "status": "complete",
        "stage": "ai-render",
        "workdir": str(workdir),
        "render_manifest": str(render_manifest),
        "source_model_path": source_model_path,
        "ai_manifest": str(ai_manifest),
        "score_report": str(score_report),
        "score_summary": summarize_report(score_report),
        "final_image": str(final_image),
        "final_image_url": _file_url(final_image),
        "comparison_image": str(comparison),
        "comparison_image_url": _file_url(comparison),
    }


def _model_status(config: dict[str, Any]) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for key, model in config.get("models", {}).items():
        paths: list[str] = []
        for path_key in ("local_path", "base_model", "canny_controlnet", "depth_controlnet", "repo_root", "model_path"):
            value = model.get(path_key)
            if value:
                paths.append(str(value))
        status[key] = {
            "backend": model.get("backend", ""),
            "enabled": bool(model.get("enabled", False)),
            "paths": [{"path": path, "exists": Path(path).exists()} for path in paths],
        }
    return status


def _latest_artifacts() -> dict[str, str]:
    candidates = [
        OUTPUTS_ROOT / "hunyuan_shape_sd_render" / "white_vs_geometry_locked_render.png",
        OUTPUTS_ROOT / "hunyuan_shape_sd_render" / "final_geometry_locked_render.png",
        OUTPUTS_ROOT / "hunyuan_shape_sd_render" / "white_model_reference.png",
    ]
    result: dict[str, str] = {}
    labels = ("comparison_image", "final_image", "white_image")
    for label, path in zip(labels, candidates):
        if path.exists():
            result[label] = str(path)
            result[f"{label}_url"] = _file_url(path)
    web_runs = sorted((OUTPUTS_ROOT / "web-runs").glob("run-*/run_summary.json")) if (OUTPUTS_ROOT / "web-runs").exists() else []
    if web_runs:
        try:
            latest = json.loads(web_runs[-1].read_text(encoding="utf-8"))
            for key in ("comparison_image", "final_image", "white_image"):
                if latest.get(key) and Path(latest[key]).exists():
                    result[key] = latest[key]
                    result[f"{key}_url"] = _file_url(latest[key])
        except Exception:
            pass
    return result


def _job_snapshot(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = dict(JOBS.get(job_id, {}))
        job["logs"] = list(job.get("logs", []))
        job["stages"] = [dict(stage, logs=list(stage.get("logs", [])), artifacts=dict(stage.get("artifacts", {}))) for stage in job.get("stages", [])]
        return job


def _append_log(job_id: str, stage: str, message: str, progress: float) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        timestamp = _now()
        stages = _stage_map(job)
        current_stage_id = job.get("stage")
        if current_stage_id and current_stage_id != stage and current_stage_id in stages:
            _finish_stage(stages[current_stage_id], "complete", timestamp)
        if stage not in stages:
            stages[stage] = _stage_state(stage)
            job["stages"] = list(stages.values())
        current = stages[stage]
        _ensure_stage_running(current, timestamp)
        current["progress"] = progress
        current["message"] = message
        current.setdefault("logs", []).append(
            {
                "time": time.strftime("%H:%M:%S"),
                "message": message,
                "progress": progress,
            }
        )
        current["logs"] = current["logs"][-6:]
        if stage == "complete":
            _finish_stage(current, "complete", timestamp)
        job["stage"] = stage
        job["progress"] = progress
        job.setdefault("logs", []).append(
            {
                "time": time.strftime("%H:%M:%S"),
                "stage": stage,
                "message": message,
                "progress": progress,
            }
        )


def _run_job(job_id: str, options: WorkflowOptions) -> None:
    try:
        summary = run_workflow(options, progress=lambda stage, message, fraction: _append_log(job_id, stage, message, fraction))
        artifacts = {
            key: summary[key]
            for key in (
                "white_image",
                "final_image",
                "comparison_image",
                "three_view_contact",
                "model_path",
                "render_manifest",
                "ai_manifest",
                "score_report",
                "agent_report",
                "workdir",
            )
            if summary.get(key)
        }
        artifacts.update({f"{key}_url": _file_url(value) for key, value in artifacts.items() if Path(str(value)).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}})
        with JOBS_LOCK:
            _mark_stage_artifacts(JOBS[job_id], summary)
            JOBS[job_id].update(
                {
                    "status": "complete",
                    "progress": 1.0,
                    "summary": summary,
                    "artifacts": artifacts,
                    "finished_at": time.time(),
                }
            )
    except Exception as exc:
        with JOBS_LOCK:
            job = JOBS[job_id]
            stages = _stage_map(job)
            failed_stage = stages.get(job.get("stage"))
            if failed_stage:
                failed_stage["message"] = str(exc)
                _finish_stage(failed_stage, "failed", _now())
            JOBS[job_id].update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "finished_at": time.time(),
                }
            )


def _init_auto_stage_states() -> list[dict[str, Any]]:
    return [_stage_state(stage_id) for stage_id in AUTO_STAGE_IDS]


def _init_auto_scene_stage_states() -> list[dict[str, Any]]:
    return [_stage_state(stage_id) for stage_id in AUTO_SCENE_STAGE_IDS]


def _run_auto_job(job_id: str, options: AutoRunOptions) -> None:
    try:
        summary = run_auto_agent(options, progress=lambda stage, message, fraction: _append_log(job_id, stage, message, fraction))
        artifacts = dict(summary.get("artifacts", {}))
        artifacts.update({f"{key}_url": _file_url(value) for key, value in artifacts.items() if Path(str(value)).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".json"}})
        final_status = "done" if summary.get("status") == "complete" else summary.get("status", "needs_review")
        with JOBS_LOCK:
            _mark_auto_stage_artifacts(JOBS[job_id], summary)
            JOBS[job_id].update(
                {
                    "status": final_status,
                    "progress": 1.0,
                    "summary": summary,
                    "artifacts": artifacts,
                    "finished_at": time.time(),
                }
            )
    except Exception as exc:
        with JOBS_LOCK:
            job = JOBS[job_id]
            stages = _stage_map(job)
            failed_stage = stages.get(job.get("stage"))
            if failed_stage:
                failed_stage["message"] = str(exc)
                _finish_stage(failed_stage, "failed", _now())
            JOBS[job_id].update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "finished_at": time.time(),
                }
            )


def _run_auto_scene_job(job_id: str, options: AutoSceneOptions) -> None:
    try:
        summary = run_auto_scene(options, progress=lambda stage, message, fraction: _append_log(job_id, stage, message, fraction))
        artifacts = dict(summary.get("artifacts", {}))
        artifacts.update(
            {
                f"{key}_url": _file_url(value)
                for key, value in artifacts.items()
                if Path(str(value)).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".json", ".glb"}
            }
        )
        final_status = "done" if summary.get("status") == "complete" else summary.get("status", "needs_review")
        with JOBS_LOCK:
            _mark_auto_scene_stage_artifacts(JOBS[job_id], summary)
            JOBS[job_id].update(
                {
                    "status": final_status,
                    "progress": 1.0,
                    "summary": summary,
                    "artifacts": artifacts,
                    "finished_at": time.time(),
                }
            )
    except Exception as exc:
        with JOBS_LOCK:
            job = JOBS[job_id]
            stages = _stage_map(job)
            failed_stage = stages.get(job.get("stage"))
            if failed_stage:
                failed_stage["message"] = str(exc)
                _finish_stage(failed_stage, "failed", _now())
            JOBS[job_id].update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "finished_at": time.time(),
                }
            )


def _start_auto_job(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = f"auto-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
    workdir = OUTPUTS_ROOT / "auto" / run_id
    options = AutoRunOptions(
        request=payload.get("request") or payload.get("natural_language_request") or payload.get("prompt") or "clean product render",
        output_dir=workdir,
        config_path=ROOT / "configs/local.json",
        source_mode=payload.get("source_mode") or "auto",
        model_path=Path(payload["model_path"]) if payload.get("model_path") else None,
        reference_image=Path(payload["reference_image"]) if payload.get("reference_image") else None,
        output_views=int(payload.get("output_views") or payload.get("views") or 3),
        quality_mode=payload.get("quality_mode") or payload.get("quality") or "balanced",
        geometry_mode=payload.get("geometry_mode") or payload.get("geometry") or "strict",
        style_preset=payload.get("style_preset") or payload.get("style") or "product",
        backend_model_key=payload.get("backend_model_key") or payload.get("model_key"),
        backend=payload.get("backend"),
        num_candidates_per_view=int(payload.get("num_candidates_per_view") or payload.get("candidates") or 3),
        max_retries=int(payload.get("max_retries") or 2),
        seed=int(payload.get("seed") or 20260610),
        dry_run=bool(payload.get("dry_run") or payload.get("backend") == "mock"),
        use_llm=not bool(payload.get("no_llm")),
    )
    with JOBS_LOCK:
        JOBS[run_id] = {
            "id": run_id,
            "task_id": run_id,
            "status": "running",
            "stage": "queued",
            "progress": 0.0,
            "workdir": str(workdir),
            "logs": [],
            "stages": _init_auto_stage_states(),
            "created_at": time.time(),
            "mode": "auto_agent",
        }
    thread = threading.Thread(target=_run_auto_job, args=(run_id, options), daemon=True)
    thread.start()
    snapshot = _job_snapshot(run_id)
    return {
        "task_id": run_id,
        "status": snapshot.get("status", "running"),
        "stage": snapshot.get("stage", "queued"),
        "artifacts": snapshot.get("artifacts", {}),
        "job": snapshot,
    }


def _start_auto_scene_job(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = f"scene-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
    workdir = OUTPUTS_ROOT / "auto_scene" / run_id
    options = AutoSceneOptions(
        request=payload.get("request") or payload.get("natural_language_request") or payload.get("prompt") or "modular product scene",
        output_dir=workdir,
        config_path=ROOT / "configs/local.json",
        output_views=int(payload.get("output_views") or payload.get("views") or 3),
        quality_mode=payload.get("quality_mode") or payload.get("quality") or "balanced",
        geometry_mode=payload.get("geometry_mode") or payload.get("geometry") or "strict",
        style_preset=payload.get("style_preset") or payload.get("style") or "exhibition",
        backend_model_key=payload.get("backend_model_key") or payload.get("model_key"),
        backend=payload.get("backend"),
        num_candidates_per_view=int(payload.get("num_candidates_per_view") or payload.get("candidates") or 3),
        max_retries=int(payload.get("max_retries") or 2),
        seed=int(payload.get("seed") or 20260610),
        allow_procedural_fallback=bool(payload.get("allow_procedural_fallback", True)),
        require_concept_confirmation=bool(payload.get("require_concept_confirmation", False)),
        dry_run=bool(payload.get("dry_run") or payload.get("backend") == "mock"),
        use_llm=not bool(payload.get("no_llm")),
        render_backend=payload.get("render_backend") or "auto",
        hero_model_path=Path(payload["hero_model_path"]) if payload.get("hero_model_path") else None,
    )
    with JOBS_LOCK:
        JOBS[run_id] = {
            "id": run_id,
            "task_id": run_id,
            "status": "running",
            "stage": "queued",
            "progress": 0.0,
            "workdir": str(workdir),
            "logs": [],
            "stages": _init_auto_scene_stage_states(),
            "created_at": time.time(),
            "mode": "auto_scene",
        }
    thread = threading.Thread(target=_run_auto_scene_job, args=(run_id, options), daemon=True)
    thread.start()
    snapshot = _job_snapshot(run_id)
    return {
        "task_id": run_id,
        "status": snapshot.get("status", "running"),
        "stage": snapshot.get("stage", "queued"),
        "artifacts": snapshot.get("artifacts", {}),
        "job": snapshot,
    }


def _start_job(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
    workdir = OUTPUTS_ROOT / "web-runs" / run_id
    config = load_config(ROOT / "configs/local.json")
    agent_render = bool(payload.get("agent_render", False))
    model_key = payload.get("model_key") or (
        config.get("agent", {}).get("default_model_key")
        if agent_render
        else config.get("ai", {}).get("default_model_key")
    ) or "flux2_klein_4b"
    options = WorkflowOptions(
        prompt=payload.get("prompt") or "faceted black obsidian sci fi artifact, cyan magenta studio lighting",
        workdir=workdir,
        source_mode=payload.get("source_mode") or "existing_mesh",
        model_path=Path(payload["model_path"]) if payload.get("model_path") else None,
        reference_image=Path(payload["reference_image"]) if payload.get("reference_image") else None,
        views=int(payload.get("views") or 1),
        render_resolution=int(payload.get("render_resolution") or 1536),
        render_samples=int(payload.get("render_samples") or 64),
        ai_width=int(payload.get("ai_width") or config.get("models", {}).get(model_key, {}).get("width", 1024)),
        ai_height=int(payload.get("ai_height") or config.get("models", {}).get(model_key, {}).get("height", 1024)),
        candidates=int(payload.get("candidates") or 1),
        seed=int(payload.get("seed") or 20260610),
        steps=int(payload.get("steps") or config.get("models", {}).get(model_key, {}).get("steps", 4)),
        guidance_scale=float(payload.get("guidance_scale") or config.get("models", {}).get(model_key, {}).get("guidance_scale", 1.0)),
        canny_scale=float(payload.get("canny_scale") or 2.85),
        depth_scale=float(payload.get("depth_scale") or 0.55),
        geometry_lock=bool(payload.get("geometry_lock", True)),
        negative_prompt=payload.get("negative_prompt") or "",
        model_key=model_key,
        agent_render=agent_render,
        agent_max_generations=int(payload.get("agent_max_generations") or 10),
        agent_target_view=payload.get("agent_target_view") or config.get("agent", {}).get("target_view", "view_locked"),
        agent_expand_views=bool(payload.get("agent_expand_views", True)),
    )
    with JOBS_LOCK:
        JOBS[run_id] = {
            "id": run_id,
            "status": "running",
            "stage": "queued",
            "progress": 0.0,
            "workdir": str(workdir),
            "logs": [],
            "stages": _init_stage_states(agent_render=agent_render),
            "created_at": time.time(),
        }
    thread = threading.Thread(target=_run_job, args=(run_id, options), daemon=True)
    thread.start()
    return _job_snapshot(run_id)


class AppHandler(BaseHTTPRequestHandler):
    server_version = "Local3DAIWeb/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {format % args}")

    def _send_file_response(self, path: Path, *, include_body: bool = True) -> None:
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if include_body:
            self.wfile.write(data)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/file":
            params = parse_qs(parsed.query)
            requested = Path(params.get("path", [""])[0])
            if not requested.is_absolute():
                requested = ROOT / requested
            requested = requested.resolve()
            if not _is_safe_path(requested) or not requested.exists() or not requested.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            self._send_file_response(requested, include_body=False)
            return
        static_path = WEB_ROOT / ("index.html" if path == "/" else path.lstrip("/"))
        static_path = static_path.resolve()
        if not _is_safe_path(static_path) or not static_path.exists() or not static_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self._send_file_response(static_path, include_body=False)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/config":
                config = load_config(ROOT / "configs/local.json")
                default_model_key = config.get("ai", {}).get("default_model_key", "flux2_klein_4b")
                default_model = config["models"].get(default_model_key, config["models"]["flux2_klein_4b"])
                _json_response(
                    self,
                    {
                        "workspace": str(ROOT),
                        "outputs": str(OUTPUTS_ROOT),
                        "model_status": _model_status(config),
                        "defaults": {
                            "prompt": "same exact 3d model silhouette, faceted black obsidian bird mask, long central beak preserved, large hollow open side slots preserved, side handles preserved, angular top facets preserved, dark reflective ceramic glass, subtle cyan magenta glowing seams, studio render",
                            "negative_prompt": default_model.get("negative_prompt", ""),
                            "render_resolution": config["render"]["resolution"],
                            "ai_width": default_model.get("width", 1024),
                            "ai_height": default_model.get("height", 1024),
                            "steps": default_model.get("steps", 4),
                            "canny_scale": default_model.get("canny_scale", 0.0),
                            "depth_scale": default_model.get("depth_scale", 0.0),
                            "agent_render": config.get("agent", {}).get("enabled", True),
                            "agent_max_generations": config.get("web", {}).get(
                                "agent_max_generations", config.get("agent", {}).get("max_generations", 10)
                            ),
                            "agent_target_view": config.get("agent", {}).get("target_view", "view_locked"),
                            "shape_quality": config["models"]["hunyuan3d_2_1_shape"].get("default_profile", "stable"),
                            "shape_profiles": config["models"]["hunyuan3d_2_1_shape"].get("profiles", {}),
                            "ai_steps": config.get("web", {}).get("ai_steps", default_model.get("steps", 42)),
                            "ai_resolution": config.get("web", {}).get(
                                "ai_resolution", default_model.get("width", 1536)
                            ),
                        },
                        "latest": _latest_artifacts(),
                    },
                )
                return
            if path == "/api/jobs":
                with JOBS_LOCK:
                    jobs = sorted((dict(job) for job in JOBS.values()), key=lambda item: item.get("created_at", 0), reverse=True)
                _json_response(self, {"jobs": jobs})
                return
            if path == "/api/auto-agent/status":
                params = parse_qs(parsed.query)
                config = load_config(ROOT / "configs/local.json")
                check_hf_mirror = params.get("check_hf_mirror", ["0"])[0].lower() in {"1", "true", "yes"}
                _json_response(self, qwen_runtime_status(config, check_hf_mirror=check_hf_mirror))
                return
            if path.startswith("/api/auto-run/"):
                job_id = path.rsplit("/", 1)[-1]
                job = _job_snapshot(job_id)
                if not job:
                    _json_response(self, {"error": "job not found"}, HTTPStatus.NOT_FOUND)
                    return
                _json_response(self, job)
                return
            if path.startswith("/api/auto-scene/"):
                job_id = path.rsplit("/", 1)[-1]
                job = _job_snapshot(job_id)
                if not job:
                    _json_response(self, {"error": "job not found"}, HTTPStatus.NOT_FOUND)
                    return
                _json_response(self, job)
                return
            if path.startswith("/api/jobs/"):
                job_id = path.rsplit("/", 1)[-1]
                job = _job_snapshot(job_id)
                if not job:
                    _json_response(self, {"error": "job not found"}, HTTPStatus.NOT_FOUND)
                    return
                _json_response(self, job)
                return
            if path == "/api/file":
                params = parse_qs(parsed.query)
                requested = Path(params.get("path", [""])[0])
                if not requested.is_absolute():
                    requested = ROOT / requested
                requested = requested.resolve()
                if not _is_safe_path(requested) or not requested.exists() or not requested.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                    return
                self._send_file_response(requested)
                return

            static_path = WEB_ROOT / ("index.html" if path == "/" else path.lstrip("/"))
            static_path = static_path.resolve()
            if not _is_safe_path(static_path) or not static_path.exists() or not static_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            self._send_file_response(static_path)
        except Exception as exc:
            status = getattr(exc, "status", HTTPStatus.INTERNAL_SERVER_ERROR)
            _json_response(self, _stage_exception_payload(exc), status)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/run":
                payload = _read_json(self)
                _json_response(self, _start_job(payload), HTTPStatus.ACCEPTED)
                return
            if path == "/api/auto-run":
                payload = _read_json(self)
                _json_response(self, _start_auto_job(payload), HTTPStatus.ACCEPTED)
                return
            if path == "/api/auto-scene":
                payload = _read_json(self)
                _json_response(self, _start_auto_scene_job(payload), HTTPStatus.ACCEPTED)
                return
            if path == "/api/stage/3d":
                payload = _read_json(self)
                _json_response(self, _stage_generate_3d(payload))
                return
            if path == "/api/stage/white-render":
                payload = _read_json(self)
                _json_response(self, _stage_white_render(payload))
                return
            if path == "/api/stage/ai-render":
                payload = _read_json(self)
                _json_response(self, _stage_ai_render(payload))
                return
            if path == "/api/upload":
                upload = _read_multipart_upload(self)
                if upload is None:
                    _json_response(self, {"error": "file field is required"}, HTTPStatus.BAD_REQUEST)
                    return
                filename, data = upload
                UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
                safe_name = Path(filename).name.replace(" ", "_")
                target = UPLOAD_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe_name}"
                with target.open("wb") as fh:
                    fh.write(data)
                _json_response(self, {"path": str(target), "url": _file_url(target)})
                return
            _json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            status = getattr(exc, "status", HTTPStatus.INTERNAL_SERVER_ERROR)
            _json_response(self, _stage_exception_payload(exc), status)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if os.environ.get("LOCAL3DAI_NO_PYTHON_REEXEC") != "1":
        try:
            configured_python = Path(load_config(ROOT / "configs/local.json")["system"].get("python", "")).resolve()
            current_python = Path(sys.executable).resolve()
            if configured_python.exists() and configured_python != current_python:
                env = os.environ.copy()
                env["LOCAL3DAI_NO_PYTHON_REEXEC"] = "1"
                os.execve(str(configured_python), [str(configured_python), "-m", "local3dai.webapp", *argv], env)
        except Exception as exc:
            print(f"[web] Python re-exec skipped: {exc}", file=sys.stderr)

    parser = argparse.ArgumentParser(description="Run the Local3DAI web control panel.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7866)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Local3DAI web control panel: http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
