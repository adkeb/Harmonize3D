from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .agent import AgentRunOptions, run_agent_render, summarize_agent_report
from .ai.backends import build_backend
from .ai.geometry import create_comparison_image
from .config import load_config, resolve_path
from .modelgen import generate_3d_model
from .rendering.blender import render_model_with_blender
from .scoring import score_candidates, summarize_report


Progress = Callable[[str, str, float], None]


@dataclass
class WorkflowOptions:
    prompt: str
    workdir: Path
    source_mode: str = "existing_mesh"
    model_path: Path | None = None
    reference_image: Path | None = None
    model_backend: str = "procedural-crystal"
    config_path: Path = Path("configs/local.json")
    views: int = 1
    render_resolution: int = 1536
    render_samples: int | None = None
    ai_width: int = 1536
    ai_height: int = 1536
    candidates: int = 1
    seed: int = 20260610
    steps: int = 42
    guidance_scale: float = 8.0
    canny_scale: float = 2.85
    depth_scale: float = 0.55
    geometry_lock: bool = True
    negative_prompt: str = ""
    model_key: str = "sdxl_controlnet_geometry"
    agent_render: bool = False
    agent_max_generations: int = 10
    agent_target_view: str = "view_05"
    agent_expand_views: bool = True


def _direct_download_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return env


def _model_cfg(config: dict[str, Any], key: str) -> dict[str, Any]:
    model = config.get("models", {}).get(key)
    if not model:
        raise RuntimeError(f"Unknown model key: {key}")
    return model


def _run_hunyuan_shape(
    *,
    config: dict[str, Any],
    reference_image: Path,
    output_model: Path,
    metadata: Path,
    seed: int,
    progress: Progress,
    shape_overrides: dict[str, Any] | None = None,
) -> Path:
    hunyuan = dict(_model_cfg(config, "hunyuan3d_2_1_shape"))
    if shape_overrides:
        hunyuan.update(shape_overrides)
    script = Path(hunyuan.get("script", "scripts/run_hunyuan_shape_white.py"))
    command = [
        str(Path(config["system"].get("python") or ".venv/bin/python")),
        str(script),
        "--repo-root",
        str(hunyuan["repo_root"]),
        "--model-path",
        str(hunyuan["model_path"]),
        "--image",
        str(reference_image),
        "--output",
        str(output_model),
        "--metadata",
        str(metadata),
        "--steps",
        str(int(hunyuan.get("steps", 50))),
        "--guidance-scale",
        str(float(hunyuan.get("guidance_scale", 5.0))),
        "--octree-resolution",
        str(int(hunyuan.get("octree_resolution", 384))),
        "--num-chunks",
        str(int(hunyuan.get("num_chunks", 8000))),
        "--seed",
        str(seed),
    ]
    progress("source", "Running Hunyuan3D 2.1 shape generation", 0.16)
    subprocess.run(command, check=True, env=_direct_download_env())
    if not output_model.exists():
        raise RuntimeError(f"Hunyuan3D did not write the expected mesh: {output_model}")
    return output_model


def _best_ranked_image(report_path: Path) -> Path:
    with report_path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    ranked = report.get("ranked", [])
    if not ranked:
        raise RuntimeError("No AI candidates were produced or ranked.")
    return Path(ranked[0].get("ranked_copy") or ranked[0]["file"])


def run_workflow(options: WorkflowOptions, progress: Progress | None = None) -> dict[str, Any]:
    def emit(stage: str, message: str, fraction: float) -> None:
        if progress:
            progress(stage, message, fraction)

    config = load_config(options.config_path)
    workdir = Path(options.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    emit("prepare", "Preparing local workflow directories", 0.03)
    model_path: Path
    emit("source", f"Resolving 3D source: {options.source_mode}", 0.09)
    if options.source_mode == "hunyuan_reference":
        if not options.reference_image:
            raise RuntimeError("Hunyuan3D mode requires a reference image.")
        model_path = _run_hunyuan_shape(
            config=config,
            reference_image=Path(options.reference_image),
            output_model=workdir / "white_mesh.glb",
            metadata=workdir / "hunyuan_shape_metadata.json",
            seed=options.seed,
            progress=emit,
        )
    elif options.source_mode == "model_path":
        if not options.model_path:
            raise RuntimeError("Model path mode requires a .glb/.gltf/.obj/.fbx path.")
        model_path = Path(options.model_path)
        if not model_path.exists():
            raise RuntimeError(f"Model path does not exist: {model_path}")
    elif options.source_mode == "procedural":
        model_path = generate_3d_model(
            prompt=options.prompt,
            output=workdir / "generated_model.obj",
            backend=options.model_backend,
        )
    else:
        existing = options.model_path or Path("outputs/hunyuan_shape_sd_render/white_mesh.glb")
        model_path = Path(existing)
        if not model_path.exists():
            raise RuntimeError(f"Default white mesh was not found: {model_path}")
    emit("source", f"3D source ready: {model_path}", 0.24)

    emit("render", "Rendering white model channels in Blender", 0.31)
    render_cfg = config["render"]
    renders_dir = workdir / "renders"
    render_manifest = render_model_with_blender(
        model_path=model_path,
        output_dir=renders_dir,
        blender_script=resolve_path(config, config["paths"]["blender_script"]),
        blender_path=config["system"].get("blender_path") or None,
        views=options.views,
        resolution=options.render_resolution,
        engine=render_cfg["engine"],
        samples=options.render_samples or int(render_cfg["samples"]),
        camera_distance=float(render_cfg["camera_distance"]),
    )

    if options.agent_render:
        emit("agent", "Running Agent v1 render tuning", 0.62)
        agent_cfg = config.get("agent", {})
        agent_summary = run_agent_render(
            AgentRunOptions(
                input_renders=Path(render_manifest),
                output_dir=workdir / "agent",
                prompt=options.prompt,
                config_path=options.config_path,
                model_key=options.model_key or "hidream_o1_image_full",
                target_view=options.agent_target_view or str(agent_cfg.get("target_view", "view_05")),
                max_generations=options.agent_max_generations or int(agent_cfg.get("max_generations", 10)),
                seed=options.seed,
                expand_views=options.agent_expand_views,
                expand_view_ids=tuple(agent_cfg.get("expand_view_ids", ["view_01", "view_05", "view_06"])),
                default_reference_channels=tuple(agent_cfg.get("default_reference_channels", ["rgb", "edge"])),
                experimental_reference_channels=tuple(agent_cfg.get("experimental_reference_channels", [])),
                pass_threshold=float(agent_cfg.get("pass_threshold", 0.62)),
                roughness_weight=float(agent_cfg.get("roughness_weight", 0.25)),
                edge_weight=float(agent_cfg.get("edge_weight", 0.35)),
                mask_weight=float(agent_cfg.get("mask_weight", 0.25)),
                background_weight=float(agent_cfg.get("background_weight", 0.15)),
                negative_prompt=options.negative_prompt,
                steps=options.steps,
                width=options.ai_width,
                height=options.ai_height,
            ),
            progress=lambda stage, message, fraction: emit(stage, message, 0.62 + fraction * 0.34),
        )
        with Path(render_manifest).open("r", encoding="utf-8") as fh:
            render_data = json.load(fh)
        target_view_id = agent_summary.get("target_view")
        white_image = ""
        for view in render_data.get("views", []):
            if view.get("view_id") == target_view_id:
                white_image = view["files"]["rgb"]
                break
        if not white_image and render_data.get("views"):
            white_image = render_data["views"][0]["files"]["rgb"]
        emit("package", "Packaging Agent artifacts and run summary", 0.97)
        summary = {
            "type": "workflow_summary",
            "status": agent_summary["status"],
            "workdir": str(workdir),
            "prompt": options.prompt,
            "source_mode": options.source_mode,
            "model_path": str(model_path),
            "render_manifest": str(render_manifest),
            "ai_manifest": "",
            "score_report": "",
            "agent_report": agent_summary["agent_report"],
            "score_summary": summarize_agent_report(agent_summary["agent_report"]),
            "white_image": white_image,
            "final_image": agent_summary["final_image"],
            "comparison_image": agent_summary["comparison_image"],
            "three_view_contact": agent_summary.get("three_view_contact", ""),
            "agent_summary": agent_summary,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        with (workdir / "run_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        emit("complete", "Agent workflow complete", 1.0)
        return summary

    emit("ai", "Generating geometry-constrained AI render candidates", 0.62)
    model_cfg = _model_cfg(config, options.model_key)
    backend_name = model_cfg.get("backend") or config["ai"]["default_backend"]
    ai_manifest = build_backend(backend_name).generate(
        render_manifest,
        workdir / "candidates",
        prompt=options.prompt,
        negative_prompt=options.negative_prompt or model_cfg.get("negative_prompt", ""),
        candidates_per_view=options.candidates,
        seed=options.seed,
        model_ref=model_cfg.get("local_path") or "",
        model_config=model_cfg,
        device=config["ai"].get("device", "cuda:0"),
        dtype=config["ai"].get("dtype", "float16"),
        variant=model_cfg.get("variant") or config["ai"].get("variant"),
        steps=options.steps,
        guidance_scale=options.guidance_scale,
        strength=float(model_cfg.get("strength", config["ai"].get("strength", 0.68))),
        width=options.ai_width,
        height=options.ai_height,
        canny_scale=options.canny_scale,
        depth_scale=options.depth_scale,
        control_only=bool(model_cfg.get("control_only", True)),
        geometry_lock=options.geometry_lock,
    )

    emit("score", "Scoring candidates against the white model structure", 0.84)
    score_cfg = config["score"]
    score_report = score_candidates(
        render_manifest_path=render_manifest,
        ai_manifest_path=ai_manifest,
        output_dir=workdir / "score",
        edge_weight=float(score_cfg["edge_weight"]),
        mask_weight=float(score_cfg["mask_weight"]),
        prompt_weight=float(score_cfg["prompt_weight"]),
        copy_top_k=int(score_cfg["copy_top_k"]),
    )
    best_image = _best_ranked_image(score_report)
    final_image = workdir / "final.png"
    shutil.copy2(best_image, final_image)

    with Path(render_manifest).open("r", encoding="utf-8") as fh:
        render_data = json.load(fh)
    white_image = Path(render_data["views"][0]["files"]["rgb"])
    comparison = create_comparison_image(
        white_image=white_image,
        final_image=final_image,
        output=workdir / "white_vs_final.png",
    )

    emit("package", "Packaging final artifacts and run summary", 0.97)
    summary = {
        "type": "workflow_summary",
        "status": "complete",
        "workdir": str(workdir),
        "prompt": options.prompt,
        "source_mode": options.source_mode,
        "model_path": str(model_path),
        "render_manifest": str(render_manifest),
        "ai_manifest": str(ai_manifest),
        "score_report": str(score_report),
        "score_summary": summarize_report(score_report),
        "white_image": str(white_image),
        "final_image": str(final_image),
        "comparison_image": str(comparison),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    with (workdir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    emit("complete", "Workflow complete", 1.0)
    return summary
