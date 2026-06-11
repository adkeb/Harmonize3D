from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .ai.backends import build_backend
from .config import load_config, resolve_path, write_config
from .environment import detect_environment, print_report
from .modelgen import generate_3d_model
from .rendering.blender import render_model_with_blender
from .sample import create_sample_renders
from .scoring import score_candidates, summarize_report


def _load(path: str | None) -> dict[str, Any]:
    return load_config(path or "configs/local.json")


def _model_config(config: dict[str, Any], model_key: str | None) -> dict[str, Any]:
    if not model_key:
        return {}
    model = config.get("models", {}).get(model_key)
    if not model:
        raise ValueError(f"Unknown model key in config: {model_key}")
    return model


def _model_ref(config: dict[str, Any], model_key: str | None) -> str:
    model = _model_config(config, model_key)
    if not model:
        return ""
    return model.get("local_path") or model.get("model_id") or ""


def cmd_doctor(args: argparse.Namespace) -> int:
    report = detect_environment(include_model_scan=args.scan_models)
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
        report.setdefault("tools", {})["configured_blender"] = config.get("system", {}).get("blender_path", "")
    print_report(report)
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    report = detect_environment(include_model_scan=False)
    gpu = report.get("gpu", {})
    config_path = Path(args.config)
    config = load_config(config_path) if config_path.exists() else load_config(None)
    configured_blender = config.get("system", {}).get("blender_path") or report.get("tools", {}).get("blender", "")
    config["system"].update(
        {
            "workspace": str(Path.cwd()),
            "os": f"{report['platform']['system']} {report['platform']['release']}",
            "python": report["platform"]["python_executable"],
            "gpu_name": gpu.get("name", ""),
            "gpu_vram_mb": gpu.get("memory_total_mb", 0),
            "nvidia_driver": gpu.get("driver_version", ""),
            "cuda_runtime_reported_by_driver": gpu.get("cuda_runtime_reported_by_driver", ""),
            "blender_path": configured_blender,
        }
    )
    config["paths"].update(
        {
            "models_dir": str(Path.cwd() / "models"),
            "outputs_dir": str(Path.cwd() / "outputs"),
            "blender_script": str(Path.cwd() / "blender_scripts" / "batch_render.py"),
        }
    )
    path = write_config(args.config, config, force=args.force)
    print(f"Wrote {path}")
    return 0


def cmd_sample_renders(args: argparse.Namespace) -> int:
    manifest = create_sample_renders(args.output, views=args.views, resolution=args.resolution)
    print(f"Wrote sample render manifest: {manifest}")
    return 0


def cmd_generate_3d(args: argparse.Namespace) -> int:
    config = _load(args.config)
    gen_cfg = config["model_generation"]
    model_path = generate_3d_model(
        prompt=args.prompt,
        output=args.output or gen_cfg["output_dir"],
        backend=args.backend or gen_cfg["default_backend"],
        external_command=args.external_command or gen_cfg.get("external_command", ""),
    )
    print(f"Wrote 3D model: {model_path}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    config = _load(args.config)
    render_cfg = config["render"]
    manifest = render_model_with_blender(
        model_path=args.model,
        output_dir=args.output,
        blender_script=resolve_path(config, config["paths"]["blender_script"]),
        blender_path=args.blender or config["system"].get("blender_path") or None,
        views=args.views or int(render_cfg["views"]),
        resolution=args.resolution or int(render_cfg["resolution"]),
        engine=args.engine or render_cfg["engine"],
        samples=args.samples or int(render_cfg["samples"]),
        camera_distance=args.camera_distance or float(render_cfg["camera_distance"]),
    )
    print(f"Wrote render manifest: {manifest}")
    return 0


def cmd_ai_render(args: argparse.Namespace) -> int:
    config = _load(args.config)
    ai_cfg = config["ai"]
    model_cfg = _model_config(config, args.model_key)
    backend_name = args.backend or model_cfg.get("backend") or ai_cfg["default_backend"]
    backend = build_backend(backend_name)
    manifest = backend.generate(
        args.input_renders,
        args.output,
        prompt=args.prompt,
        candidates_per_view=args.candidates or int(ai_cfg["candidates_per_view"]),
        seed=args.seed if args.seed is not None else int(ai_cfg["seed"]),
        model_ref=args.model_ref or _model_ref(config, args.model_key),
        device=args.device or ai_cfg["device"],
        dtype=args.dtype or ai_cfg["dtype"],
        variant=args.variant if args.variant is not None else model_cfg.get("variant") or ai_cfg.get("variant"),
        steps=args.steps if args.steps is not None else int(ai_cfg["steps"]),
        guidance_scale=args.guidance_scale if args.guidance_scale is not None else float(ai_cfg["guidance_scale"]),
        strength=args.strength if args.strength is not None else float(ai_cfg["strength"]),
        model_config=model_cfg,
        negative_prompt=args.negative_prompt if args.negative_prompt is not None else model_cfg.get("negative_prompt", ""),
        width=args.ai_width if args.ai_width is not None else int(model_cfg.get("width", ai_cfg.get("width", 1536))),
        height=args.ai_height if args.ai_height is not None else int(model_cfg.get("height", ai_cfg.get("height", 1536))),
        canny_scale=args.canny_scale if args.canny_scale is not None else float(model_cfg.get("canny_scale", 2.85)),
        depth_scale=args.depth_scale if args.depth_scale is not None else float(model_cfg.get("depth_scale", 0.55)),
        control_only=args.control_only if args.control_only is not None else bool(model_cfg.get("control_only", True)),
        geometry_lock=args.geometry_lock if args.geometry_lock is not None else bool(model_cfg.get("geometry_lock", True)),
    )
    print(f"Wrote AI manifest: {manifest}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    config = _load(args.config)
    score_cfg = config["score"]
    report = score_candidates(
        render_manifest_path=args.input_renders,
        ai_manifest_path=args.input_candidates,
        output_dir=args.output,
        edge_weight=float(score_cfg["edge_weight"]),
        mask_weight=float(score_cfg["mask_weight"]),
        prompt_weight=float(score_cfg["prompt_weight"]),
        copy_top_k=args.top_k or int(score_cfg["copy_top_k"]),
    )
    print(summarize_report(report))
    print(f"Wrote score report: {report}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _load(args.config)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    workdir = Path(args.workdir or Path(config["paths"]["outputs_dir"]) / f"run-{timestamp}")
    renders_dir = workdir / "renders"
    ai_dir = workdir / "candidates"
    score_dir = workdir / "score"
    model_cfg = _model_config(config, args.model_key)
    backend = args.backend or model_cfg.get("backend") or config["ai"]["default_backend"]
    model_path = args.model

    if model_path is None and args.generate_model:
        print("Generating 3D model...")
        gen_cfg = config["model_generation"]
        model_path = str(
            generate_3d_model(
                prompt=args.prompt,
                output=args.model_output or workdir / "generated_model.obj",
                backend=args.model_backend or gen_cfg["default_backend"],
                external_command=gen_cfg.get("external_command", ""),
            )
        )

    if model_path:
        print("Rendering model with Blender...")
        render_manifest = render_model_with_blender(
            model_path=model_path,
            output_dir=renders_dir,
            blender_script=resolve_path(config, config["paths"]["blender_script"]),
            blender_path=config["system"].get("blender_path") or None,
            views=args.views or int(config["render"]["views"]),
            resolution=args.resolution or int(config["render"]["resolution"]),
            engine=config["render"]["engine"],
            samples=int(config["render"]["samples"]),
            camera_distance=float(config["render"]["camera_distance"]),
        )
    else:
        print("No model supplied; creating synthetic sample render channels...")
        render_manifest = create_sample_renders(
            renders_dir,
            views=args.views or min(4, int(config["render"]["views"])),
            resolution=args.resolution or min(512, int(config["render"]["resolution"])),
        )

    ai_manifest = build_backend(backend).generate(
        render_manifest,
        ai_dir,
        prompt=args.prompt,
        candidates_per_view=args.candidates or int(config["ai"]["candidates_per_view"]),
        seed=args.seed if args.seed is not None else int(config["ai"]["seed"]),
        model_ref=args.model_ref or _model_ref(config, args.model_key),
        device=args.device or config["ai"]["device"],
        dtype=args.dtype or config["ai"]["dtype"],
        variant=args.variant if args.variant is not None else model_cfg.get("variant") or config["ai"].get("variant"),
        steps=args.steps or int(config["ai"]["steps"]),
        guidance_scale=args.guidance_scale if args.guidance_scale is not None else float(config["ai"]["guidance_scale"]),
        strength=args.strength if args.strength is not None else float(config["ai"]["strength"]),
        model_config=model_cfg,
        negative_prompt=args.negative_prompt if args.negative_prompt is not None else model_cfg.get("negative_prompt", ""),
        width=args.ai_width if args.ai_width is not None else int(model_cfg.get("width", config["ai"].get("width", 1536))),
        height=args.ai_height if args.ai_height is not None else int(model_cfg.get("height", config["ai"].get("height", 1536))),
        canny_scale=args.canny_scale if args.canny_scale is not None else float(model_cfg.get("canny_scale", 2.85)),
        depth_scale=args.depth_scale if args.depth_scale is not None else float(model_cfg.get("depth_scale", 0.55)),
        control_only=args.control_only if args.control_only is not None else bool(model_cfg.get("control_only", True)),
        geometry_lock=args.geometry_lock if args.geometry_lock is not None else bool(model_cfg.get("geometry_lock", True)),
    )
    report = score_candidates(
        render_manifest_path=render_manifest,
        ai_manifest_path=ai_manifest,
        output_dir=score_dir,
        edge_weight=float(config["score"]["edge_weight"]),
        mask_weight=float(config["score"]["mask_weight"]),
        prompt_weight=float(config["score"]["prompt_weight"]),
        copy_top_k=args.top_k or int(config["score"]["copy_top_k"]),
    )
    summary = {
        "workdir": str(workdir),
        "model_path": model_path or "",
        "render_manifest": str(render_manifest),
        "ai_manifest": str(ai_manifest),
        "score_report": str(report),
    }
    with (workdir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(summarize_report(report))
    print(f"Run complete: {workdir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local3dai", description="Local 3D to AI image rendering pipeline")
    parser.add_argument("--config", default="configs/local.json", help="Path to local JSON config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect local runtime configuration")
    doctor.add_argument("--scan-models", action="store_true", help="Scan common folders for local image model weights")
    doctor.set_defaults(func=cmd_doctor)

    init = subparsers.add_parser("init", help="Generate a local config from the current machine")
    init.add_argument("--config", default="configs/local.json")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    sample = subparsers.add_parser("sample-renders", help="Create synthetic render channels for pipeline testing")
    sample.add_argument("--output", default="outputs/sample_renders")
    sample.add_argument("--views", type=int, default=4)
    sample.add_argument("--resolution", type=int, default=512)
    sample.set_defaults(func=cmd_sample_renders)

    generate_3d = subparsers.add_parser("generate-3d", help="Generate or fetch a 3D model for the pipeline")
    generate_3d.add_argument("--prompt", required=True)
    generate_3d.add_argument("--output")
    generate_3d.add_argument("--backend", choices=["sample", "external", "procedural-crystal"])
    generate_3d.add_argument("--external-command")
    generate_3d.set_defaults(func=cmd_generate_3d)

    render = subparsers.add_parser("render", help="Render RGB/depth/edge/normal/mask channels with Blender")
    render.add_argument("--model", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--blender")
    render.add_argument("--views", type=int)
    render.add_argument("--resolution", type=int)
    render.add_argument("--engine")
    render.add_argument("--samples", type=int)
    render.add_argument("--camera-distance", type=float)
    render.set_defaults(func=cmd_render)

    ai_render = subparsers.add_parser("ai-render", help="Generate final image candidates from render channels")
    ai_render.add_argument("--input-renders", required=True)
    ai_render.add_argument("--output", required=True)
    ai_render.add_argument("--prompt", required=True)
    ai_render.add_argument("--backend")
    ai_render.add_argument("--model-key")
    ai_render.add_argument("--model-ref")
    ai_render.add_argument("--candidates", type=int)
    ai_render.add_argument("--seed", type=int)
    ai_render.add_argument("--device")
    ai_render.add_argument("--dtype")
    ai_render.add_argument("--variant")
    ai_render.add_argument("--steps", type=int)
    ai_render.add_argument("--guidance-scale", type=float)
    ai_render.add_argument("--strength", type=float)
    ai_render.add_argument("--negative-prompt")
    ai_render.add_argument("--ai-width", type=int)
    ai_render.add_argument("--ai-height", type=int)
    ai_render.add_argument("--canny-scale", type=float)
    ai_render.add_argument("--depth-scale", type=float)
    ai_render.add_argument("--control-only", dest="control_only", action="store_true", default=None)
    ai_render.add_argument("--img2img-control", dest="control_only", action="store_false")
    ai_render.add_argument("--geometry-lock", dest="geometry_lock", action="store_true", default=None)
    ai_render.add_argument("--no-geometry-lock", dest="geometry_lock", action="store_false")
    ai_render.set_defaults(func=cmd_ai_render)

    score = subparsers.add_parser("score", help="Score and rank candidates against source channels")
    score.add_argument("--input-renders", required=True)
    score.add_argument("--input-candidates", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--top-k", type=int)
    score.set_defaults(func=cmd_score)

    run = subparsers.add_parser("run", help="Run the full pipeline")
    run.add_argument("--prompt", required=True)
    run.add_argument("--model", help="Optional .glb/.gltf/.obj path. If omitted, synthetic renders are used.")
    run.add_argument("--generate-model", action="store_true", help="Generate a model first when --model is omitted")
    run.add_argument("--model-output")
    run.add_argument("--model-backend", choices=["sample", "external", "procedural-crystal"])
    run.add_argument("--workdir")
    run.add_argument("--backend")
    run.add_argument("--model-key")
    run.add_argument("--model-ref")
    run.add_argument("--device")
    run.add_argument("--dtype")
    run.add_argument("--variant")
    run.add_argument("--steps", type=int)
    run.add_argument("--guidance-scale", type=float)
    run.add_argument("--strength", type=float)
    run.add_argument("--negative-prompt")
    run.add_argument("--ai-width", type=int)
    run.add_argument("--ai-height", type=int)
    run.add_argument("--canny-scale", type=float)
    run.add_argument("--depth-scale", type=float)
    run.add_argument("--control-only", dest="control_only", action="store_true", default=None)
    run.add_argument("--img2img-control", dest="control_only", action="store_false")
    run.add_argument("--geometry-lock", dest="geometry_lock", action="store_true", default=None)
    run.add_argument("--no-geometry-lock", dest="geometry_lock", action="store_false")
    run.add_argument("--views", type=int)
    run.add_argument("--resolution", type=int)
    run.add_argument("--candidates", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--top-k", type=int)
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
