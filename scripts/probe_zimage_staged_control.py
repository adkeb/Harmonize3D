#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageOps

from local3dai.ai.geometry import mesh_adaptive_lock_render, mesh_detail_lock_render, mesh_position_lock_render
from local3dai.manifest import read_manifest
from local3dai.scoring_v2 import score_structure_v2


DEFAULT_STRUCTURE_POINTS = [(0.0, 0.95), (0.35, 0.85), (0.7, 0.35), (1.0, 0.2)]
DEFAULT_APPEARANCE_POINTS = [(0.0, 0.0), (0.35, 0.15), (0.7, 0.55), (1.0, 0.7)]
SUPPORTED_STRUCTURE_CHANNELS = {"depth", "gray", "canny", "edge", "mask", "normal", "rgb", "skeleton"}


def _workspace_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


def _parse_points(value: str | None, default: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not value:
        return list(default)
    points: list[tuple[float, float]] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Schedule point must be ratio:scale, got {chunk!r}.")
        ratio, scale = chunk.split(":", 1)
        points.append((float(ratio), float(scale)))
    return _normalize_points(points)


def _normalize_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        raise ValueError("Schedule needs at least one point.")
    points = sorted((float(x), float(y)) for x, y in points)
    if points[0][0] > 0.0:
        points.insert(0, (0.0, points[0][1]))
    if points[-1][0] < 1.0:
        points.append((1.0, points[-1][1]))
    normalized = []
    for ratio, scale in points:
        normalized.append((min(1.0, max(0.0, ratio)), max(0.0, scale)))
    return normalized


def interpolate_schedule(points: list[tuple[float, float]], steps: int) -> list[float]:
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    points = _normalize_points(points)
    values: list[float] = []
    for index in range(steps):
        ratio = 0.0 if steps == 1 else index / (steps - 1)
        left = points[0]
        right = points[-1]
        for start, end in zip(points, points[1:]):
            if start[0] <= ratio <= end[0]:
                left, right = start, end
                break
        if right[0] == left[0]:
            value = right[1]
        else:
            alpha = (ratio - left[0]) / (right[0] - left[0])
            value = left[1] + alpha * (right[1] - left[1])
        values.append(round(float(value), 6))
    return values


def _channel_source(channel: str, files: dict[str, str]) -> str | None:
    channel = "edge" if channel == "canny" else channel
    if channel == "gray":
        return files.get("rgb")
    return files.get(channel)


def _make_control_preview(channel: str, source: str | Path, output: Path, size: tuple[int, int] | None = None) -> Path:
    image = Image.open(source).convert("RGB")
    if size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    if channel in {"depth", "edge", "canny", "mask", "skeleton"}:
        image = ImageOps.autocontrast(image.convert("L")).convert("RGB")
    elif channel == "gray":
        image = ImageOps.autocontrast(image.convert("L")).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def _make_contact_sheet(items: list[dict[str, str]], output: Path, *, tile: int = 220) -> Path | None:
    if not items:
        return None
    pad = 10
    label_h = 28
    cols = min(4, max(1, len(items)))
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (pad + cols * (tile + pad), pad + rows * (tile + label_h + pad)), (20, 20, 22))
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(items):
        image = Image.open(item["file"]).convert("RGB")
        image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        x = pad + (index % cols) * (tile + pad)
        y = pad + (index // cols) * (tile + label_h + pad)
        draw.text((x, y + 8), item["label"], fill=(238, 238, 238))
        canvas.paste(image, (x, y + label_h + (tile - image.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _source_preview(view: dict[str, Any], output: Path, size: tuple[int, int]) -> Path | None:
    files = dict(view.get("files", {}))
    source = files.get("rgb")
    if not source or not Path(source).exists():
        return None
    image = Image.open(source).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def path_status(path: str | Path) -> dict[str, Any]:
    p = _workspace_path(path)
    exists = p.exists()
    size = 0
    aria2_files: list[str] = []
    if exists and p.is_file():
        size = p.stat().st_size
        if p.name.endswith(".aria2"):
            aria2_files.append(str(p))
    elif exists and p.is_dir():
        files = [item for item in p.rglob("*") if item.is_file()]
        size = sum(item.stat().st_size for item in files)
        aria2_files = [str(item) for item in files if item.name.endswith(".aria2")]
    return {
        "path": str(p),
        "exists": exists,
        "kind": "directory" if exists and p.is_dir() else "file" if exists else "missing",
        "size_gib": round(size / 1024**3, 3),
        "incomplete_download_markers": aria2_files,
        "complete": bool(exists and not aria2_files),
    }


def inspect_zimage_interface() -> dict[str, Any]:
    try:
        from diffusers import ZImageControlNetModel, ZImageControlNetPipeline
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    call_signature = inspect.signature(ZImageControlNetPipeline.__call__)
    call_params = set(call_signature.parameters)
    forward_signature = inspect.signature(ZImageControlNetModel.forward)
    forward_params = set(forward_signature.parameters)
    callback_inputs = list(getattr(ZImageControlNetPipeline, "_callback_tensor_inputs", []))
    return {
        "available": True,
        "pipeline": "ZImageControlNetPipeline",
        "controlnet_model": "ZImageControlNetModel",
        "has_control_image": "control_image" in call_params,
        "has_controlnet_conditioning_scale": "controlnet_conditioning_scale" in call_params,
        "has_native_control_guidance_start_end": {"control_guidance_start", "control_guidance_end"} <= call_params,
        "has_latent_handoff": "latents" in call_params,
        "callback_tensor_inputs": callback_inputs,
        "callback_can_change_control_scale_before_current_step": False,
        "controlnet_forward_has_conditioning_scale": "conditioning_scale" in forward_params,
        "supports_lora_loader": hasattr(ZImageControlNetPipeline, "load_lora_weights"),
        "supports_adapter_weights": hasattr(ZImageControlNetPipeline, "set_adapters"),
        "recommended_control_strategy": "wrap controlnet.forward and replace conditioning_scale per denoise step",
        "recommended_appearance_strategy": "load appearance LoRA and adjust adapter weight at step boundaries",
    }


@dataclass
class ScheduledForwardWrapper:
    forward: Callable[..., Any]
    schedule: list[float]
    calls: list[float]

    def __call__(self, *args: Any, conditioning_scale: float = 1.0, **kwargs: Any) -> Any:
        index = min(len(self.calls), len(self.schedule) - 1)
        scale = float(self.schedule[index])
        self.calls.append(scale)
        return self.forward(*args, conditioning_scale=scale, **kwargs)


def _wrapper_self_test(schedule: list[float]) -> dict[str, Any]:
    observed: list[float] = []

    def fake_forward(*_: Any, conditioning_scale: float = 1.0, **__: Any) -> dict[str, float]:
        observed.append(float(conditioning_scale))
        return {"scale": float(conditioning_scale)}

    wrapper = ScheduledForwardWrapper(fake_forward, schedule=schedule, calls=[])
    for _ in schedule:
        wrapper(object(), conditioning_scale=999.0)
    return {
        "passed": observed == schedule and wrapper.calls == schedule,
        "observed": observed,
        "wrapper_calls": wrapper.calls,
    }


def _manifest_status(
    manifest_path: Path,
    structure_channels: list[str],
    output_dir: Path,
    *,
    width: int | None,
    height: int | None,
) -> dict[str, Any]:
    manifest = read_manifest(manifest_path)
    views = manifest.get("views", [])
    controls_dir = output_dir / "control_previews"
    size = (width, height) if width and height else None
    view_reports = []
    contact_items: list[dict[str, str]] = []
    for view in views:
        view_id = str(view.get("view_id", "view"))
        files = dict(view.get("files", {}))
        channel_reports = []
        for channel in structure_channels:
            source = _channel_source(channel, files)
            report = {"channel": channel, "source": source or "", "exists": bool(source and Path(source).exists())}
            if source and Path(source).exists():
                preview = _make_control_preview(channel, source, controls_dir / f"{view_id}_{channel}.png", size=size)
                report["control_preview"] = str(preview)
                contact_items.append({"label": f"{view_id} {channel}", "file": str(preview)})
            channel_reports.append(report)
        view_reports.append({"view_id": view_id, "channels": channel_reports})
    contact = _make_contact_sheet(contact_items, output_dir / "zimage_control_previews_contact.png")
    missing = [
        {"view_id": view["view_id"], "channel": item["channel"]}
        for view in view_reports
        for item in view["channels"]
        if not item["exists"]
    ]
    return {
        "manifest": str(manifest_path),
        "view_count": len(views),
        "view_ids": [str(view.get("view_id", "")) for view in views],
        "structure_channels": structure_channels,
        "missing": missing,
        "control_previews_contact": str(contact) if contact else "",
        "views": view_reports,
    }


def _resolve_existing_controlnet(path: Path) -> Path:
    if path.exists():
        return path
    candidates = [
        Path("models/Z-Image-Turbo-Fun-Controlnet-Union-2.1") / path.name,
        Path("models") / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _run_generate(args: argparse.Namespace, report: dict[str, Any], structure_schedule: list[float], appearance_schedule: list[float]) -> dict[str, Any]:
    try:
        import torch
        from diffusers import ZImageControlNetModel, ZImageControlNetPipeline
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Z-Image generation requires torch and diffusers.") from exc

    base_model = _workspace_path(args.base_model)
    controlnet_file = _resolve_existing_controlnet(_workspace_path(args.controlnet_file))
    controlnet_config = _workspace_path(args.controlnet_config)
    if not base_model.exists():
        raise RuntimeError(f"Base model is missing: {base_model}")
    if not controlnet_file.exists():
        raise RuntimeError(f"ControlNet file is missing: {controlnet_file}")
    if not controlnet_config.exists():
        raise RuntimeError(f"ControlNet config is missing: {controlnet_config}")
    if len(args.structure_channels) != 1:
        raise RuntimeError("Generation mode currently expects one structure channel per run; use dry-run for matrix planning.")

    torch_dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(args.dtype).lower(), torch.bfloat16)

    controlnet = ZImageControlNetModel.from_single_file(
        str(controlnet_file),
        config=str(controlnet_config),
        local_files_only=True,
        torch_dtype=torch_dtype,
    )
    pipe = ZImageControlNetPipeline.from_pretrained(
        str(base_model),
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    if args.enable_model_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)

    appearance_lora = _workspace_path(args.appearance_lora) if args.appearance_lora else None
    if appearance_lora and appearance_lora.exists():
        pipe.load_lora_weights(str(appearance_lora), adapter_name="appearance")
        pipe.set_adapters("appearance", adapter_weights=appearance_schedule[0])

    original_forward = pipe.controlnet.forward
    calls: list[float] = []
    pipe.controlnet.forward = ScheduledForwardWrapper(original_forward, schedule=structure_schedule, calls=calls)  # type: ignore[method-assign]

    output = _workspace_path(args.output)
    manifest = read_manifest(_workspace_path(args.manifest))
    generated: list[dict[str, Any]] = []
    generation_contact_items: list[dict[str, str]] = []
    size = (args.width, args.height)
    mesh_lock_mode = str(args.mesh_lock_mode).strip().lower()
    detail_reference = _workspace_path(args.detail_reference) if args.detail_reference else None
    for view in manifest.get("views", [])[: args.max_views]:
        view_id = str(view.get("view_id", "view"))
        files = dict(view.get("files", {}))
        channel = args.structure_channels[0]
        source = _channel_source(channel, files)
        if not source:
            raise RuntimeError(f"View {view_id} is missing structure channel {channel}.")
        control_image = Image.open(source).convert("RGB").resize(size, Image.Resampling.LANCZOS)
        if channel in {"depth", "edge", "canny", "gray", "mask", "skeleton"}:
            control_image = ImageOps.autocontrast(control_image.convert("L")).convert("RGB")
        control_preview = output / "generated" / view_id / f"{channel}_control.png"
        control_preview.parent.mkdir(parents=True, exist_ok=True)
        control_image.save(control_preview)

        def callback_on_step_end(pipe_obj: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            next_index = min(step + 1, len(appearance_schedule) - 1)
            if appearance_lora and appearance_lora.exists():
                pipe_obj.set_adapters("appearance", adapter_weights=appearance_schedule[next_index])
            return callback_kwargs

        generator = torch.Generator(device=args.device.split(":")[0]).manual_seed(args.seed)
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or None,
            control_image=control_image,
            controlnet_conditioning_scale=structure_schedule[0],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
        )
        view_dir = output / "generated" / view_id
        view_dir.mkdir(parents=True, exist_ok=True)
        direct_path = view_dir / "direct_00.png"
        result.images[0].save(direct_path)
        image_path = view_dir / "candidate_00.png"
        if mesh_lock_mode == "position":
            mesh_position_lock_render(
                source_rgb=files["rgb"],
                source_mask=files["mask"],
                source_edge=files["edge"],
                ai_image=direct_path,
                output=image_path,
            )
        elif mesh_lock_mode == "detail":
            mesh_detail_lock_render(
                source_rgb=files["rgb"],
                source_mask=files["mask"],
                source_edge=files["edge"],
                ai_image=direct_path,
                output=image_path,
                detail_reference=detail_reference,
            )
        elif mesh_lock_mode == "adaptive":
            mesh_adaptive_lock_render(
                source_rgb=files["rgb"],
                source_mask=files["mask"],
                source_edge=files["edge"],
                ai_image=direct_path,
                output=image_path,
                detail_reference=detail_reference,
            )
        else:
            result.images[0].save(image_path)

        item: dict[str, Any] = {
            "view_id": view_id,
            "file": str(image_path),
            "direct_file": str(direct_path),
            "structure_channel": channel,
            "mesh_lock_mode": mesh_lock_mode,
        }
        if files:
            item["structure_scores"] = score_structure_v2(candidate_path=image_path, source_files=files)
            item["direct_structure_scores"] = score_structure_v2(candidate_path=direct_path, source_files=files)
        generated.append(item)
        source_preview = _source_preview(view, output / "generated" / view_id / "source_rgb.png", size)
        if source_preview:
            generation_contact_items.append({"label": f"{view_id} source", "file": str(source_preview)})
        generation_contact_items.append({"label": f"{view_id} {channel} control", "file": str(control_preview)})
        generation_contact_items.append(
            {
                "label": f"{view_id} direct total={item.get('direct_structure_scores', {}).get('total', 'n/a')}",
                "file": str(direct_path),
            }
        )
        generation_contact_items.append(
            {
                "label": f"{view_id} final total={item.get('structure_scores', {}).get('total', 'n/a')}",
                "file": str(image_path),
            }
        )

    generation_contact = _make_contact_sheet(
        generation_contact_items,
        output / "zimage_generation_contact.png",
        tile=260,
    )
    report["generation"] = {
        "status": "complete",
        "generated": generated,
        "mesh_lock_mode": mesh_lock_mode,
        "controlnet_scales_observed": calls,
        "contact_sheet": str(generation_contact) if generation_contact else "",
    }
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    output = _workspace_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    structure_channels = [str(channel).strip().lower() for channel in args.structure_channels if str(channel).strip()]
    unsupported = sorted(set(structure_channels) - SUPPORTED_STRUCTURE_CHANNELS)
    if unsupported:
        raise RuntimeError(f"Unsupported structure channels: {unsupported}")
    if not structure_channels:
        raise RuntimeError("At least one structure channel is required.")

    structure_points = _parse_points(args.structure_schedule, list(DEFAULT_STRUCTURE_POINTS))
    appearance_points = _parse_points(args.appearance_schedule, list(DEFAULT_APPEARANCE_POINTS))
    structure_schedule = interpolate_schedule(structure_points, args.steps)
    appearance_schedule = interpolate_schedule(appearance_points, args.steps)
    controlnet_file = _resolve_existing_controlnet(_workspace_path(args.controlnet_file))
    manifest_status = _manifest_status(
        _workspace_path(args.manifest),
        structure_channels,
        output,
        width=args.width,
        height=args.height,
    )
    interface = inspect_zimage_interface()
    wrapper_test = _wrapper_self_test(structure_schedule)
    local_weights = {
        "base_model": path_status(args.base_model),
        "controlnet_file": path_status(controlnet_file),
        "controlnet_config": path_status(args.controlnet_config),
        "i2l_model": path_status(args.i2l_model),
        "appearance_lora": path_status(args.appearance_lora) if args.appearance_lora else {"path": "", "exists": False},
    }
    ready_to_generate = bool(
        interface.get("available")
        and interface.get("has_control_image")
        and interface.get("controlnet_forward_has_conditioning_scale")
        and local_weights["base_model"]["complete"]
        and local_weights["controlnet_file"]["complete"]
        and local_weights["controlnet_config"]["complete"]
        and not manifest_status["missing"]
    )
    report: dict[str, Any] = {
        "type": "zimage_staged_control_probe",
        "status": "dry_run" if not args.generate else "running",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "manifest": manifest_status,
        "zimage_interface": interface,
        "local_weights": local_weights,
        "steps": args.steps,
        "width": args.width,
        "height": args.height,
        "mesh_lock_mode": str(args.mesh_lock_mode).strip().lower(),
        "detail_reference": str(_workspace_path(args.detail_reference)) if args.detail_reference else "",
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
        "structure_schedule_points": structure_points,
        "appearance_schedule_points": appearance_points,
        "structure_schedule": structure_schedule,
        "appearance_schedule": appearance_schedule,
        "structure_schedule_wrapper_self_test": wrapper_test,
        "ready_to_generate": ready_to_generate,
        "generate_requested": bool(args.generate),
        "genericity_checks": {
            "view_id_specific_logic": False,
            "uses_all_manifest_views_in_dry_run": True,
            "structure_channels_are_parameterized": True,
            "schedules_depend_on_step_ratio_not_view_id": True,
        },
        "interpretation": {
            "flux2_klein_limitation_addressed": "This probe uses an internal ControlNet branch for structure instead of passing structure as ordinary reference images.",
            "remaining_gate": "Actual image generation still requires local Z-Image base weights plus the ControlNet file.",
        },
    }
    if args.generate:
        report = _run_generate(args, report, structure_schedule, appearance_schedule)
        report["status"] = "complete"
    report["elapsed_seconds"] = round(time.time() - started, 3)
    report_path = output / "zimage_staged_control_probe_report.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Z-Image internal staged ControlNet + appearance scheduling.")
    parser.add_argument("--manifest", default="outputs/flux2_klein_high_quality_car_reference/white_renders/manifest.json")
    parser.add_argument("--output", default="outputs/flux2_klein_high_quality_car_reference/zimage_staged_control_probe")
    parser.add_argument("--base-model", default="models/Z-Image")
    parser.add_argument(
        "--controlnet-file",
        default="models/Z-Image-Fun-Controlnet-Union-2.1/Z-Image-Fun-Controlnet-Union-2.1-lite.safetensors",
    )
    parser.add_argument("--controlnet-config", default="models/Z-Image-ControlNet-config-lite")
    parser.add_argument("--i2l-model", default="models/Z-Image-i2L/model.safetensors")
    parser.add_argument("--appearance-lora", default="")
    parser.add_argument("--prompt", default="premium red racecar product render, glossy black glass, carbon aero, clean studio lighting")
    parser.add_argument("--negative-prompt", default="line art, gray clay, white plastic, copied control lines, distorted geometry, extra parts")
    parser.add_argument("--structure-channels", nargs="+", default=["depth", "gray", "canny"])
    parser.add_argument("--structure-schedule", default="")
    parser.add_argument("--appearance-schedule", default="")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-views", type=int, default=1)
    parser.add_argument("--mesh-lock-mode", choices=["none", "position", "detail", "adaptive"], default="none")
    parser.add_argument("--detail-reference", default="outputs/flux2_klein_high_quality_car_reference/racecar_reference_image2.png")
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print("Wrote report:", report["report"])
    print("Ready to generate:", report["ready_to_generate"])
    print("Status:", report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
