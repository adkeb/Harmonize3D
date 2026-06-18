#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from local3dai.ai.backends import build_backend
from local3dai.ai.geometry import mesh_position_lock_render
from local3dai.config import load_config
from local3dai.manifest import read_manifest, write_manifest
from local3dai.scoring_v2 import score_structure_v2


def _view_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(view["view_id"]): view for view in manifest.get("views", [])}


def _single_view_manifest(source_manifest: dict[str, Any], view: dict[str, Any], output: Path) -> Path:
    data = {**source_manifest, "views": [view], "view_graph": {"edges": []}}
    return write_manifest(output, data)


def _model_ref(model: dict[str, Any]) -> str:
    return model.get("local_path") or model.get("model_path") or model.get("model_id") or ""


def _make_contact(items: list[dict[str, str]], output: Path, *, tile: int = 320) -> Path:
    pad = 12
    label_h = 36
    cols = min(4, max(1, len(items)))
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (pad + cols * (tile + pad), pad + rows * (tile + label_h + pad)), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(items):
        image = Image.open(item["file"]).convert("RGB")
        image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        x = pad + (index % cols) * (tile + pad)
        y = pad + (index // cols) * (tile + label_h + pad)
        draw.text((x, y + 9), item["label"], fill=(242, 242, 242))
        canvas.paste(image, (x, y + label_h + (tile - image.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _edge_leak_score(candidate: str | Path, source_edge: str | Path, source_mask: str | Path) -> float:
    import cv2

    image = Image.open(candidate).convert("L")
    size = image.size
    edge = Image.open(source_edge).convert("L").resize(size, Image.Resampling.LANCZOS)
    mask = Image.open(source_mask).convert("L").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.uint8)
    src_edge = np.asarray(edge, dtype=np.uint8)
    src_mask = np.asarray(mask, dtype=np.uint8)
    border = np.concatenate([src_mask[0, :], src_mask[-1, :], src_mask[:, 0], src_mask[:, -1]], axis=0)
    bg = float(np.median(border))
    object_mask = src_mask > max(30.0, bg + 18.0)
    if not object_mask.any():
        return 0.0
    candidate_edges = cv2.Canny(arr, 80, 160) > 0
    structure_edges = cv2.Canny(src_edge, 80, 160) > 0
    overlap = np.logical_and(np.logical_and(candidate_edges, structure_edges), object_mask)
    leak_ratio = float(overlap.sum()) / max(float(np.logical_and(structure_edges, object_mask).sum()), 1.0)
    return round(float(np.clip(leak_ratio, 0.0, 1.0)), 6)


def _inspect_flux2_capability() -> dict[str, Any]:
    from diffusers import Flux2KleinPipeline

    signature = inspect.signature(Flux2KleinPipeline.__call__)
    params = list(signature.parameters)
    callback_inputs = list(getattr(Flux2KleinPipeline, "_callback_tensor_inputs", []))
    return {
        "pipeline": "Flux2KleinPipeline",
        "supports_image_list": "image" in params,
        "callback_tensor_inputs": callback_inputs,
        "has_per_reference_weight": False,
        "has_controlnet_scale": False,
        "has_ip_adapter_scale": False,
        "true_dual_image_step_weight_schedule_supported": False,
        "reason": "Callback exposes latents/prompt_embeds only; no per-reference image embeddings, ControlNet scale, or IP-Adapter scale are exposed.",
    }


def _generate(
    *,
    backend: Any,
    manifest: Path,
    output: Path,
    prompt: str,
    model: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    steps: int,
    width: int,
    height: int,
    reference_channels: list[str],
    appearance_reference: Path | None = None,
    appearance_reference_images: list[Path] | None = None,
) -> Path:
    return backend.generate(
        manifest,
        output,
        prompt=prompt,
        negative_prompt=model.get("negative_prompt", ""),
        candidates_per_view=1,
        seed=seed,
        model_ref=_model_ref(model),
        model_config={**model, "reference_channels": reference_channels},
        device=config["ai"].get("device", "cuda:0"),
        dtype=model.get("dtype") or config["ai"].get("dtype", "bfloat16"),
        steps=steps,
        guidance_scale=float(model.get("guidance_scale", config["ai"].get("guidance_scale", 1.0))),
        width=width,
        height=height,
        reference_channels=reference_channels,
        appearance_reference=appearance_reference,
        appearance_reference_images=appearance_reference_images,
        appearance_reference_order="after",
        mesh_position_lock=False,
        geometry_lock=False,
    )


def _first_candidate(manifest_path: Path) -> dict[str, Any]:
    data = read_manifest(manifest_path)
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"No candidates in {manifest_path}")
    return candidates[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    model_key = args.model_key or config["ai"].get("default_model_key", "flux2_klein_4b")
    model = dict(config["models"][model_key])
    backend_name = model.get("backend", config["ai"].get("default_backend", "flux2-klein"))
    if backend_name != "flux2-klein":
        raise RuntimeError(f"This probe targets current Flux2 Klein only, got {backend_name!r}.")

    render_manifest = read_manifest(args.manifest)
    views = _view_by_id(render_manifest)
    view = views.get(args.view)
    if not view:
        raise RuntimeError(f"View {args.view!r} is missing from manifest.")
    single_manifest = _single_view_manifest(render_manifest, view, output / "single_view_manifest.json")
    backend = build_backend(backend_name)
    reference = Path(args.appearance_reference)

    capability = _inspect_flux2_capability()
    structure_prompt = (
        "exact current-view mesh structure only, preserve camera pose, silhouette, depth, panel layout, "
        "wheel openings, canopy footprint, rear wing, front splitter, clean clay structure frame"
    )
    appearance_prompt = (
        "premium red racecar product render, glossy black glass, carbon aero, black wheels, studio lighting, "
        "use appearance reference only for material and color, do not change camera pose or geometry"
    )

    experiments: list[dict[str, Any]] = []
    specs = [
        {
            "name": "A_structure_only",
            "prompt": structure_prompt,
            "channels": ["rgb", "edge", "depth", "normal", "mask", "skeleton"],
            "appearance": None,
            "steps": args.structure_steps,
            "seed": args.seed,
        },
        {
            "name": "B_appearance_only",
            "prompt": appearance_prompt,
            "channels": [],
            "appearance": reference,
            "steps": args.appearance_steps,
            "seed": args.seed + 100,
        },
        {
            "name": "C_simultaneous_structure_appearance",
            "prompt": f"{structure_prompt}. {appearance_prompt}",
            "channels": ["rgb", "edge", "depth", "normal", "mask", "skeleton"],
            "appearance": reference,
            "steps": args.total_steps,
            "seed": args.seed + 200,
        },
    ]

    for spec in specs:
        exp_dir = output / spec["name"]
        ai_manifest = _generate(
            backend=backend,
            manifest=single_manifest,
            output=exp_dir,
            prompt=spec["prompt"],
            model=model,
            config=config,
            seed=int(spec["seed"]),
            steps=int(spec["steps"]),
            width=args.width,
            height=args.height,
            reference_channels=list(spec["channels"]),
            appearance_reference=spec["appearance"],
        )
        candidate = _first_candidate(ai_manifest)
        score = score_structure_v2(candidate_path=candidate["file"], source_files=view["files"])
        experiments.append(
            {
                "name": spec["name"],
                "mode": "single_call",
                "prompt": spec["prompt"],
                "reference_channels": spec["channels"],
                "appearance_reference": str(spec["appearance"]) if spec["appearance"] else "",
                "steps": spec["steps"],
                "seed": spec["seed"],
                "ai_manifest": str(ai_manifest),
                "image": candidate["file"],
                "structure_scores": score,
                "edge_leak_score": _edge_leak_score(candidate["file"], view["files"]["edge"], view["files"]["mask"]),
            }
        )

    stage1 = experiments[0]
    two_stage_manifest = _single_view_manifest(
        {
            **render_manifest,
            "views": [
                {
                    **view,
                    "files": {
                        **dict(view["files"]),
                        "rgb": stage1["image"],
                    },
                }
            ],
        },
        {
            **view,
            "files": {
                **dict(view["files"]),
                "rgb": stage1["image"],
            },
        },
        output / "two_stage_stage1_as_rgb_manifest.json",
    )
    two_stage_ai = _generate(
        backend=backend,
        manifest=two_stage_manifest,
        output=output / "D_two_stage_approx",
        prompt=appearance_prompt,
        model=model,
        config=config,
        seed=args.seed + 300,
        steps=args.appearance_steps,
        width=args.width,
        height=args.height,
        reference_channels=["rgb"],
        appearance_reference=reference,
    )
    two_candidate = _first_candidate(two_stage_ai)
    two_stage_locked = output / "D_two_stage_approx" / "mesh_position_locked.png"
    mesh_position_lock_render(
        source_rgb=view["files"]["rgb"],
        source_mask=view["files"]["mask"],
        source_edge=view["files"]["edge"],
        ai_image=two_candidate["file"],
        output=two_stage_locked,
    )
    two_score = score_structure_v2(candidate_path=two_stage_locked, source_files=view["files"])
    experiments.append(
        {
            "name": "D_two_stage_approx",
            "mode": "two_separate_calls_not_true_step_schedule",
            "prompt": appearance_prompt,
            "reference_channels": ["stage1_rgb", "appearance_reference"],
            "appearance_reference": str(reference),
            "steps": [args.structure_steps, args.appearance_steps],
            "seed": args.seed + 300,
            "ai_manifest": str(two_stage_ai),
            "image": str(two_stage_locked),
            "direct_image": two_candidate["file"],
            "structure_scores": two_score,
            "edge_leak_score": _edge_leak_score(two_stage_locked, view["files"]["edge"], view["files"]["mask"]),
        }
    )

    contact = _make_contact(
        [
            {"label": "white rgb", "file": view["files"]["rgb"]},
            {"label": "white edge", "file": view["files"]["edge"]},
            {"label": "appearance ref", "file": str(reference)},
        ]
        + [
            {
                "label": f"{item['name']} total={item['structure_scores']['total']:.3f}",
                "file": item["image"],
            }
            for item in experiments
        ],
        output / "dual_image_staged_weight_probe_contact.png",
    )

    report = {
        "type": "dual_image_staged_weight_probe",
        "status": "complete",
        "model_key": model_key,
        "backend": backend_name,
        "view_id": args.view,
        "source_manifest": str(args.manifest),
        "appearance_reference": str(reference),
        "capability": capability,
        "interpretation": {
            "true_step_level_dual_image_weighting": False,
            "why": capability["reason"],
            "tested_fallback": "separate single-call baselines plus a two-call approximation using stage-1 structure output as the second call's rgb reference.",
        },
        "parameters": {
            "width": args.width,
            "height": args.height,
            "structure_steps": args.structure_steps,
            "appearance_steps": args.appearance_steps,
            "total_steps": args.total_steps,
            "seed": args.seed,
        },
        "experiments": experiments,
        "contact_sheet": str(contact),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path = output / "dual_image_staged_weight_probe_report.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe current Flux2 Klein dual-image staged weighting capability.")
    parser.add_argument("--manifest", default="outputs/flux2_klein_high_quality_car_reference/white_renders/manifest.json")
    parser.add_argument("--appearance-reference", default="outputs/flux2_klein_high_quality_car_reference/racecar_reference_image2.png")
    parser.add_argument("--output", default="outputs/flux2_klein_high_quality_car_reference/dual_image_staged_weight_probe")
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--model-key", default="flux2_klein_4b")
    parser.add_argument("--view", default="view_locked")
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--structure-steps", type=int, default=8)
    parser.add_argument("--appearance-steps", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=16)
    args = parser.parse_args()
    report = run(args)
    print("Wrote report:", report["report"])
    print("Contact sheet:", report["contact_sheet"])
    print("True step-level dual-image weighting:", report["capability"]["true_dual_image_step_weight_schedule_supported"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
