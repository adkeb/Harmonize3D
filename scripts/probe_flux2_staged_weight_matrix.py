#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from local3dai.ai.backends import build_backend
from local3dai.ai.geometry import mesh_position_lock_render
from local3dai.config import load_config
from local3dai.manifest import read_manifest, write_manifest
from local3dai.scoring_v2 import foreground_mask_from_image, score_structure_v2, source_mask_from_render


FULL_CHANNELS = ("rgb", "edge", "depth", "normal", "mask", "skeleton")
LIGHT_CHANNELS = ("rgb", "depth", "mask")


@dataclass(frozen=True)
class Recipe:
    name: str
    mode: str
    channels: tuple[str, ...]
    appearance: bool
    order: str
    prompt_kind: str
    steps_kind: str
    lock: bool = False


RECIPES = (
    Recipe(
        name="A_structure_full",
        mode="single_call",
        channels=FULL_CHANNELS,
        appearance=False,
        order="after",
        prompt_kind="structure",
        steps_kind="structure",
    ),
    Recipe(
        name="B_appearance_only",
        mode="single_call",
        channels=(),
        appearance=True,
        order="after",
        prompt_kind="appearance",
        steps_kind="appearance",
    ),
    Recipe(
        name="C_simultaneous_full_after",
        mode="single_call",
        channels=FULL_CHANNELS,
        appearance=True,
        order="after",
        prompt_kind="combined",
        steps_kind="total",
    ),
    Recipe(
        name="D_simultaneous_full_before",
        mode="single_call",
        channels=FULL_CHANNELS,
        appearance=True,
        order="before",
        prompt_kind="combined",
        steps_kind="total",
    ),
    Recipe(
        name="E_simultaneous_light_after",
        mode="single_call",
        channels=LIGHT_CHANNELS,
        appearance=True,
        order="after",
        prompt_kind="combined_light",
        steps_kind="total",
    ),
    Recipe(
        name="F_two_stage_full_direct",
        mode="two_separate_calls_not_true_step_schedule",
        channels=("stage1_rgb",),
        appearance=True,
        order="after",
        prompt_kind="appearance",
        steps_kind="appearance",
        lock=False,
    ),
    Recipe(
        name="G_two_stage_full_position_lock",
        mode="two_separate_calls_not_true_step_schedule",
        channels=("stage1_rgb",),
        appearance=True,
        order="after",
        prompt_kind="appearance",
        steps_kind="appearance",
        lock=True,
    ),
)


def _model_ref(model: dict[str, Any]) -> str:
    return model.get("local_path") or model.get("model_path") or model.get("model_id") or ""


def _view_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(view.get("view_id")): view for view in manifest.get("views", [])}


def _select_views(manifest: dict[str, Any], requested: str, max_views: int) -> list[dict[str, Any]]:
    available = [view for view in manifest.get("views", []) if view.get("view_id")]
    if not available:
        raise RuntimeError("Render manifest has no views.")
    if requested.strip().lower() == "all":
        selected = available
    else:
        by_id = _view_by_id(manifest)
        selected = []
        for view_id in [part.strip() for part in requested.split(",") if part.strip()]:
            if view_id not in by_id:
                raise RuntimeError(f"Requested view {view_id!r} is missing from manifest.")
            selected.append(by_id[view_id])
    if max_views > 0:
        selected = selected[:max_views]
    return selected


def _single_view_manifest(source_manifest: dict[str, Any], view: dict[str, Any], output: Path) -> Path:
    data = {**source_manifest, "views": [view], "view_graph": {"edges": []}}
    return write_manifest(output, data)


def _manifest_with_stage_rgb(source_manifest: dict[str, Any], view: dict[str, Any], stage_rgb: str, output: Path) -> Path:
    staged_view = {**view, "files": {**dict(view["files"]), "rgb": stage_rgb}}
    data = {**source_manifest, "views": [staged_view], "view_graph": {"edges": []}}
    return write_manifest(output, data)


def _step_count(recipe: Recipe, args: argparse.Namespace) -> int:
    if recipe.steps_kind == "structure":
        return int(args.structure_steps)
    if recipe.steps_kind == "appearance":
        return int(args.appearance_steps)
    return int(args.total_steps)


def _prompts(args: argparse.Namespace) -> dict[str, str]:
    structure = (
        "preserve the current camera view and exact mesh structure, keep silhouette, depth ordering, "
        "wheel openings, canopy footprint, rear wing, front splitter, clean clay geometry frame"
    )
    appearance = (
        args.prompt
        or "premium red race car product render, glossy red paint, black glass, black wheels, subtle carbon aero, clean studio background"
    )
    combined = (
        f"{appearance}. Use the appearance reference only for material, color, and product render style. "
        f"{structure}. Do not add new parts and do not trace control-map lines into the material."
    )
    combined_light = (
        f"{appearance}. Follow only broad pose, depth, silhouette, and mask from the 3D render. "
        "Avoid copying technical guide lines, edge-map strokes, or gray clay material."
    )
    return {
        "structure": structure,
        "appearance": appearance,
        "combined": combined,
        "combined_light": combined_light,
    }


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
        "reason": (
            "Flux2 Klein accepts multiple input images as one ordered list, but the exposed call/callback "
            "does not expose separate reference-image embeddings, per-reference weights, or ControlNet/IP-Adapter scales."
        ),
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
    reference_channels: tuple[str, ...],
    appearance_reference: Path | None,
    appearance_reference_order: str,
) -> Path:
    model_config = {
        **model,
        "reference_channels": list(reference_channels),
        "appearance_reference_order": appearance_reference_order,
    }
    return backend.generate(
        manifest,
        output,
        prompt=prompt,
        negative_prompt=model.get("negative_prompt", ""),
        candidates_per_view=1,
        seed=seed,
        model_ref=_model_ref(model),
        model_config=model_config,
        device=config["ai"].get("device", "cuda:0"),
        dtype=model.get("dtype") or config["ai"].get("dtype", "bfloat16"),
        steps=steps,
        guidance_scale=float(model.get("guidance_scale", config["ai"].get("guidance_scale", 1.0))),
        width=width,
        height=height,
        reference_channels=list(reference_channels),
        appearance_reference=appearance_reference,
        appearance_reference_order=appearance_reference_order,
        mesh_position_lock=False,
        geometry_lock=False,
    )


def _first_candidate(manifest_path: Path) -> dict[str, Any]:
    data = read_manifest(manifest_path)
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"No candidates in {manifest_path}")
    return candidates[0]


def _edge_leak_score(candidate: str | Path, source_edge: str | Path, source_mask: str | Path) -> float:
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
    return round(float(np.clip(overlap.sum() / max(float(np.logical_and(structure_edges, object_mask).sum()), 1.0), 0, 1)), 6)


def _appearance_stats(candidate: str | Path, source_files: dict[str, str]) -> dict[str, float]:
    image = Image.open(candidate).convert("RGB")
    size = image.size
    arr = np.asarray(image, dtype=np.float32) / 255.0
    source_mask = source_mask_from_render(source_files["mask"], size=size)
    candidate_mask = foreground_mask_from_image(candidate, size=size)
    mask = np.logical_or(source_mask, candidate_mask)
    if not mask.any():
        return {"redness": 0.0, "dark_part_ratio": 0.0, "saturation": 0.0}
    pixels = arr[mask]
    maxc = pixels.max(axis=1)
    minc = pixels.min(axis=1)
    saturation = (maxc - minc) / np.maximum(maxc, 1e-4)
    red_dominance = np.clip(pixels[:, 0] - np.maximum(pixels[:, 1], pixels[:, 2]), 0, 1)
    gray = pixels.mean(axis=1)
    source_pixels = arr[source_mask]
    source_maxc = source_pixels.max(axis=1)
    source_minc = source_pixels.min(axis=1)
    source_saturation = (source_maxc - source_minc) / np.maximum(source_maxc, 1e-4)
    source_gray = source_pixels.mean(axis=1)
    clay_like = np.logical_and(source_saturation < 0.08, source_gray > 0.45)
    red_paint = np.logical_and(source_pixels[:, 0] > np.maximum(source_pixels[:, 1], source_pixels[:, 2]) + 0.08, source_saturation > 0.18)
    return {
        "redness": round(float(red_dominance.mean()), 6),
        "dark_part_ratio": round(float((gray < 0.28).mean()), 6),
        "saturation": round(float(saturation.mean()), 6),
        "red_paint_coverage": round(float(red_paint.mean()), 6),
        "clay_like_ratio": round(float(clay_like.mean()), 6),
    }


def _make_contact(items: list[dict[str, str]], output: Path, *, tile: int = 260) -> Path:
    pad = 10
    label_h = 42
    cols = min(4, max(1, len(items)))
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (pad + cols * (tile + pad), pad + rows * (tile + label_h + pad)), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(items):
        image = Image.open(item["file"]).convert("RGB")
        image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        x = pad + (index % cols) * (tile + pad)
        y = pad + (index // cols) * (tile + label_h + pad)
        draw.text((x, y + 6), item["label"][:62], fill=(242, 242, 242))
        canvas.paste(image, (x, y + label_h + (tile - image.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _summarize(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for recipe_name in sorted({item["recipe"] for item in experiments}):
        items = [item for item in experiments if item["recipe"] == recipe_name]
        totals = [float(item["structure_scores"]["total"]) for item in items]
        silhouettes = [float(item["structure_scores"]["silhouette_iou"]) for item in items]
        added = [float(item["structure_scores"]["added_part_penalty"]) for item in items]
        edge_leak = [float(item["edge_leak_score"]) for item in items]
        redness = [float(item["appearance_stats"]["redness"]) for item in items]
        red_paint = [float(item["appearance_stats"]["red_paint_coverage"]) for item in items]
        clay = [float(item["appearance_stats"]["clay_like_ratio"]) for item in items]
        failures = sum(1 for item in items if item["structure_scores"].get("failure_reasons"))
        summary[recipe_name] = {
            "count": len(items),
            "mean_structure_total": round(float(np.mean(totals)), 6),
            "min_silhouette_iou": round(float(np.min(silhouettes)), 6),
            "mean_added_part_penalty": round(float(np.mean(added)), 6),
            "mean_edge_leak_score": round(float(np.mean(edge_leak)), 6),
            "mean_redness": round(float(np.mean(redness)), 6),
            "mean_red_paint_coverage": round(float(np.mean(red_paint)), 6),
            "mean_clay_like_ratio": round(float(np.mean(clay)), 6),
            "failure_count": failures,
        }
    return summary


def _recipes_from_arg(value: str) -> tuple[Recipe, ...]:
    if not value or value.strip().lower() == "all":
        return RECIPES
    requested = [part.strip() for part in value.split(",") if part.strip()]
    by_name = {recipe.name: recipe for recipe in RECIPES}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise RuntimeError(f"Unknown recipe(s): {', '.join(missing)}")
    selected = [by_name[name] for name in requested]
    needs_stage1 = any(recipe.mode != "single_call" for recipe in selected)
    if needs_stage1 and "A_structure_full" not in requested:
        selected.insert(0, by_name["A_structure_full"])
    ordered_names = {recipe.name for recipe in selected}
    return tuple(recipe for recipe in RECIPES if recipe.name in ordered_names)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    model_key = args.model_key or config["ai"].get("default_model_key", "flux2_klein_4b")
    model = dict(config["models"][model_key])
    backend_name = model.get("backend", config["ai"].get("default_backend", "flux2-klein"))
    if backend_name != "flux2-klein":
        raise RuntimeError(f"This probe targets Flux2 Klein only, got {backend_name!r}.")
    reference = Path(args.appearance_reference).expanduser().resolve()
    if not reference.exists():
        raise RuntimeError(f"Appearance reference does not exist: {reference}")

    render_manifest = read_manifest(args.manifest)
    selected_views = _select_views(render_manifest, args.views, args.max_views)
    backend = build_backend(backend_name)
    prompts = _prompts(args)
    capability = _inspect_flux2_capability()
    recipes = _recipes_from_arg(args.recipes)
    experiments: list[dict[str, Any]] = []
    contacts: list[dict[str, str]] = [{"label": "appearance reference", "file": str(reference)}]

    for view_index, view in enumerate(selected_views):
        view_id = str(view["view_id"])
        single_manifest = _single_view_manifest(render_manifest, view, root / view_id / "single_view_manifest.json")
        stage1_image = ""
        stage1_manifest_path = ""
        for recipe_index, recipe in enumerate(recipes):
            exp_dir = root / view_id / recipe.name
            if recipe.mode == "single_call":
                ai_manifest = _generate(
                    backend=backend,
                    manifest=single_manifest,
                    output=exp_dir,
                    prompt=prompts[recipe.prompt_kind],
                    model=model,
                    config=config,
                    seed=int(args.seed + view_index * 10000 + recipe_index * 100),
                    steps=_step_count(recipe, args),
                    width=args.width,
                    height=args.height,
                    reference_channels=recipe.channels,
                    appearance_reference=reference if recipe.appearance else None,
                    appearance_reference_order=recipe.order,
                )
                candidate = _first_candidate(ai_manifest)
                image_path = candidate["file"]
                direct_path = candidate.get("direct_file", image_path)
                if recipe.name == "A_structure_full":
                    stage1_image = image_path
                    stage1_manifest_path = str(ai_manifest)
            else:
                if not stage1_image:
                    raise RuntimeError("Two-stage recipe requires A_structure_full to run first.")
                stage_manifest = _manifest_with_stage_rgb(
                    render_manifest,
                    view,
                    stage1_image,
                    root / view_id / f"{recipe.name}_stage1_as_rgb_manifest.json",
                )
                ai_manifest = _generate(
                    backend=backend,
                    manifest=stage_manifest,
                    output=exp_dir,
                    prompt=prompts[recipe.prompt_kind],
                    model=model,
                    config=config,
                    seed=int(args.seed + view_index * 10000 + recipe_index * 100),
                    steps=_step_count(recipe, args),
                    width=args.width,
                    height=args.height,
                    reference_channels=("rgb",),
                    appearance_reference=reference,
                    appearance_reference_order=recipe.order,
                )
                candidate = _first_candidate(ai_manifest)
                direct_path = candidate.get("direct_file", candidate["file"])
                if recipe.lock:
                    locked = exp_dir / "mesh_position_locked.png"
                    mesh_position_lock_render(
                        source_rgb=view["files"]["rgb"],
                        source_mask=view["files"]["mask"],
                        source_edge=view["files"]["edge"],
                        ai_image=direct_path,
                        output=locked,
                    )
                    image_path = str(locked)
                else:
                    image_path = candidate["file"]
            scores = score_structure_v2(candidate_path=image_path, source_files=view["files"])
            item = {
                "view_id": view_id,
                "recipe": recipe.name,
                "mode": recipe.mode,
                "reference_channels": list(recipe.channels),
                "appearance_reference": str(reference) if recipe.appearance else "",
                "appearance_reference_order": recipe.order,
                "stage1_manifest": stage1_manifest_path if recipe.mode != "single_call" else "",
                "steps": [args.structure_steps, args.appearance_steps] if recipe.mode != "single_call" else _step_count(recipe, args),
                "seed": int(args.seed + view_index * 10000 + recipe_index * 100),
                "image": image_path,
                "direct_image": direct_path,
                "structure_scores": scores,
                "edge_leak_score": _edge_leak_score(image_path, view["files"]["edge"], view["files"]["mask"]),
                "appearance_stats": _appearance_stats(image_path, view["files"]),
            }
            experiments.append(item)
            contacts.append(
                {
                    "label": f"{view_id} {recipe.name} total={scores['total']:.3f} leak={item['edge_leak_score']:.2f}",
                    "file": image_path,
                }
            )

    aggregate = _summarize(experiments)
    ranked = sorted(
        aggregate.items(),
        key=lambda pair: (
            pair[1]["mean_structure_total"],
            -pair[1]["mean_edge_leak_score"],
            pair[1]["mean_redness"],
        ),
        reverse=True,
    )
    best_recipe = ranked[0][0] if ranked else ""
    contact = _make_contact(contacts, root / "flux2_staged_weight_matrix_contact.png")
    report = {
        "type": "flux2_staged_weight_matrix",
        "status": "complete",
        "model_key": model_key,
        "backend": backend_name,
        "source_manifest": str(args.manifest),
        "selected_views": [str(view["view_id"]) for view in selected_views],
        "appearance_reference": str(reference),
        "capability": capability,
        "parameters": {
            "width": args.width,
            "height": args.height,
            "structure_steps": args.structure_steps,
            "appearance_steps": args.appearance_steps,
            "total_steps": args.total_steps,
            "seed": args.seed,
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
            "recipes": [recipe.name for recipe in recipes],
        },
        "interpretation": {
            "true_step_level_dual_image_weighting": False,
            "tested_fallbacks": [
                "single-call ordered multi-image references",
                "single-call lighter structure channels",
                "two-call stage approximation using the stage-1 structure image as the stage-2 rgb reference",
                "optional post-generation mesh position lock for the two-stage approximation",
            ],
            "best_recipe_by_structure_first": best_recipe,
        },
        "aggregate_by_recipe": aggregate,
        "experiments": experiments,
        "contact_sheet": str(contact),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path = root / "flux2_staged_weight_matrix_report.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Flux2 Klein staged structure/appearance reference behavior.")
    parser.add_argument("--manifest", default="outputs/flux2_klein_high_quality_car_reference/white_renders/manifest.json")
    parser.add_argument("--appearance-reference", default="outputs/flux2_klein_high_quality_car_reference/racecar_reference_image2.png")
    parser.add_argument("--output", default="outputs/flux2_klein_high_quality_car_reference/flux2_staged_weight_matrix")
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--model-key", default="flux2_klein_4b")
    parser.add_argument("--views", default="all")
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--structure-steps", type=int, default=4)
    parser.add_argument("--appearance-steps", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=8)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--recipes", default="all")
    args = parser.parse_args()
    report = run(args)
    print("Wrote report:", report["report"])
    print("Contact sheet:", report["contact_sheet"])
    print("Best recipe:", report["interpretation"]["best_recipe_by_structure_first"])
    print("True step-level dual-image weighting:", report["capability"]["true_dual_image_step_weight_schedule_supported"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
