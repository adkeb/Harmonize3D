#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from local3dai.ai.backends import build_backend
from local3dai.config import load_config
from local3dai.manifest import read_manifest, write_manifest
from local3dai.scoring_v2 import DEFAULT_STRUCTURE_WEIGHTS, foreground_mask_from_image, score_structure_v2, source_mask_from_render


DEFAULT_CHANNELS = ("rgb", "edge", "depth", "normal", "mask", "skeleton")
DEFAULT_VIEWS = ("view_locked", "view_left_30", "view_right_30")


def _clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _model_ref(model: dict[str, Any]) -> str:
    return model.get("local_path") or model.get("model_path") or model.get("model_id") or ""


def _view_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(view.get("view_id")): view for view in manifest.get("views", [])}


def _select_views(manifest: dict[str, Any], view_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    views = _view_by_id(manifest)
    selected = [views[view_id] for view_id in view_ids if view_id in views]
    if not selected:
        selected = list(manifest.get("views", []))[:3]
    if not selected:
        raise RuntimeError("Render manifest has no views.")
    return selected


def _single_view_manifest(source_manifest: dict[str, Any], view: dict[str, Any], output: Path) -> Path:
    single = {
        **source_manifest,
        "views": [view],
        "view_graph": {"edges": []},
    }
    return write_manifest(output, single)


def _score_candidates(
    *,
    render_manifest: dict[str, Any],
    ai_manifest_path: Path,
    output_dir: Path,
    weights: dict[str, float] | None = None,
    copy_top_k: int = 0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ai_manifest = read_manifest(ai_manifest_path)
    views = _view_by_id(render_manifest)
    ranked: list[dict[str, Any]] = []
    for candidate in ai_manifest.get("candidates", []):
        view_id = str(candidate.get("view_id"))
        view = views.get(view_id)
        if not view:
            raise RuntimeError(f"Candidate view {view_id!r} is missing from render manifest.")
        scores = score_structure_v2(
            candidate_path=candidate["file"],
            source_files=view["files"],
            weights=weights,
        )
        ranked.append(
            {
                "candidate_id": candidate.get("candidate_id", ""),
                "view_id": view_id,
                "file": candidate["file"],
                "scores": scores,
                "candidate": candidate,
            }
        )
    ranked.sort(key=lambda item: float(item["scores"]["total"]), reverse=True)
    if copy_top_k:
        ranked_dir = output_dir / "ranked"
        ranked_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(ranked[:copy_top_k], start=1):
            source = Path(item["file"])
            target = ranked_dir / f"{index:02d}_{item['candidate_id']}_{source.name}"
            shutil.copy2(source, target)
            item["ranked_copy"] = str(target)
    report = {
        "type": "feature_consistency_structure_report",
        "score_version": "structure_v2",
        "weights": {**DEFAULT_STRUCTURE_WEIGHTS, **(weights or {})},
        "count": len(ranked),
        "ranked": ranked,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _mask_from_render(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    return source_mask_from_render(path, size=size)


def _edge_density(path: str | Path, size: tuple[int, int]) -> float:
    gray = Image.open(path).convert("L")
    if gray.size != size:
        gray = gray.resize(size, Image.Resampling.LANCZOS)
    edges = cv2.Canny(np.asarray(gray, dtype=np.uint8), 80, 160) > 0
    return float(edges.mean())


def _feature_stats(image_path: str | Path, source_files: dict[str, str]) -> dict[str, float]:
    image = Image.open(image_path).convert("RGB")
    size = image.size
    arr = np.asarray(image, dtype=np.float32) / 255.0
    source_mask = _mask_from_render(source_files["mask"], size)
    candidate_mask = foreground_mask_from_image(image_path, size=size)
    mask = np.logical_or(source_mask, candidate_mask)
    if not mask.any():
        return {
            "area_ratio": 0.0,
            "center_x": 0.5,
            "center_y": 0.5,
            "edge_density": 0.0,
            "body_r": 0.0,
            "body_g": 0.0,
            "body_b": 0.0,
            "dark_ratio": 0.0,
            "dark_center_x": 0.5,
            "dark_center_y": 0.5,
        }
    ys, xs = np.where(mask)
    h, w = mask.shape
    gray = arr.mean(axis=2)
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    saturation = (maxc - minc) / np.maximum(maxc, 1e-4)
    paint_mask = np.logical_and(mask, gray > 0.20)
    saturated_paint = np.logical_and(paint_mask, saturation > 0.22)
    if saturated_paint.sum() > max(12, int(mask.sum() * 0.025)):
        paint_mask = saturated_paint
    if paint_mask.any():
        body = arr[paint_mask].mean(axis=0)
    else:
        body = arr[mask].mean(axis=0)
    dark = np.logical_and(mask, gray < 0.26)
    if dark.any():
        dys, dxs = np.where(dark)
        dark_center_x = float(dxs.mean() / max(w - 1, 1))
        dark_center_y = float(dys.mean() / max(h - 1, 1))
    else:
        dark_center_x = 0.5
        dark_center_y = 0.5
    return {
        "area_ratio": float(mask.mean()),
        "center_x": float(xs.mean() / max(w - 1, 1)),
        "center_y": float(ys.mean() / max(h - 1, 1)),
        "edge_density": _edge_density(image_path, size),
        "body_r": float(body[0]),
        "body_g": float(body[1]),
        "body_b": float(body[2]),
        "dark_ratio": float(dark.sum() / max(mask.sum(), 1)),
        "dark_center_x": dark_center_x,
        "dark_center_y": dark_center_y,
    }


def _pair_shape_score(a: dict[str, float], b: dict[str, float]) -> float:
    area = 1.0 - min(abs(a["area_ratio"] - b["area_ratio"]) / max(a["area_ratio"], b["area_ratio"], 0.02), 1.0)
    edge = 1.0 - min(abs(a["edge_density"] - b["edge_density"]) / max(a["edge_density"], b["edge_density"], 0.01), 1.0)
    center = 1.0 - min(math.hypot(a["center_x"] - b["center_x"], a["center_y"] - b["center_y"]) / 0.30, 1.0)
    return _clamp01((area + edge + center) / 3.0)


def _pair_color_score(a: dict[str, float], b: dict[str, float]) -> float:
    dist = math.sqrt(
        (a["body_r"] - b["body_r"]) ** 2
        + (a["body_g"] - b["body_g"]) ** 2
        + (a["body_b"] - b["body_b"]) ** 2
    )
    return _clamp01(1.0 - min(dist / 0.35, 1.0))


def _pair_part_style_score(a: dict[str, float], b: dict[str, float]) -> float:
    ratio = 1.0 - min(abs(a["dark_ratio"] - b["dark_ratio"]) / max(a["dark_ratio"], b["dark_ratio"], 0.04), 1.0)
    center = 1.0 - min(math.hypot(a["dark_center_x"] - b["dark_center_x"], a["dark_center_y"] - b["dark_center_y"]) / 0.35, 1.0)
    return _clamp01(0.65 * ratio + 0.35 * center)


def _view_graph_edges(render_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    graph = render_manifest.get("view_graph") or {}
    edges = graph.get("edges") if isinstance(graph, dict) else []
    return list(edges or [])


def _multiview_feature_scores(items: list[dict[str, Any]], render_manifest: dict[str, Any]) -> dict[str, float]:
    if len(items) <= 1:
        return {
            "body_color_consistency": 1.0,
            "part_style_consistency": 1.0,
            "mask_edge_feature_consistency": 1.0,
            "geometry_reprojection_proxy": 1.0,
            "no_cross_view_hallucination_score": 1.0,
            "total": 1.0,
        }

    pair_indices = list(itertools.combinations(range(len(items)), 2))
    color_scores = [_pair_color_score(items[i]["features"], items[j]["features"]) for i, j in pair_indices]
    style_scores = [_pair_part_style_score(items[i]["features"], items[j]["features"]) for i, j in pair_indices]
    shape_scores = [_pair_shape_score(items[i]["features"], items[j]["features"]) for i, j in pair_indices]

    by_view = {item["view_id"]: item for item in items}
    graph_scores: list[float] = []
    graph_weights: list[float] = []
    for edge in _view_graph_edges(render_manifest):
        source = by_view.get(str(edge.get("source")))
        target = by_view.get(str(edge.get("target")))
        if not source or not target:
            continue
        structure_delta = abs(float(source["scores"]["total"]) - float(target["scores"]["total"]))
        structure_score = 1.0 - min(structure_delta / 0.30, 1.0)
        graph_score = 0.5 * structure_score + 0.5 * _pair_shape_score(source["features"], target["features"])
        graph_scores.append(_clamp01(graph_score))
        graph_weights.append(float(edge.get("overlap_ratio", 1.0)))
    if graph_scores:
        geometry_proxy = float(np.average(graph_scores, weights=graph_weights))
    else:
        geometry_proxy = float(np.mean(shape_scores))

    penalties = [float(item["scores"].get("added_part_penalty", 1.0)) for item in items]
    no_hallucination = 1.0 - min(float(np.mean(penalties)) / 0.15, 1.0)

    body_color = _clamp01(float(np.mean(color_scores)))
    part_style = _clamp01(float(np.mean(style_scores)))
    mask_edge = _clamp01(float(np.mean(shape_scores)))
    geometry_proxy = _clamp01(geometry_proxy)
    no_hallucination = _clamp01(no_hallucination)
    total = _clamp01(
        0.30 * body_color
        + 0.20 * part_style
        + 0.20 * mask_edge
        + 0.15 * geometry_proxy
        + 0.15 * no_hallucination
    )
    return {
        "body_color_consistency": round(body_color, 6),
        "part_style_consistency": round(part_style, 6),
        "mask_edge_feature_consistency": round(mask_edge, 6),
        "geometry_reprojection_proxy": round(geometry_proxy, 6),
        "no_cross_view_hallucination_score": round(no_hallucination, 6),
        "total": round(total, 6),
    }


def _make_contact(items: list[dict[str, str]], output: Path, *, tile: int = 360) -> Path:
    if not items:
        return output
    pad = 12
    label_h = 34
    canvas = Image.new("RGB", (pad + len(items) * (tile + pad), tile + label_h + pad * 2), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(items):
        image = Image.open(item["file"]).convert("RGB")
        image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        x = pad + index * (tile + pad)
        y = pad + label_h + (tile - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((x, pad + 8), item["label"], fill=(242, 242, 242))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _make_white_feature_final_comparison(
    *,
    selected_items: list[dict[str, Any]],
    render_manifest: dict[str, Any],
    anchor: Path,
    output: Path,
) -> Path:
    views = _view_by_id(render_manifest)
    items: list[dict[str, str]] = [{"label": "feature anchor", "file": str(anchor)}]
    for item in selected_items:
        view_id = item["view_id"]
        white = views[view_id]["files"]["rgb"]
        items.append({"label": f"{view_id} white", "file": white})
        items.append({"label": f"{view_id} final", "file": item["image"]})
    return _make_contact(items, output, tile=300)


def _candidate_item(candidate: dict[str, Any], render_manifest: dict[str, Any]) -> dict[str, Any]:
    view = _view_by_id(render_manifest)[candidate["view_id"]]
    return {
        "view_id": candidate["view_id"],
        "candidate_id": candidate["candidate_id"],
        "image": candidate["file"],
        "scores": candidate["scores"],
        "features": _feature_stats(candidate["file"], view["files"]),
    }


def _select_feature_combo(
    *,
    candidates_by_view: dict[str, list[dict[str, Any]]],
    render_manifest: dict[str, Any],
    view_ids: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    candidate_lists = [candidates_by_view[view_id] for view_id in view_ids if candidates_by_view.get(view_id)]
    if not candidate_lists:
        return [], _multiview_feature_scores([], render_manifest)

    best_items: list[dict[str, Any]] = []
    best_scores: dict[str, float] = {}
    best_value: tuple[int, float] = (-1, -1.0)
    for combo in itertools.product(*candidate_lists):
        items = [_candidate_item(candidate, render_manifest) for candidate in combo]
        consistency = _multiview_feature_scores(items, render_manifest)
        mean_structure = float(np.mean([float(item["scores"]["total"]) for item in items]))
        selection_score = 0.55 * mean_structure + 0.45 * consistency["total"]
        hard_gates_pass = all(
            float(item["scores"].get("silhouette_iou", 0.0)) >= 0.75
            and float(item["scores"].get("edge_chamfer_score", 0.0)) >= 0.65
            and float(item["scores"].get("added_part_penalty", 1.0)) <= 0.15
            for item in items
        )
        candidate_value = (1 if hard_gates_pass else 0, selection_score)
        if candidate_value > best_value:
            best_value = candidate_value
            best_items = items
            best_scores = {
                **consistency,
                "mean_structure": round(_clamp01(mean_structure), 6),
                "selection_score": round(_clamp01(selection_score), 6),
                "hard_gates_pass": hard_gates_pass,
            }
    return best_items, best_scores


def _baseline_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    report = read_manifest(path)
    mv = report.get("multiview_scores") or {}
    structure_scores = report.get("structure_scores") or {}
    mean_structure = mv.get("mean_structure")
    if mean_structure is None and structure_scores:
        mean_structure = float(np.mean([float(score.get("total", 0.0)) for score in structure_scores.values()]))
    return {
        "status": report.get("status", "unknown"),
        "path": str(path),
        "mean_structure": round(float(mean_structure or 0.0), 6),
        "multiview_total": round(float(mv.get("total", 0.0)), 6),
        "body_color_consistency": round(float(mv.get("body_color_consistency", 0.0)), 6),
    }


def _mesh_lock_kwargs(mode: str) -> dict[str, bool]:
    return {
        "geometry_lock": mode == "geometry",
        "mesh_position_lock": mode == "position",
        "mesh_detail_lock": mode == "detail",
        "mesh_adaptive_lock": mode == "adaptive",
        "mesh_quality_lock": mode == "quality",
    }


def run_feature_consistency(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    model_key = args.model_key or config["ai"].get("default_model_key", "flux2_klein_4b")
    model = dict(config["models"][model_key])
    backend_name = model.get("backend", config["ai"].get("default_backend", "flux2-klein"))
    if backend_name != "flux2-klein":
        raise RuntimeError(f"Feature consistency test must use Flux2 Klein, got {backend_name!r}.")
    valid_lock_modes = {"none", "geometry", "position", "detail", "adaptive", "quality"}
    if args.mesh_lock_mode not in valid_lock_modes:
        raise RuntimeError("--mesh-lock-mode must be one of: none, geometry, position, detail, adaptive, quality.")
    if args.anchor_mesh_lock_mode not in valid_lock_modes:
        raise RuntimeError("--anchor-mesh-lock-mode must be one of: none, geometry, position, detail, adaptive, quality.")
    backend = build_backend(backend_name)
    render_manifest_path = Path(args.manifest)
    render_manifest = read_manifest(render_manifest_path)
    view_ids = tuple(args.views.split(",")) if args.views else DEFAULT_VIEWS
    selected_views = _select_views(render_manifest, view_ids)
    selected_view_ids = tuple(str(view["view_id"]) for view in selected_views)
    score_cfg = config["score"]
    score_weights = dict(score_cfg.get("structure_v2", {}))
    channels = list(DEFAULT_CHANNELS)
    detail_reference = Path(args.detail_reference).expanduser().resolve() if args.detail_reference else None
    if detail_reference and not detail_reference.exists():
        raise RuntimeError(f"Detail reference image does not exist: {detail_reference}")
    detail_reference_images = [detail_reference] if detail_reference else None

    anchor_view = _view_by_id(render_manifest).get(args.anchor_view) or selected_views[0]
    anchor_manifest = _single_view_manifest(render_manifest, anchor_view, root / "anchor_render_manifest.json")
    anchor_manifest_data = read_manifest(anchor_manifest)
    anchor_dir = root / "anchor_candidates"
    anchor_path = root / "feature_reference_anchor.png"
    external_anchor = Path(args.appearance_anchor).expanduser().resolve() if args.appearance_anchor else None
    if external_anchor:
        if not external_anchor.exists():
            raise RuntimeError(f"Appearance anchor does not exist: {external_anchor}")
        shutil.copy2(external_anchor, anchor_path)
        anchor_scores = score_structure_v2(candidate_path=anchor_path, source_files=anchor_view["files"], weights=score_weights)
        best_anchor = {
            "candidate_id": "external_anchor",
            "file": str(anchor_path),
            "scores": anchor_scores,
        }
        anchor_report = {
            "type": "external_anchor_score",
            "score_version": "structure_v2",
            "source": str(external_anchor),
            "ranked": [best_anchor],
        }
        anchor_dir.mkdir(parents=True, exist_ok=True)
        (anchor_dir / "external_anchor_report.json").write_text(
            json.dumps(anchor_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        anchor_contact = _make_contact(
            [{"label": f"external anchor {anchor_scores['total']:.3f}", "file": str(anchor_path)}],
            root / "feature_reference_contact_sheet.png",
            tile=300,
        )
    else:
        anchor_ai_manifest = backend.generate(
            anchor_manifest,
            anchor_dir / "candidates",
            prompt=args.anchor_prompt or args.prompt,
            negative_prompt=args.negative_prompt or model.get("negative_prompt", ""),
            candidates_per_view=args.anchor_candidates,
            seed=args.seed,
            model_ref=_model_ref(model),
            model_config={**model, "reference_channels": channels, "geometry_lock": False},
            device=args.device or config["ai"].get("device", "cuda:0"),
            dtype=args.dtype or model.get("dtype") or config["ai"].get("dtype", "bfloat16"),
            steps=args.steps or int(model.get("steps", config["ai"].get("steps", 4))),
            guidance_scale=args.guidance_scale if args.guidance_scale is not None else float(model.get("guidance_scale", config["ai"].get("guidance_scale", 1.0))),
            width=args.width or int(model.get("width", 1024)),
            height=args.height or int(model.get("height", 1024)),
            reference_channels=channels,
            appearance_reference_images=detail_reference_images,
            appearance_reference_order=args.appearance_reference_order,
            detail_reference=detail_reference,
            **_mesh_lock_kwargs(args.anchor_mesh_lock_mode),
        )
        anchor_report = _score_candidates(
            render_manifest=anchor_manifest_data,
            ai_manifest_path=anchor_ai_manifest,
            output_dir=anchor_dir / "score",
            weights=score_weights,
            copy_top_k=args.anchor_candidates,
        )
        if not anchor_report["ranked"]:
            raise RuntimeError("Anchor generation produced no candidates.")
        best_anchor = anchor_report["ranked"][0]
        shutil.copy2(best_anchor.get("ranked_copy") or best_anchor["file"], anchor_path)
        anchor_contact = _make_contact(
            [
                {
                    "label": f"{item['candidate_id']} {item['scores']['total']:.3f}",
                    "file": item.get("ranked_copy") or item["file"],
                }
                for item in anchor_report["ranked"]
            ],
            root / "feature_reference_contact_sheet.png",
            tile=300,
        )

    candidates_by_view: dict[str, list[dict[str, Any]]] = {}
    view_reports: dict[str, Any] = {}
    for index, view in enumerate(selected_views):
        view_id = str(view["view_id"])
        view_dir = root / "view_candidates" / view_id
        single_manifest = _single_view_manifest(render_manifest, view, view_dir / "render_manifest.json")
        ai_manifest = backend.generate(
            single_manifest,
            view_dir / "candidates",
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or model.get("negative_prompt", ""),
            candidates_per_view=args.candidates_per_view,
            seed=args.seed + 10000 + index * 1000,
            model_ref=_model_ref(model),
            model_config={**model, "reference_channels": channels, "geometry_lock": False},
            device=args.device or config["ai"].get("device", "cuda:0"),
            dtype=args.dtype or model.get("dtype") or config["ai"].get("dtype", "bfloat16"),
            steps=args.steps or int(model.get("steps", config["ai"].get("steps", 4))),
            guidance_scale=args.guidance_scale if args.guidance_scale is not None else float(model.get("guidance_scale", config["ai"].get("guidance_scale", 1.0))),
            width=args.width or int(model.get("width", 1024)),
            height=args.height or int(model.get("height", 1024)),
            reference_channels=channels,
            appearance_reference=anchor_path,
            appearance_reference_images=detail_reference_images,
            appearance_reference_order=args.appearance_reference_order,
            **_mesh_lock_kwargs(args.mesh_lock_mode),
            detail_reference=detail_reference,
        )
        report = _score_candidates(
            render_manifest=render_manifest,
            ai_manifest_path=ai_manifest,
            output_dir=view_dir / "score",
            weights=score_weights,
            copy_top_k=args.candidates_per_view,
        )
        view_reports[view_id] = {
            "ai_manifest": str(ai_manifest),
            "score_report": str(view_dir / "score" / "report.json"),
            "ranked": report["ranked"],
        }
        candidates_by_view[view_id] = report["ranked"]

    selected_items, consistency_scores = _select_feature_combo(
        candidates_by_view=candidates_by_view,
        render_manifest=render_manifest,
        view_ids=selected_view_ids,
    )
    if not selected_items:
        raise RuntimeError("No multiview candidate combination could be selected.")

    final_view_images: dict[str, str] = {}
    for item in selected_items:
        target = root / f"final_{item['view_id']}.png"
        shutil.copy2(item["image"], target)
        final_view_images[item["view_id"]] = str(target)
    multiview_contact = _make_contact(
        [{"label": item["view_id"], "file": final_view_images[item["view_id"]]} for item in selected_items],
        root / "multiview_feature_consistency_contact.png",
        tile=420,
    )
    comparison = _make_white_feature_final_comparison(
        selected_items=selected_items,
        render_manifest=render_manifest,
        anchor=anchor_path,
        output=root / "white_feature_final_comparison.png",
    )

    baseline = _baseline_summary(Path(args.baseline_report))
    mean_structure = float(consistency_scores.get("mean_structure", 0.0))
    body_color = float(consistency_scores.get("body_color_consistency", 0.0))
    baseline_mean = float(baseline.get("mean_structure", 0.0))
    baseline_body = float(baseline.get("body_color_consistency", 0.0))
    acceptance = {
        "anchor_total_at_least_0_80": float(best_anchor["scores"]["total"]) >= 0.80,
        "selected_views_silhouette_iou_at_least_0_75": all(float(item["scores"]["silhouette_iou"]) >= 0.75 for item in selected_items),
        "selected_views_added_part_penalty_at_most_0_15": all(float(item["scores"].get("added_part_penalty", 1.0)) <= 0.15 for item in selected_items),
        "selected_views_edge_chamfer_at_least_0_65": all(float(item["scores"].get("edge_chamfer_score", 0.0)) >= 0.65 for item in selected_items),
        "mean_structure_not_more_than_0_05_below_baseline": baseline["status"] != "missing" and mean_structure + 0.05 >= baseline_mean,
        "body_color_consistency_at_least_baseline": baseline["status"] != "missing" and body_color >= baseline_body,
    }
    status = "complete" if all(acceptance.values()) else "needs_review"
    report = {
        "type": "flux2_feature_consistency",
        "status": status,
        "model_key": model_key,
        "backend": backend_name,
        "source_manifest": str(render_manifest_path),
        "source_model_path": render_manifest.get("source", ""),
        "views": list(selected_view_ids),
        "reference_policy": "appearance_anchor_plus_model_render_channels",
        "appearance_reference_files": [str(anchor_path)],
        "appearance_reference_order": args.appearance_reference_order,
        "detail_reference_file": str(detail_reference) if detail_reference else "",
        "mesh_lock_mode": args.mesh_lock_mode,
        "anchor_mesh_lock_mode": args.anchor_mesh_lock_mode,
        "reference_channels": channels,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "parameters": {
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "seed": args.seed,
            "anchor_candidates": args.anchor_candidates,
            "candidates_per_view": args.candidates_per_view,
        },
        "anchor": {
            "source": str(external_anchor) if external_anchor else "generated",
            "file": str(anchor_path),
            "score": best_anchor["scores"],
            "candidate_id": best_anchor["candidate_id"],
            "contact_sheet": str(anchor_contact),
            "score_report": str(anchor_dir / "external_anchor_report.json" if external_anchor else anchor_dir / "score" / "report.json"),
        },
        "selected_views": selected_items,
        "final_view_images": final_view_images,
        "multiview_scores": consistency_scores,
        "baseline": baseline,
        "baseline_delta": {
            "mean_structure": round(mean_structure - baseline_mean, 6) if baseline["status"] != "missing" else None,
            "body_color_consistency": round(body_color - baseline_body, 6) if baseline["status"] != "missing" else None,
            "multiview_total": round(float(consistency_scores.get("total", 0.0)) - float(baseline.get("multiview_total", 0.0)), 6)
            if baseline["status"] != "missing"
            else None,
        },
        "acceptance": acceptance,
        "view_reports": view_reports,
        "feature_reference_contact_sheet": str(anchor_contact),
        "multiview_feature_consistency_contact": str(multiview_contact),
        "white_feature_final_comparison": str(comparison),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path = root / "consistency_report.json"
    report["consistency_report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Flux2 Klein appearance-anchor multiview consistency.")
    parser.add_argument("--manifest", default="outputs/flux2_klein_high_quality_car_reference/white_renders/manifest.json")
    parser.add_argument("--output", default="outputs/flux2_klein_high_quality_car_reference/flux2_feature_consistency")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--anchor-prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--config", default="configs/local.json")
    parser.add_argument("--model-key", default="flux2_klein_4b")
    parser.add_argument("--baseline-report", default="outputs/flux2_klein_high_quality_car_reference/flux2_matrix_new_white/multiview_agent/agent_report.json")
    parser.add_argument("--views", default="view_locked,view_left_30,view_right_30")
    parser.add_argument("--anchor-view", default="view_locked")
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--device")
    parser.add_argument("--dtype")
    parser.add_argument("--anchor-candidates", type=int, default=4)
    parser.add_argument("--candidates-per-view", type=int, default=3)
    parser.add_argument("--detail-reference", default="")
    parser.add_argument("--appearance-anchor", default="")
    parser.add_argument("--appearance-reference-order", choices=["before", "after"], default="after")
    parser.add_argument("--mesh-lock-mode", choices=["none", "geometry", "position", "detail", "adaptive", "quality"], default="position")
    parser.add_argument("--anchor-mesh-lock-mode", choices=["none", "geometry", "position", "detail", "adaptive", "quality"], default="none")
    args = parser.parse_args()
    report = run_feature_consistency(args)
    print("Wrote consistency report:", report["consistency_report"])
    print("Status:", report["status"])
    print("Feature anchor:", report["anchor"]["file"])
    print("Multiview contact:", report["multiview_feature_consistency_contact"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
