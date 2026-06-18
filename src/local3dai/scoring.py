from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .manifest import read_manifest, write_manifest
from .scoring_v2 import score_structure_v2


def _load_gray(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def _edge_map(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    gray = _load_gray(path, size=size)
    return cv2.Canny(gray, 80, 160) > 0


def _binary_mask(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    gray = _load_gray(path, size=size)
    return gray > 30


def _foreground_mask_from_image(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.int16)
    border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
    background = np.median(border, axis=0)
    diff = np.abs(arr - background).mean(axis=2)
    mask = diff > 14
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask_u8 > 0


def _f1(a: np.ndarray, b: np.ndarray) -> float:
    tp = float(np.logical_and(a, b).sum())
    fp = float(np.logical_and(~a, b).sum())
    fn = float(np.logical_and(a, ~b).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return 0.0 if union == 0 else inter / union


def score_candidates(
    *,
    render_manifest_path: str | Path,
    ai_manifest_path: str | Path,
    output_dir: str | Path,
    edge_weight: float = 0.55,
    mask_weight: float = 0.25,
    prompt_weight: float = 0.2,
    copy_top_k: int = 1,
    version: str = "legacy",
    structure_weights: dict[str, float] | None = None,
) -> Path:
    render_manifest = read_manifest(render_manifest_path)
    ai_manifest = read_manifest(ai_manifest_path)
    output = Path(output_dir)
    ranked_dir = output / "ranked"
    ranked_dir.mkdir(parents=True, exist_ok=True)

    views = {view["view_id"]: view for view in render_manifest.get("views", [])}
    rows: list[dict[str, Any]] = []
    for candidate in ai_manifest.get("candidates", []):
        view = views.get(candidate["view_id"])
        if not view:
            continue
        candidate_path = Path(candidate["file"])
        with Image.open(candidate_path) as candidate_image:
            size = candidate_image.size
        if version == "structure_v2":
            scores = score_structure_v2(
                candidate_path=candidate_path,
                source_files=view["files"],
                weights=structure_weights,
            )
            total = float(scores["total"])
        else:
            source_edge = _edge_map(view["files"]["edge"], size=size)
            candidate_edge = _edge_map(candidate_path, size=size)
            source_mask = _binary_mask(view["files"]["mask"], size=size)
            candidate_mask = _foreground_mask_from_image(candidate_path, size=size)
            edge_score = _f1(source_edge, candidate_edge)
            mask_score = _mask_iou(source_mask, candidate_mask)
            prompt_score = 0.5
            total = edge_weight * edge_score + mask_weight * mask_score + prompt_weight * prompt_score
            if math.isnan(total):
                total = 0.0
            scores = {
                "edge_f1": round(edge_score, 6),
                "mask_iou": round(mask_score, 6),
                "prompt_proxy": round(prompt_score, 6),
                "total": round(total, 6),
            }
        row = {
            **candidate,
            "scores": scores,
        }
        rows.append(row)

    rows.sort(key=lambda item: item["scores"]["total"], reverse=True)
    for rank, row in enumerate(rows[:copy_top_k], start=1):
        src = Path(row["file"])
        dst = ranked_dir / f"{rank:02d}_{row['candidate_id']}_{src.name}"
        shutil.copy2(src, dst)
        row["ranked_copy"] = str(dst)

    report = {
        "type": "score_report",
        "score_version": version,
        "weights": {
            "edge_weight": edge_weight,
            "mask_weight": mask_weight,
            "prompt_weight": prompt_weight,
            "structure_v2": structure_weights or {},
        },
        "count": len(rows),
        "ranked": rows,
    }
    write_manifest(output / "report.json", report)
    return output / "report.json"


def summarize_report(path: str | Path, limit: int = 5) -> str:
    with Path(path).open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    lines = [f"Scored {report.get('count', 0)} candidates"]
    for index, row in enumerate(report.get("ranked", [])[:limit], start=1):
        lines.append(f"{index}. {row['candidate_id']} total={row['scores']['total']} file={row['file']}")
    return "\n".join(lines)
