from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


DEFAULT_STRUCTURE_WEIGHTS: dict[str, float] = {
    "silhouette_weight": 0.30,
    "edge_chamfer_weight": 0.25,
    "added_part_weight": 0.20,
    "roughness_weight": 0.15,
    "background_weight": 0.10,
}


def _clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


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
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]], axis=0)
    background = float(np.median(border))
    if background > 20:
        diff = gray.astype(np.float32) - background
        mask = diff > max(18.0, float(np.std(border)) * 2.0)
    else:
        mask = gray > 30
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_u8 > 0


def source_mask_from_render(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    return _binary_mask(path, size=size)


def foreground_mask_from_image(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
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


def _mask_iou(source: np.ndarray, candidate: np.ndarray) -> float:
    inter = float(np.logical_and(source, candidate).sum())
    union = float(np.logical_or(source, candidate).sum())
    return 0.0 if union == 0 else inter / union


def _f1(source: np.ndarray, candidate: np.ndarray) -> float:
    tp = float(np.logical_and(source, candidate).sum())
    fp = float(np.logical_and(~source, candidate).sum())
    fn = float(np.logical_and(source, ~candidate).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def _edge_chamfer_score(source_edge: np.ndarray, candidate_edge: np.ndarray, *, tolerance_px: float = 8.0) -> float:
    if not source_edge.any() and not candidate_edge.any():
        return 1.0
    if not source_edge.any() or not candidate_edge.any():
        return 0.0
    source_dist = cv2.distanceTransform((~source_edge).astype(np.uint8), cv2.DIST_L2, 3)
    candidate_dist = cv2.distanceTransform((~candidate_edge).astype(np.uint8), cv2.DIST_L2, 3)
    candidate_to_source = float(np.mean(source_dist[candidate_edge]))
    source_to_candidate = float(np.mean(candidate_dist[source_edge]))
    chamfer = (candidate_to_source + source_to_candidate) / 2.0
    return _clamp01(1.0 - chamfer / max(tolerance_px, 1e-6))


def _added_part_score(source_mask: np.ndarray, candidate_mask: np.ndarray, *, tolerance_px: int = 7) -> tuple[float, float]:
    if not candidate_mask.any():
        return 0.0, 1.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    grown_source = cv2.dilate(source_mask.astype(np.uint8), kernel, iterations=max(1, tolerance_px // 2)).astype(bool)
    added = np.logical_and(candidate_mask, ~grown_source)
    added_ratio = float(added.sum()) / max(float(candidate_mask.sum()), 1.0)
    return _clamp01(1.0 - added_ratio * 3.0), _clamp01(added_ratio)


def _roughness_score(image_path: str | Path, source_mask: np.ndarray) -> float:
    gray = _load_gray(image_path, size=(source_mask.shape[1], source_mask.shape[0]))
    if not source_mask.any():
        return 0.0
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(blurred, cv2.CV_32F)
    noise = float(np.mean(np.abs(lap[source_mask]))) / 255.0
    return _clamp01(1.0 / (1.0 + noise * 10.0))


def _background_cleanliness_score(image_path: str | Path, source_mask: np.ndarray) -> float:
    gray = _load_gray(image_path, size=(source_mask.shape[1], source_mask.shape[0]))
    bg_mask = ~cv2.dilate(source_mask.astype(np.uint8), np.ones((7, 7), dtype=np.uint8), iterations=2).astype(bool)
    if not bg_mask.any():
        return 0.0
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    residual = cv2.absdiff(gray, blurred)
    high_frequency_noise = float(np.mean(residual[bg_mask])) / 255.0
    stray_edges = float(cv2.Canny(gray, 80, 160)[bg_mask].mean()) / 255.0
    return _clamp01(1.0 - high_frequency_noise * 8.0 - stray_edges * 2.0)


def _failure_reasons(scores: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if scores["silhouette_iou"] < 0.75:
        reasons.append("silhouette_iou_below_0.75")
    if scores["edge_chamfer_score"] < 0.65:
        reasons.append("edge_chamfer_score_below_0.65")
    if scores["added_part_penalty"] > 0.15:
        reasons.append("added_part_penalty_above_0.15")
    if scores["background_cleanliness"] < 0.85:
        reasons.append("background_cleanliness_below_0.85")
    return reasons


def score_structure_v2(
    *,
    candidate_path: str | Path,
    source_files: dict[str, str],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    with Image.open(candidate_path) as candidate_image:
        size = candidate_image.size
    source_edge = _edge_map(source_files["edge"], size=size)
    candidate_edge = _edge_map(candidate_path, size=size)
    source_mask = _binary_mask(source_files["mask"], size=size)
    candidate_mask = foreground_mask_from_image(candidate_path, size=size)

    silhouette = _mask_iou(source_mask, candidate_mask)
    edge_chamfer = _edge_chamfer_score(source_edge, candidate_edge)
    added_part, added_penalty = _added_part_score(source_mask, candidate_mask)
    roughness = _roughness_score(candidate_path, source_mask)
    background = _background_cleanliness_score(candidate_path, source_mask)

    active_weights = {**DEFAULT_STRUCTURE_WEIGHTS, **(weights or {})}
    weight_sum = sum(active_weights.values()) or 1.0
    total = (
        active_weights["silhouette_weight"] * silhouette
        + active_weights["edge_chamfer_weight"] * edge_chamfer
        + active_weights["added_part_weight"] * added_part
        + active_weights["roughness_weight"] * roughness
        + active_weights["background_weight"] * background
    ) / weight_sum

    scores: dict[str, Any] = {
        "silhouette_iou": round(_clamp01(silhouette), 6),
        "edge_chamfer_score": round(_clamp01(edge_chamfer), 6),
        "added_part_score": round(_clamp01(added_part), 6),
        "added_part_penalty": round(_clamp01(added_penalty), 6),
        "roughness": round(_clamp01(roughness), 6),
        "background_cleanliness": round(_clamp01(background), 6),
        "total": round(_clamp01(total), 6),
        "weights": {key: round(float(value), 6) for key, value in active_weights.items()},
    }
    scores["mask_iou"] = scores["silhouette_iou"]
    scores["edge_f1"] = round(_clamp01(_f1(source_edge, candidate_edge)), 6)
    scores["failure_reasons"] = _failure_reasons(scores)
    return scores
