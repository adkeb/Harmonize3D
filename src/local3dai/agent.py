from __future__ import annotations

import json
import itertools
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .ai.backends import build_backend
from .ai.geometry import create_comparison_image
from .config import load_config
from .manifest import read_manifest, write_manifest
from .scoring_v2 import DEFAULT_STRUCTURE_WEIGHTS, score_structure_v2


Progress = Callable[[str, str, float], None]

DEFAULT_MODEL_KEY = "flux2_klein_4b"
DEFAULT_REFERENCE_CHANNELS = ("rgb", "edge", "depth", "normal", "mask", "skeleton")
DEFAULT_EXPAND_VIEW_IDS = ("view_locked", "view_left_30", "view_right_30")


@dataclass
class AgentRunOptions:
    input_renders: Path
    output_dir: Path
    prompt: str
    config_path: Path = Path("configs/local.json")
    model_key: str = DEFAULT_MODEL_KEY
    backend: str | None = None
    model_ref: str | None = None
    target_view: str = "view_locked"
    max_generations: int = 10
    seed: int = 20260610
    expand_views: bool = True
    expand_view_ids: tuple[str, ...] = DEFAULT_EXPAND_VIEW_IDS
    default_reference_channels: tuple[str, ...] = DEFAULT_REFERENCE_CHANNELS
    experimental_reference_channels: tuple[str, ...] = ()
    pass_threshold: float = 0.62
    roughness_weight: float = 0.25
    edge_weight: float = 0.35
    mask_weight: float = 0.25
    background_weight: float = 0.15
    negative_prompt: str = ""
    device: str | None = None
    dtype: str | None = None
    variant: str | None = None
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    geometry_lock: bool | None = None
    mesh_position_lock: bool | None = None
    mesh_detail_lock: bool | None = None
    mesh_adaptive_lock: bool | None = None
    mesh_quality_lock: bool | None = None


@dataclass
class AgentTrial:
    trial_id: str
    view_id: str
    prompt_variant: str
    prompt: str
    reference_channels: list[str]
    seed: int
    steps: int
    guidance_scale: float
    output_file: str
    score_file: str
    scores: dict[str, Any]
    decision_reason: str


@dataclass
class AgentDecision:
    status: str
    selected_trial: str
    selected_score: float
    decision_notes: list[str] = field(default_factory=list)


@dataclass
class AgentRunSummary:
    type: str
    status: str
    output_dir: str
    prompt: str
    source_model_path: str
    render_manifest: str
    reference_policy: str
    target_view: str
    budget: dict[str, Any]
    trials: list[dict[str, Any]]
    selected_trial: dict[str, Any]
    expanded_views: list[dict[str, Any]]
    final_image: str
    comparison_image: str
    three_view_contact: str
    agent_report: str
    decision_notes: list[str]
    elapsed_seconds: float
    model_key: str = ""
    backend: str = ""
    score_version: str = "structure_v2"
    structure_scores: dict[str, Any] = field(default_factory=dict)
    multiview_scores: dict[str, Any] = field(default_factory=dict)
    retry_decisions: list[dict[str, Any]] = field(default_factory=list)
    final_view_images: dict[str, str] = field(default_factory=dict)
    multiview_contact_sheet: str = ""


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


def _f1(source: np.ndarray, candidate: np.ndarray) -> float:
    tp = float(np.logical_and(source, candidate).sum())
    fp = float(np.logical_and(~source, candidate).sum())
    fn = float(np.logical_and(source, ~candidate).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def _tolerant_edge_f1(source: np.ndarray, candidate: np.ndarray) -> float:
    kernel = np.ones((5, 5), dtype=np.uint8)
    source_u8 = source.astype(np.uint8)
    candidate_u8 = candidate.astype(np.uint8)
    source_wide = cv2.dilate(source_u8, kernel, iterations=1).astype(bool)
    candidate_wide = cv2.dilate(candidate_u8, kernel, iterations=1).astype(bool)
    recall_like = _f1(source_wide, candidate)
    precision_like = _f1(source, candidate_wide)
    return (recall_like + precision_like) / 2.0


def _mask_iou(source: np.ndarray, candidate: np.ndarray) -> float:
    inter = float(np.logical_and(source, candidate).sum())
    union = float(np.logical_or(source, candidate).sum())
    return 0.0 if union == 0 else inter / union


def _clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _roughness_score(image_path: str | Path, source_mask: np.ndarray) -> float:
    gray = _load_gray(image_path, size=(source_mask.shape[1], source_mask.shape[0]))
    foreground = source_mask
    if not foreground.any():
        return 0.0
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(blurred, cv2.CV_32F)
    noise = float(np.mean(np.abs(lap[foreground]))) / 255.0
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


def _view_features(image_path: str | Path, source_mask_path: str | Path) -> dict[str, float]:
    with Image.open(image_path) as image:
        size = image.size
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    candidate_mask = _foreground_mask_from_image(image_path, size=size)
    source_mask = _binary_mask(source_mask_path, size=size)
    mask = np.logical_or(candidate_mask, source_mask)
    edge_density = float(_edge_map(image_path, size=size).mean())
    if not mask.any():
        return {
            "area_ratio": 0.0,
            "edge_density": edge_density,
            "center_x": 0.5,
            "center_y": 0.5,
            "body_r": 0.0,
            "body_g": 0.0,
            "body_b": 0.0,
        }
    ys, xs = np.where(mask)
    h, w = mask.shape
    body_color = arr[mask].mean(axis=0) / 255.0
    return {
        "area_ratio": float(mask.mean()),
        "edge_density": edge_density,
        "center_x": float(xs.mean() / max(w - 1, 1)),
        "center_y": float(ys.mean() / max(h - 1, 1)),
        "body_r": float(body_color[0]),
        "body_g": float(body_color[1]),
        "body_b": float(body_color[2]),
    }


def _score_candidate(
    *,
    candidate_path: str | Path,
    source_files: dict[str, str],
    structure_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    return score_structure_v2(
        candidate_path=candidate_path,
        source_files=source_files,
        weights=structure_weights,
    )


def _choose_view(manifest: dict[str, Any], preferred_view_id: str) -> dict[str, Any]:
    views = list(manifest.get("views", []))
    if not views:
        raise RuntimeError("Render manifest has no views.")
    for view in views:
        if view.get("view_id") == preferred_view_id:
            return view
    return views[0]


def _expand_view_list(manifest: dict[str, Any], target_view_id: str, requested_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    views = list(manifest.get("views", []))
    by_id = {view.get("view_id"): view for view in views}
    selected: list[dict[str, Any]] = []
    for view_id in requested_ids:
        view = by_id.get(view_id)
        if view and view not in selected:
            selected.append(view)
    target = by_id.get(target_view_id)
    if target and target not in selected:
        selected.insert(0, target)
    for view in views:
        if len(selected) >= min(3, len(views)):
            break
        if view not in selected:
            selected.append(view)
    return selected[:3]


def _write_single_view_manifest(source_manifest: dict[str, Any], view: dict[str, Any], output: Path) -> Path:
    single = {
        **{key: value for key, value in source_manifest.items() if key != "views"},
        "type": "render_manifest",
        "views": [view],
    }
    return write_manifest(output, single)


def _prompt_variant(base_prompt: str, variant: str) -> str:
    base = base_prompt.strip()
    scene_mode = any(token in base.lower() for token in ("assembled 3d scene", "multi-module", "scene layout", "booth", "showroom", "exhibition"))
    if scene_mode:
        variants = {
            "product_quality": (
                "Render the exact same multi-object scene shown in the reference channels. Preserve every module, "
                "including the hero vehicle, platform, screens, robotic arms, light strips, floor plane, camera angle, "
                "scale relationships, and visible silhouettes. Produce a premium commercial CGI scene render with "
                "clean materials, crisp reflections, controlled futuristic lighting, and no layout changes. LED screens "
                "must remain blank abstract blue glow panels with no readable text, letters, numbers, UI, icons, people, "
                "portraits, logos, or interface graphics."
            ),
            "clean_studio": (
                "Create a clean high-end visualization of the entire assembled scene, not an isolated object. Keep all "
                "reference modules visible in their original positions and proportions. Use polished showroom materials, "
                "cool accent lights, a tidy reflective floor, blank abstract screen glow, and a professional uncluttered background."
            ),
            "geometry_first": (
                "Scene geometry preservation is the highest priority. Do not remove the screens, platform, robotic arms, "
                "light strips, or floor. Do not crop into a single object. Only improve materials, lighting, reflections, "
                "and render polish while respecting every contour and placement from the render channels. Do not invent "
                "screen UI, text, people, icons, numbers, or logos."
            ),
        }
        if base:
            return f"{base}\n\n{variants[variant]}"
        return variants[variant]
    variants = {
        "product_quality": (
            "Render the exact same 3D object shown in the reference images. Preserve the silhouette, "
            "proportions, holes, cutouts, hard-surface planes, bevels, wheel arches, cabin volume, spoiler, "
            "vents, and all visible edges. High-end automotive product render, clean glossy painted body, "
            "polished black glass, precise panel gaps, controlled studio lighting, crisp reflections, "
            "smooth high-resolution finish, no gritty texture, no random surface noise."
        ),
        "clean_studio": (
            "Create a clean studio product visualization from the reference geometry only. Keep the object shape "
            "locked to the white model, with the same outline, scale, camera angle, openings, and edge structure. "
            "Use premium material shading, smooth clearcoat, subtle realistic reflections, uncluttered background, "
            "sharp but not oversharpened details, noise-free high quality commercial rendering."
        ),
        "geometry_first": (
            "Geometry preservation is the highest priority. Do not redesign the object, do not alter the silhouette, "
            "do not add or remove parts, do not close open holes, and do not change the camera perspective. "
            "Only assign believable material, paint, glass, trim, soft studio light, and clean render polish while "
            "respecting every contour and edge in the white model references."
        ),
    }
    if base:
        return f"{base}\n\n{variants[variant]}"
    return variants[variant]


def _negative_prompt(configured: str) -> str:
    parts = [
        configured.strip(),
        "geometry drift, changed silhouette, extra parts, missing parts, closed holes, warped body, noisy texture, gritty surface, low resolution, blur, readable text, letters, numbers, UI, icons, people, portraits, watermark, logo",
    ]
    return ", ".join(part for part in parts if part)


def _model_config(config: dict[str, Any], model_key: str) -> dict[str, Any]:
    model = config.get("models", {}).get(model_key)
    if not model:
        raise RuntimeError(f"Unknown model key: {model_key}")
    return dict(model)


def _generation_kwargs(
    *,
    config: dict[str, Any],
    model_cfg: dict[str, Any],
    options: AgentRunOptions,
    prompt: str,
    seed: int,
    guidance_scale: float,
) -> dict[str, Any]:
    ai_cfg = config.get("ai", {})
    geometry_lock = bool(model_cfg.get("geometry_lock", True)) if options.geometry_lock is None else bool(options.geometry_lock)
    return {
        "prompt": prompt,
        "negative_prompt": _negative_prompt(options.negative_prompt or model_cfg.get("negative_prompt", "")),
        "candidates_per_view": 1,
        "seed": seed,
        "model_ref": options.model_ref or model_cfg.get("local_path") or model_cfg.get("model_path") or model_cfg.get("model_id", ""),
        "model_config": model_cfg,
        "device": options.device or ai_cfg.get("device", "cuda:0"),
        "dtype": options.dtype or ai_cfg.get("dtype", "float16"),
        "variant": options.variant if options.variant is not None else model_cfg.get("variant") or ai_cfg.get("variant"),
        "steps": options.steps if options.steps is not None else int(model_cfg.get("steps", ai_cfg.get("steps", 50))),
        "guidance_scale": guidance_scale,
        "strength": float(model_cfg.get("strength", ai_cfg.get("strength", 0.68))),
        "width": options.width if options.width is not None else int(model_cfg.get("width", ai_cfg.get("width", 2048))),
        "height": options.height if options.height is not None else int(model_cfg.get("height", ai_cfg.get("height", 2048))),
        "canny_scale": float(model_cfg.get("canny_scale", 2.85)),
        "depth_scale": float(model_cfg.get("depth_scale", 0.55)),
        "control_channels": list(model_cfg.get("control_channels", ["canny", "depth"])),
        "reference_channels": list(model_cfg.get("reference_channels", [])),
        "control_only": bool(model_cfg.get("control_only", True)),
        "geometry_lock": geometry_lock,
        "mesh_position_lock": bool(options.mesh_position_lock) if options.mesh_position_lock is not None else bool(model_cfg.get("mesh_position_lock", False)),
        "mesh_detail_lock": bool(options.mesh_detail_lock) if options.mesh_detail_lock is not None else bool(model_cfg.get("mesh_detail_lock", False)),
        "mesh_adaptive_lock": bool(options.mesh_adaptive_lock) if options.mesh_adaptive_lock is not None else bool(model_cfg.get("mesh_adaptive_lock", False)),
        "mesh_quality_lock": bool(options.mesh_quality_lock) if options.mesh_quality_lock is not None else bool(model_cfg.get("mesh_quality_lock", False)),
        "max_sequence_length": int(model_cfg.get("max_sequence_length", 512)),
    }


def _trial_decision(scores: dict[str, Any], pass_threshold: float) -> str:
    if scores["total"] >= pass_threshold:
        return "passes structure quality threshold"
    if scores.get("added_part_penalty", 0.0) > 0.15:
        return "added part penalty triggered geometry-first retry"
    if scores.get("silhouette_iou", scores.get("mask_iou", 0.0)) < 0.75:
        return "silhouette mismatch triggered geometry-first retry"
    if scores.get("edge_chamfer_score", 0.0) < 0.65:
        return "edge chamfer drift triggered canny-guided retry"
    if scores.get("roughness", 0.0) < 0.55 or scores.get("background_cleanliness", 0.0) < 0.55:
        return "roughness/background penalty triggered cleaner prompt retry"
    return "below threshold, keeping best candidate for review"


def _run_trial(
    *,
    backend: Any,
    source_manifest: dict[str, Any],
    view: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
    model_cfg: dict[str, Any],
    options: AgentRunOptions,
    trial_number: int,
    prompt_variant: str,
    guidance_scale: float,
    seed: int,
    structure_weights: dict[str, float] | None = None,
) -> AgentTrial:
    trial_id = f"trial_{trial_number:02d}_{view['view_id']}_{prompt_variant}"
    trial_dir = output_dir / "agent_trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    single_manifest = _write_single_view_manifest(source_manifest, view, trial_dir / "render_manifest.json")
    prompt = _prompt_variant(options.prompt, prompt_variant)
    manifest_path = backend.generate(
        single_manifest,
        trial_dir / "candidates",
        **_generation_kwargs(
            config=config,
            model_cfg=model_cfg,
            options=options,
            prompt=prompt,
            seed=seed,
            guidance_scale=guidance_scale,
        ),
    )
    ai_manifest = read_manifest(manifest_path)
    candidates = ai_manifest.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Backend produced no candidates for {trial_id}.")
    candidate = candidates[0]
    scores = _score_candidate(
        candidate_path=candidate["file"],
        source_files=view["files"],
        structure_weights=structure_weights,
    )
    score_report = {
        "type": "agent_score",
        "trial_id": trial_id,
        "view_id": view["view_id"],
        "candidate": candidate,
        "scores": scores,
        "weights": {
            "structure_v2": structure_weights or DEFAULT_STRUCTURE_WEIGHTS,
        },
    }
    score_path = write_manifest(trial_dir / "agent_score.json", score_report)
    return AgentTrial(
        trial_id=trial_id,
        view_id=view["view_id"],
        prompt_variant=prompt_variant,
        prompt=prompt,
        reference_channels=list(model_cfg.get("reference_channels", DEFAULT_REFERENCE_CHANNELS)),
        seed=seed,
        steps=int(_generation_kwargs(
            config=config,
            model_cfg=model_cfg,
            options=options,
            prompt=prompt,
            seed=seed,
            guidance_scale=guidance_scale,
        )["steps"]),
        guidance_scale=guidance_scale,
        output_file=str(candidate["file"]),
        score_file=str(score_path),
        scores=scores,
        decision_reason=_trial_decision(scores, options.pass_threshold),
    )


def _multiview_consistency_scores(items: list[dict[str, Any]]) -> dict[str, float]:
    if len(items) <= 1:
        return {
            "body_color_consistency": 1.0,
            "mask_edge_feature_consistency": 1.0,
            "total": 1.0,
        }
    features = [item["features"] for item in items]
    ref = features[0]
    shape_scores: list[float] = []
    color_scores: list[float] = []
    for feature in features[1:]:
        area = 1.0 - min(abs(feature["area_ratio"] - ref["area_ratio"]) / max(ref["area_ratio"], 0.02), 1.0)
        edge = 1.0 - min(abs(feature["edge_density"] - ref["edge_density"]) / max(ref["edge_density"], 0.01), 1.0)
        center_dist = math.hypot(feature["center_x"] - ref["center_x"], feature["center_y"] - ref["center_y"])
        center = 1.0 - min(center_dist / 0.25, 1.0)
        color_dist = math.sqrt(
            (feature["body_r"] - ref["body_r"]) ** 2
            + (feature["body_g"] - ref["body_g"]) ** 2
            + (feature["body_b"] - ref["body_b"]) ** 2
        )
        shape_scores.append((area + edge + center) / 3.0)
        color_scores.append(1.0 - min(color_dist / 0.35, 1.0))
    body_color = _clamp01(float(np.mean(color_scores)))
    mask_edge = _clamp01(float(np.mean(shape_scores)))
    total = _clamp01(0.55 * body_color + 0.45 * mask_edge)
    return {
        "body_color_consistency": round(body_color, 6),
        "mask_edge_feature_consistency": round(mask_edge, 6),
        "total": round(total, 6),
    }


def _make_contact_sheet(items: list[dict[str, Any]], output: Path) -> Path:
    if not items:
        return output
    tile = 720
    label_h = 48
    pad = 24
    width = len(items) * tile + (len(items) + 1) * pad
    height = tile + label_h + pad * 2
    canvas = Image.new("RGB", (width, height), (16, 16, 18))
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(items):
        image = Image.open(item["image"]).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        x = pad + index * (tile + pad)
        canvas.paste(image, (x, pad + label_h))
        draw.text((x, pad + 14), item["view_id"], fill=(242, 242, 242))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _trial_item(trial: AgentTrial, view_files: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        "view_id": trial.view_id,
        "trial_id": trial.trial_id,
        "image": trial.output_file,
        "scores": trial.scores,
        "features": _view_features(trial.output_file, view_files[trial.view_id]["mask"]),
    }


def _select_multiview_combo(
    *,
    view_group: list[dict[str, Any]],
    trials_by_view: dict[str, list[AgentTrial]],
    view_files: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    candidate_lists: list[list[AgentTrial]] = []
    for view in view_group:
        candidates = trials_by_view.get(view["view_id"], [])
        if not candidates:
            continue
        candidate_lists.append(candidates)
    if not candidate_lists:
        return [], _multiview_consistency_scores([])

    best_items: list[dict[str, Any]] = []
    best_scores: dict[str, float] = {}
    best_value = -1.0
    for combo in itertools.product(*candidate_lists):
        items = [_trial_item(trial, view_files) for trial in combo]
        multiview_scores = _multiview_consistency_scores(items)
        mean_structure = float(np.mean([float(trial.scores["total"]) for trial in combo]))
        selection_score = 0.65 * mean_structure + 0.35 * multiview_scores["total"]
        if selection_score > best_value:
            best_value = selection_score
            best_items = items
            best_scores = {
                **multiview_scores,
                "mean_structure": round(_clamp01(mean_structure), 6),
                "selection_score": round(_clamp01(selection_score), 6),
            }
    return best_items, best_scores


def _serializable_trial(trial: AgentTrial) -> dict[str, Any]:
    return asdict(trial)


def run_agent_render(options: AgentRunOptions, progress: Progress | None = None) -> dict[str, Any]:
    def emit(message: str, fraction: float) -> None:
        if progress:
            progress("agent", message, fraction)

    started = time.time()
    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(options.config_path)
    agent_cfg = config.get("agent", {})
    ai_cfg = config.get("ai", {})
    model_key = (
        options.model_key
        or agent_cfg.get("default_model_key")
        or ai_cfg.get("default_model_key")
        or DEFAULT_MODEL_KEY
    )
    model_cfg = _model_config(config, model_key)
    configured_channels = tuple(options.default_reference_channels or agent_cfg.get("default_reference_channels", DEFAULT_REFERENCE_CHANNELS))
    model_cfg["reference_channels"] = list(configured_channels or options.default_reference_channels)
    backend_name = options.backend or model_cfg.get("backend") or config.get("ai", {}).get("default_backend", "mock")
    backend = build_backend(backend_name)
    source_manifest = read_manifest(options.input_renders)
    source_model_path = str(source_manifest.get("source", ""))
    target_view = _choose_view(source_manifest, options.target_view)
    max_generations = max(1, int(options.max_generations))
    candidates_per_view = max(1, int(agent_cfg.get("candidates_per_view", 3)))
    score_cfg = config.get("score", {})
    score_version = str(score_cfg.get("version", "structure_v2"))
    structure_weights = {**DEFAULT_STRUCTURE_WEIGHTS, **dict(score_cfg.get("structure_v2", {}))}
    view_group = (
        _expand_view_list(source_manifest, target_view["view_id"], options.expand_view_ids)
        if options.expand_views
        else [target_view]
    )
    notes: list[str] = [
        f"Using backend={backend_name}, model_key={model_key}, reference_channels={model_cfg['reference_channels']}.",
        f"AI references are restricted to Blender render channels from source_model_path={source_model_path}.",
        f"Score version={score_version}; structure scoring checks silhouette, edge alignment, added parts, roughness, and background cleanliness.",
    ]

    trials: list[AgentTrial] = []
    base_guidance = float(model_cfg.get("guidance_scale", config.get("ai", {}).get("guidance_scale", 2.2)))
    trial_specs = [
        ("product_quality", base_guidance),
        ("clean_studio", max(1.0, base_guidance * 0.85)),
        ("geometry_first", max(1.0, base_guidance * 0.70)),
    ]
    emit(f"Target view selected: {target_view['view_id']}; planning {len(view_group)} MeshLock views", 0.04)
    for candidate_index in range(candidates_per_view):
        variant, guidance = trial_specs[candidate_index % len(trial_specs)]
        for view_index, view in enumerate(view_group):
            if len(trials) >= max_generations:
                notes.append("Stopped candidate search at max generation budget.")
                break
            trial_number = len(trials) + 1
            progress_fraction = 0.06 + 0.72 * (trial_number / max(max_generations, 1))
            emit(f"Running {variant} trial on {view['view_id']}", min(progress_fraction, 0.82))
            trial = _run_trial(
                backend=backend,
                source_manifest=source_manifest,
                view=view,
                output_dir=output_dir,
                config=config,
                model_cfg=model_cfg,
                options=options,
                trial_number=trial_number,
                prompt_variant=variant,
                guidance_scale=guidance,
                seed=options.seed + candidate_index * 997 + view_index * 10000,
                structure_weights=structure_weights,
            )
            trials.append(trial)
            failures = trial.scores.get("failure_reasons") or []
            notes.append(
                f"{trial.trial_id}: total={trial.scores['total']} reason={trial.decision_reason}"
                f" failures={failures}."
            )
        if len(trials) >= max_generations:
            break

    if not trials:
        raise RuntimeError("Agent produced no trials.")

    trials_by_view: dict[str, list[AgentTrial]] = {}
    for trial in trials:
        trials_by_view.setdefault(trial.view_id, []).append(trial)
    view_files = {view["view_id"]: view["files"] for view in source_manifest.get("views", [])}
    expanded, multiview_scores = _select_multiview_combo(
        view_group=view_group,
        trials_by_view=trials_by_view,
        view_files=view_files,
    )
    selected_by_view: dict[str, AgentTrial] = {
        view_id: max(items, key=lambda item: float(item.scores["total"]))
        for view_id, items in trials_by_view.items()
    }
    expanded_trial_ids = {item["trial_id"] for item in expanded}
    selected_trials = [trial for trial in trials if trial.trial_id in expanded_trial_ids]
    selected = next((trial for trial in selected_trials if trial.view_id == target_view["view_id"]), None)
    if selected is None:
        selected = selected_by_view.get(target_view["view_id"]) or max(trials, key=lambda item: float(item.scores["total"]))
    if not expanded:
        expanded = [
            _trial_item(selected, view_files)
        ]

    structure_scores = {item["view_id"]: item["scores"] for item in expanded}
    if len(expanded) > 1:
        notes.append(
            "Selected multiview combination "
            f"{[item['trial_id'] for item in expanded]} with selection_score={multiview_scores.get('selection_score')}."
        )
    status = "complete" if float(selected.scores["total"]) >= options.pass_threshold else "needs_review"
    weak_views = [
        item["view_id"]
        for item in expanded
        if float(item["scores"].get("total", 0.0)) < options.pass_threshold
    ]
    if weak_views:
        status = "needs_review"
        notes.append(f"Views below pass threshold: {weak_views}.")
    if len(expanded) > 1:
        notes.append(f"MeshLock multiview consistency={multiview_scores['total']}.")
        if multiview_scores["total"] < 0.55:
            status = "needs_review"
            notes.append("Expanded views failed the multiview consistency threshold.")

    final_image = output_dir / "final.png"
    shutil.copy2(selected.output_file, final_image)
    final_view_images: dict[str, str] = {}
    for item in expanded:
        view_final = output_dir / f"final_{item['view_id']}.png"
        shutil.copy2(item["image"], view_final)
        final_view_images[item["view_id"]] = str(view_final)
    comparison = create_comparison_image(
        white_image=target_view["files"]["rgb"],
        final_image=final_image,
        output=output_dir / "white_vs_final.png",
    )
    contact = ""
    if len(expanded) > 1:
        contact = str(_make_contact_sheet(expanded, output_dir / "multiview_contact_sheet.png"))
        shutil.copy2(contact, output_dir / "three_view_contact.png")

    decision = AgentDecision(
        status=status,
        selected_trial=selected.trial_id,
        selected_score=selected.scores["total"],
        decision_notes=notes,
    )
    retry_decisions = [
        {
            "trial_id": trial.trial_id,
            "view_id": trial.view_id,
            "total": trial.scores.get("total", 0.0),
            "decision_reason": trial.decision_reason,
            "failure_reasons": trial.scores.get("failure_reasons", []),
        }
        for trial in trials
    ]
    report_path = output_dir / "agent_report.json"
    summary = AgentRunSummary(
        type="agent_run_summary",
        status=decision.status,
        output_dir=str(output_dir),
        prompt=options.prompt,
        source_model_path=source_model_path,
        render_manifest=str(options.input_renders),
        reference_policy="model_render_channels_only",
        target_view=target_view["view_id"],
        budget={
            "max_generations": max_generations,
            "generations_used": len(trials),
            "expand_views": options.expand_views,
            "planned_views": [view["view_id"] for view in view_group],
            "candidates_per_view": candidates_per_view,
            "pass_threshold": options.pass_threshold,
        },
        trials=[_serializable_trial(trial) for trial in trials],
        selected_trial=_serializable_trial(selected),
        expanded_views=[
            {key: value for key, value in item.items() if key != "features"} | {"features": item["features"]}
            for item in expanded
        ],
        final_image=str(final_image),
        comparison_image=str(comparison),
        three_view_contact=contact,
        agent_report=str(report_path),
        decision_notes=decision.decision_notes,
        elapsed_seconds=round(time.time() - started, 3),
        model_key=model_key,
        backend=backend_name,
        score_version=score_version,
        structure_scores=structure_scores,
        multiview_scores=multiview_scores,
        retry_decisions=retry_decisions,
        final_view_images=final_view_images,
        multiview_contact_sheet=contact,
    )
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(summary), fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    emit(f"Agent render complete: {status}", 1.0)
    return asdict(summary)


def summarize_agent_report(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    selected = report.get("selected_trial", {})
    return (
        f"Agent status={report.get('status')} "
        f"selected={selected.get('trial_id')} "
        f"score={selected.get('scores', {}).get('total')} "
        f"final={report.get('final_image')}"
    )
