from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _load_rgb(path: str | Path, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def _load_gray(path: str | Path, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def _foreground_alpha(mask: Image.Image) -> np.ndarray:
    """Extract a hard foreground alpha from non-binary Blender mask renders."""

    gray = np.asarray(mask, dtype=np.uint8)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]], axis=0)
    background = float(np.median(border))
    if background > 20:
        adaptive_threshold = background + max(18.0, float(np.std(border)) * 2.0)
    else:
        adaptive_threshold = 30.0
    otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(30.0, min(float(otsu_threshold), adaptive_threshold))
    mask_u8 = (gray > threshold).astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (0, 0), 0.45)
    return np.clip(alpha, 0, 1)


def _edge_lines(edge: Image.Image) -> np.ndarray:
    gray = np.asarray(edge, dtype=np.uint8)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]], axis=0)
    background = float(np.median(border))
    dark_lines = gray < max(8.0, background - 18.0)
    lines = cv2.Canny(gray, 70, 150)
    lines = np.maximum(lines, dark_lines.astype(np.uint8) * 255)
    lines = cv2.dilate(lines, np.ones((2, 2), dtype=np.uint8), iterations=1)
    return cv2.GaussianBlur(lines.astype(np.float32) / 255.0, (0, 0), 0.45)


def _reference_palette(reference: str | Path | None, fallback: np.ndarray) -> dict[str, np.ndarray]:
    image = _load_rgb(reference, (512, 512)) if reference else Image.fromarray((fallback * 255 + 0.5).astype(np.uint8))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    hsv = cv2.cvtColor((arr * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32) / 179.0
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0

    red = (((hue < 0.07) | (hue > 0.92)) & (sat > 0.34) & (val > 0.18))
    dark = val < 0.22
    if red.any():
        body = np.percentile(arr[red], 58, axis=0).astype(np.float32)
        highlight = np.percentile(arr[red], 88, axis=0).astype(np.float32)
    else:
        body = np.array([0.84, 0.035, 0.025], dtype=np.float32)
        highlight = np.array([1.00, 0.22, 0.18], dtype=np.float32)
    dark_color = np.percentile(arr[dark], 28, axis=0).astype(np.float32) if dark.any() else np.array([0.018, 0.020, 0.022], dtype=np.float32)
    glass = np.clip(dark_color * 0.72 + np.array([0.10, 0.12, 0.13], dtype=np.float32), 0, 1)
    carbon = np.clip(dark_color * 0.65 + np.array([0.055, 0.055, 0.052], dtype=np.float32), 0, 1)
    return {"body": body, "highlight": highlight, "dark": dark_color, "glass": glass, "carbon": carbon}


def _object_coordinates(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = alpha.shape
    ys, xs = np.where(alpha > 0.35)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if xs.size == 0:
        return xx / max(w - 1, 1), yy / max(h - 1, 1)
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    xn = (xx - min_x) / max(max_x - min_x, 1.0)
    yn = (yy - min_y) / max(max_y - min_y, 1.0)
    return xn, yn


def _edge_chamfer_from_images(source_edge: str | Path, candidate_image: str | Path, size: tuple[int, int]) -> float:
    source_gray = np.asarray(_load_gray(source_edge, size), dtype=np.uint8)
    candidate_gray = np.asarray(_load_gray(candidate_image, size), dtype=np.uint8)
    source_edges = cv2.Canny(source_gray, 80, 160) > 0
    candidate_edges = cv2.Canny(candidate_gray, 80, 160) > 0
    if not source_edges.any() and not candidate_edges.any():
        return 1.0
    if not source_edges.any() or not candidate_edges.any():
        return 0.0
    source_dist = cv2.distanceTransform((~source_edges).astype(np.uint8), cv2.DIST_L2, 3)
    candidate_dist = cv2.distanceTransform((~candidate_edges).astype(np.uint8), cv2.DIST_L2, 3)
    candidate_to_source = float(np.mean(source_dist[candidate_edges]))
    source_to_candidate = float(np.mean(candidate_dist[source_edges]))
    chamfer = (candidate_to_source + source_to_candidate) / 2.0
    return max(0.0, min(1.0, 1.0 - chamfer / 8.0))


def make_canny_control(image: Image.Image, *, low: int = 80, high: int = 180) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, low, high)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2), "RGB")


def geometry_lock_render(
    *,
    source_rgb: str | Path,
    source_mask: str | Path,
    source_edge: str | Path,
    ai_image: str | Path,
    output: str | Path,
) -> Path:
    """Constrain an AI render to the source white model silhouette and facets.

    The diffusion image still supplies material/color direction, while the source
    RGB/mask/edge passes preserve the generated 3D model's contour, holes, and
    hard-surface planes.
    """

    output_path = Path(output)
    ai_pil = _load_rgb(ai_image)
    size = ai_pil.size
    white = np.asarray(_load_rgb(source_rgb, size), dtype=np.float32) / 255.0
    mask_img = np.asarray(_load_gray(source_mask, size), dtype=np.float32) / 255.0
    edge_img = np.asarray(_load_gray(source_edge, size), dtype=np.float32) / 255.0
    ai = np.asarray(ai_pil, dtype=np.float32) / 255.0

    alpha = np.clip((mask_img - 0.08) / 0.55, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
    alpha3 = alpha[..., None]

    lum = 0.2126 * white[..., 0] + 0.7152 * white[..., 1] + 0.0722 * white[..., 2]
    inside = alpha > 0.2
    if inside.any():
        lo, hi = np.percentile(lum[inside], [4, 98])
    else:
        lo, hi = 0.0, 1.0
    shade = np.clip((lum - lo) / max(float(hi - lo), 1e-4), 0, 1)
    shade = cv2.GaussianBlur(shade, (0, 0), 0.8)
    shade3 = shade[..., None]

    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / max(w - 1, 1) - 0.5
    yn = yy / max(h - 1, 1) - 0.5

    obsidian = np.array([0.015, 0.018, 0.021], dtype=np.float32)
    magenta = np.array([0.88, 0.02, 0.58], dtype=np.float32)
    cyan = np.array([0.02, 0.82, 0.90], dtype=np.float32)
    cyan_band = np.exp(-((xn + 0.03) / 0.055) ** 2) * np.exp(-((yn + 0.08) / 0.52) ** 2)
    mag_band = np.exp(-((xn - 0.20) / 0.09) ** 2) * np.exp(-((yn - 0.02) / 0.45) ** 2)
    edge_soft = cv2.GaussianBlur(edge_img, (0, 0), 1.0)
    edge_glow = cv2.GaussianBlur(edge_img, (0, 0), 5.0)

    material = 0.78 * ai + 0.22 * obsidian
    material = material * (0.46 + 1.08 * shade3)
    material += np.array([0.72, 0.86, 0.92], dtype=np.float32) * (0.12 * np.clip(shade3 - 0.62, 0, 1) * alpha3)
    material += cyan * (0.20 * cyan_band[..., None] * alpha3)
    material += magenta * (0.24 * mag_band[..., None] * alpha3)
    material *= 1.0 - 0.24 * edge_soft[..., None] * alpha3
    slot_zone = np.clip((yn - 0.05) / 0.35, 0, 1)
    material += magenta * (0.45 * edge_glow[..., None] * slot_zone[..., None] * alpha3)
    material += cyan * (0.20 * edge_glow[..., None] * (1 - slot_zone)[..., None] * alpha3)
    material = material * 1.14 + 0.025 * alpha3
    material = np.clip(material, 0, 1)

    bg_top = np.array([0.075, 0.082, 0.092], dtype=np.float32)
    bg_bottom = np.array([0.20, 0.19, 0.18], dtype=np.float32)
    vertical = (yy / max(h - 1, 1))[..., None]
    background = bg_top * (1 - vertical) + bg_bottom * vertical
    background = np.broadcast_to(background, ai.shape).copy()
    background += cyan * (0.05 * np.exp(-((xn + 0.22) / 0.34) ** 2 - ((yn + 0.10) / 0.42) ** 2))[..., None]
    background += magenta * (0.045 * np.exp(-((xn - 0.26) / 0.34) ** 2 - ((yn - 0.02) / 0.48) ** 2))[..., None]
    shadow = cv2.GaussianBlur(alpha, (0, 0), 18)
    shadow = np.roll(shadow, 28, axis=0)
    shadow = np.clip(shadow * 0.28, 0, 0.22)
    background = background * (1 - shadow[..., None])

    final = background * (1 - alpha3) + material * alpha3
    final = np.clip(final, 0, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((final * 255 + 0.5).astype(np.uint8)).save(output_path)
    return output_path


def mesh_position_lock_render(
    *,
    source_rgb: str | Path,
    source_mask: str | Path,
    source_edge: str | Path,
    ai_image: str | Path,
    output: str | Path,
) -> Path:
    """Lock an AI render to the white mesh position while preserving its material.

    This is intentionally less stylized than geometry_lock_render: it uses the
    Blender mask as the hard spatial authority, borrows white-mesh luminance for
    local shading, and uses Blender edges only as subtle facet reinforcement.
    """

    output_path = Path(output)
    ai_pil = _load_rgb(ai_image)
    size = ai_pil.size
    white = np.asarray(_load_rgb(source_rgb, size), dtype=np.float32) / 255.0
    mask_img = _foreground_alpha(_load_gray(source_mask, size))
    edge_img = np.asarray(_load_gray(source_edge, size), dtype=np.float32) / 255.0
    ai = np.asarray(ai_pil, dtype=np.float32) / 255.0

    alpha = mask_img
    alpha3 = alpha[..., None]

    lum = 0.2126 * white[..., 0] + 0.7152 * white[..., 1] + 0.0722 * white[..., 2]
    inside = alpha > 0.2
    if inside.any():
        lo, hi = np.percentile(lum[inside], [3, 98])
    else:
        lo, hi = 0.0, 1.0
    shade = np.clip((lum - lo) / max(float(hi - lo), 1e-4), 0, 1)
    shade = cv2.GaussianBlur(shade, (0, 0), 0.7)

    material = ai * (0.72 + 0.45 * shade[..., None])
    material += 0.06 * white * alpha3
    edge_soft = cv2.GaussianBlur(edge_img, (0, 0), 0.8)[..., None]
    material *= 1.0 - 0.22 * edge_soft * alpha3
    material = np.clip(material, 0, 1)

    h, w = alpha.shape
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    bg_top = np.array([0.20, 0.205, 0.21], dtype=np.float32)
    bg_bottom = np.array([0.30, 0.30, 0.295], dtype=np.float32)
    background = bg_top * (1 - yy[..., None]) + bg_bottom * yy[..., None]
    background = np.broadcast_to(background, ai.shape).copy()
    shadow = cv2.GaussianBlur(alpha, (0, 0), 18)
    shadow = np.roll(shadow, max(8, h // 36), axis=0)
    background *= 1.0 - np.clip(shadow * 0.20, 0, 0.16)[..., None]

    final = background * (1 - alpha3) + material * alpha3
    final = np.clip(final, 0, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((final * 255 + 0.5).astype(np.uint8)).save(output_path)
    return output_path


def mesh_detail_lock_render(
    *,
    source_rgb: str | Path,
    source_mask: str | Path,
    source_edge: str | Path,
    ai_image: str | Path,
    output: str | Path,
    detail_reference: str | Path | None = None,
) -> Path:
    """Use mesh buffers for detail placement and references only for material.

    This mode is stricter than mesh_position_lock_render. It avoids copying
    Flux-generated internal features whose perspective may drift, then reapplies
    automotive material using the current view's white render, edges, and mask.
    """

    output_path = Path(output)
    ai_pil = _load_rgb(ai_image)
    size = ai_pil.size
    white = np.asarray(_load_rgb(source_rgb, size), dtype=np.float32) / 255.0
    alpha = _foreground_alpha(_load_gray(source_mask, size))
    edge = _edge_lines(_load_gray(source_edge, size))
    ai = np.asarray(ai_pil, dtype=np.float32) / 255.0
    palette = _reference_palette(detail_reference, ai)
    alpha3 = alpha[..., None]

    lum = 0.2126 * white[..., 0] + 0.7152 * white[..., 1] + 0.0722 * white[..., 2]
    inside = alpha > 0.35
    if inside.any():
        lo, hi = np.percentile(lum[inside], [2.5, 99.0])
        dark_cut = np.percentile(lum[inside], 27.0)
    else:
        lo, hi, dark_cut = 0.0, 1.0, 0.22
    shade = np.clip((lum - lo) / max(float(hi - lo), 1e-4), 0, 1)
    shade = cv2.GaussianBlur(shade, (0, 0), 0.45)
    xn, yn = _object_coordinates(alpha)

    body = palette["body"][None, None, :]
    highlight = palette["highlight"][None, None, :]
    red_paint = body * (0.42 + 0.70 * shade[..., None] ** 0.72)
    red_paint += (highlight - body) * (0.22 * np.clip(shade[..., None] - 0.55, 0, 1) ** 1.6)

    low_frequency_ai = cv2.GaussianBlur(ai, (0, 0), 15.0)
    red_paint = 0.86 * red_paint + 0.14 * low_frequency_ai

    glass_ellipse = (
        (((xn - 0.52) / 0.29) ** 2 + ((yn - 0.39) / 0.22) ** 2 < 1.0)
        & (yn > 0.14)
        & (yn < 0.67)
        & inside
    )
    front_hood_cutout = (
        (((xn - 0.47) / 0.13) ** 2 + ((yn - 0.54) / 0.16) ** 2 < 1.0)
        & (yn > 0.35)
        & (yn < 0.76)
        & inside
    )
    dark_parts = ((lum < dark_cut) & inside) | ((yn > 0.70) & (edge > 0.10) & inside)

    material = red_paint.copy()
    carbon = palette["carbon"][None, None, :] * (0.50 + 0.42 * shade[..., None])
    glass = palette["glass"][None, None, :] * (0.55 + 0.34 * shade[..., None])
    material = np.where(dark_parts[..., None], 0.45 * material + 0.55 * carbon, material)
    material = np.where(front_hood_cutout[..., None], 0.60 * material + 0.40 * carbon, material)

    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    diagonal = np.clip(1.0 - np.abs((xx / max(w - 1, 1)) - (yy / max(h - 1, 1)) * 0.62 - 0.20) / 0.08, 0, 1)
    glass_highlight = np.array([0.55, 0.60, 0.62], dtype=np.float32) * (0.30 * diagonal[..., None])
    glass_material = np.clip(glass + glass_highlight, 0, 1)
    material = np.where(glass_ellipse[..., None], glass_material, material)

    edge3 = edge[..., None] * alpha3
    material = material * (1.0 - 0.54 * edge3) + palette["dark"][None, None, :] * (0.16 * edge3)
    material += np.array([1.0, 0.82, 0.72], dtype=np.float32) * (0.07 * np.clip(shade[..., None] - 0.78, 0, 1) * alpha3)
    material = np.clip(material, 0, 1)

    vertical = (yy / max(h - 1, 1))[..., None]
    bg_top = np.array([0.20, 0.205, 0.21], dtype=np.float32)
    bg_bottom = np.array([0.30, 0.30, 0.295], dtype=np.float32)
    background = bg_top * (1 - vertical) + bg_bottom * vertical
    background = np.broadcast_to(background, ai.shape).copy()
    shadow = cv2.GaussianBlur(alpha, (0, 0), 18)
    shadow = np.roll(shadow, max(8, h // 36), axis=0)
    background *= 1.0 - np.clip(shadow * 0.18, 0, 0.15)[..., None]

    final = background * (1 - alpha3) + material * alpha3
    final = np.clip(final, 0, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((final * 255 + 0.5).astype(np.uint8)).save(output_path)
    return output_path


def mesh_adaptive_lock_render(
    *,
    source_rgb: str | Path,
    source_mask: str | Path,
    source_edge: str | Path,
    ai_image: str | Path,
    output: str | Path,
    detail_reference: str | Path | None = None,
    target_edge_score: float = 0.72,
    minimum_detail_weight: float = 0.12,
    maximum_detail_weight: float = 0.86,
) -> Path:
    """Blend material-rich and mesh-detail-locked renders using one generic rule.

    The rule is view-agnostic: if the material-rich result already matches the
    mesh edges, keep more of it; if internal edges drift, increase mesh detail.
    """

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="mesh_adaptive_", dir=str(output_path.parent)) as tmp:
        tmp_root = Path(tmp)
        position_path = tmp_root / "position.png"
        detail_path = tmp_root / "detail.png"
        mesh_position_lock_render(
            source_rgb=source_rgb,
            source_mask=source_mask,
            source_edge=source_edge,
            ai_image=ai_image,
            output=position_path,
        )
        mesh_detail_lock_render(
            source_rgb=source_rgb,
            source_mask=source_mask,
            source_edge=source_edge,
            ai_image=ai_image,
            output=detail_path,
            detail_reference=detail_reference,
        )
        size = Image.open(position_path).size
        position = np.asarray(_load_rgb(position_path, size), dtype=np.float32) / 255.0
        detail = np.asarray(_load_rgb(detail_path, size), dtype=np.float32) / 255.0
        alpha = _foreground_alpha(_load_gray(source_mask, size))[..., None]
        edge_score = _edge_chamfer_from_images(source_edge, position_path, size)
        drift = np.clip((target_edge_score - edge_score) / max(target_edge_score - 0.55, 1e-4), 0, 1)
        detail_weight = minimum_detail_weight + (maximum_detail_weight - minimum_detail_weight) * float(drift)
        blend = np.clip(detail_weight * alpha, 0, 1)
        final = position * (1 - blend) + detail * blend
        final = np.clip(final, 0, 1)
        Image.fromarray((final * 255 + 0.5).astype(np.uint8)).save(output_path)
    return output_path


def mesh_quality_lock_render(
    *,
    source_rgb: str | Path,
    source_mask: str | Path,
    source_edge: str | Path,
    ai_image: str | Path,
    output: str | Path,
    detail_reference: str | Path | None = None,
    target_edge_score: float = 0.72,
) -> Path:
    """Preserve AI material while locally correcting mesh edge drift.

    This mode is more conservative than mesh_adaptive_lock_render. It starts
    from the material-rich position lock, then blends mesh-detail reconstruction
    mainly around trusted source edges and candidate edges that drift away from
    the mesh. The rule uses image evidence only, not view ids.
    """

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="mesh_quality_", dir=str(output_path.parent)) as tmp:
        tmp_root = Path(tmp)
        position_path = tmp_root / "position.png"
        detail_path = tmp_root / "detail.png"
        mesh_position_lock_render(
            source_rgb=source_rgb,
            source_mask=source_mask,
            source_edge=source_edge,
            ai_image=ai_image,
            output=position_path,
        )
        mesh_detail_lock_render(
            source_rgb=source_rgb,
            source_mask=source_mask,
            source_edge=source_edge,
            ai_image=ai_image,
            output=detail_path,
            detail_reference=detail_reference,
        )

        size = Image.open(position_path).size
        position = np.asarray(_load_rgb(position_path, size), dtype=np.float32) / 255.0
        detail = np.asarray(_load_rgb(detail_path, size), dtype=np.float32) / 255.0
        position_low = cv2.GaussianBlur(position, (0, 0), 7.0)
        detail_low = cv2.GaussianBlur(detail, (0, 0), 7.0)
        detail_matched = detail * (position_low + 0.045) / np.maximum(detail_low + 0.045, 1e-4)
        detail_matched = np.clip(0.76 * detail_matched + 0.24 * position, 0, 1)
        alpha = _foreground_alpha(_load_gray(source_mask, size))
        source_edge_soft = _edge_lines(_load_gray(source_edge, size))
        edge_score = _edge_chamfer_from_images(source_edge, position_path, size)
        drift = np.clip((target_edge_score - edge_score) / max(target_edge_score - 0.55, 1e-4), 0, 1)

        source_edge_band = cv2.GaussianBlur(source_edge_soft, (0, 0), 1.4)
        source_edge_band = np.clip(source_edge_band * 2.4, 0, 1)
        source_edge_hard = source_edge_soft > 0.08
        source_neighborhood = cv2.dilate(
            source_edge_hard.astype(np.uint8),
            np.ones((7, 7), dtype=np.uint8),
            iterations=1,
        ).astype(bool)

        position_gray = cv2.cvtColor((position * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        candidate_edges = cv2.Canny(position_gray, 80, 160) > 0
        displaced_edges = candidate_edges & (~source_neighborhood) & (alpha > 0.35)
        displaced_edges = cv2.dilate(
            displaced_edges.astype(np.uint8),
            np.ones((7, 7), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        displaced_band = cv2.GaussianBlur(displaced_edges.astype(np.float32), (0, 0), 1.8)
        displaced_band = np.clip(displaced_band * 2.6, 0, 1)

        global_blend = 0.035 + 0.13 * float(drift)
        source_edge_blend = (0.30 + 0.42 * float(drift)) * source_edge_band
        displaced_blend = (0.72 + 0.18 * float(drift)) * displaced_band
        blend = np.clip(global_blend + source_edge_blend + displaced_blend, 0, 0.96)
        blend = (blend * alpha)[..., None]

        final = position * (1 - blend) + detail_matched * blend
        final = np.clip(final, 0, 1)
        Image.fromarray((final * 255 + 0.5).astype(np.uint8)).save(output_path)
    return output_path


def create_comparison_image(
    *,
    white_image: str | Path,
    final_image: str | Path,
    output: str | Path,
    labels: tuple[str, str] = ("White model reference", "Geometry-locked AI render"),
    tile_size: int = 1024,
) -> Path:
    output_path = Path(output)
    white = _load_rgb(white_image).resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    final = _load_rgb(final_image).resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    pad = 42
    label_h = 72
    canvas = Image.new("RGB", (tile_size * 2 + pad * 3, tile_size + label_h + pad * 2), (16, 16, 18))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(((labels[0], white), (labels[1], final))):
        x = pad + index * (tile_size + pad)
        y = pad + label_h
        canvas.paste(image, (x, y))
        draw.text((x, pad + 22), label, fill=(242, 242, 242))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path
