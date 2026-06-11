from __future__ import annotations

from pathlib import Path

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
