#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dark automotive material init manifest from white clay renders."
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--body-color", default="18,20,23")
    parser.add_argument("--highlight-color", default="185,194,200")
    parser.add_argument("--background", default="46,46,46")
    parser.add_argument("--accent", default="cyan-magenta")
    return parser.parse_args()


def _rgb_triplet(value: str) -> np.ndarray:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError(f"Expected R,G,B values in 0..255, got {value!r}")
    return np.array(parts, dtype=np.float32)


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _soft_mask(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L").resize(size, Image.Resampling.LANCZOS)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.2))
    return np.asarray(mask, dtype=np.float32) / 255.0


def _normalize_car_luma(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    car_pixels = luma[mask > 0.08]
    if car_pixels.size == 0:
        return np.zeros_like(luma, dtype=np.float32)
    lo, hi = np.percentile(car_pixels, [7.0, 99.6])
    norm = (luma - lo) / max(hi - lo, 1.0)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def _background(width: int, height: int, base: np.ndarray) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    vignette = np.clip(1.0 - 0.25 * np.abs(x) - 0.12 * y, 0.62, 1.0)
    bg = base[None, None, :] * vignette[..., None]
    return np.clip(bg, 0, 255)


def _accent_lights(width: int, height: int, view_index: int, mask: np.ndarray, mode: str) -> np.ndarray:
    accents = np.zeros((height, width, 3), dtype=np.float32)
    if mode == "none":
        return accents

    ys, xs = np.where(mask > 0.25)
    if xs.size == 0:
        return accents
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    car_w = max(max_x - min_x, 1)
    car_h = max(max_y - min_y, 1)

    # The accents are tiny and view-dependent; they guide the image model without changing geometry.
    anchors = {
        0: [(0.13, 0.62, (55, 235, 255)), (0.88, 0.59, (255, 55, 185))],
        1: [(0.16, 0.60, (55, 235, 255)), (0.74, 0.64, (255, 55, 185))],
        2: [(0.35, 0.62, (55, 235, 255)), (0.65, 0.62, (255, 55, 185))],
        3: [(0.18, 0.62, (55, 235, 255)), (0.84, 0.62, (255, 55, 185))],
        4: [(0.12, 0.60, (55, 235, 255)), (0.80, 0.64, (255, 55, 185))],
        5: [(0.18, 0.65, (55, 235, 255)), (0.72, 0.62, (255, 55, 185))],
        6: [(0.34, 0.64, (55, 235, 255)), (0.66, 0.64, (255, 55, 185))],
        7: [(0.17, 0.62, (55, 235, 255)), (0.82, 0.62, (255, 55, 185))],
    }
    yy, xx = np.mgrid[0:height, 0:width]
    for ax, ay, color in anchors.get(view_index % 8, []):
        cx = min_x + ax * car_w
        cy = min_y + ay * car_h
        sx = max(car_w * 0.035, 12.0)
        sy = max(car_h * 0.018, 5.0)
        glow = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        accents += glow[..., None] * np.array(color, dtype=np.float32)[None, None, :] * 0.75
    return accents * mask[..., None]


def _make_init(
    rgb_path: str | Path,
    mask_path: str | Path,
    output_path: Path,
    *,
    body: np.ndarray,
    highlight: np.ndarray,
    bg_base: np.ndarray,
    accent: str,
    view_index: int,
) -> None:
    source = Image.open(rgb_path).convert("RGB")
    width, height = source.size
    rgb = np.asarray(source, dtype=np.float32)
    mask = _soft_mask(mask_path, (width, height))
    shade = _normalize_car_luma(rgb, mask)

    body_ramp = body[None, None, :] + (highlight - body)[None, None, :] * (shade[..., None] ** 2.15)
    darker_creases = np.clip((shade - 0.18) / 0.82, 0.0, 1.0)
    body_ramp *= (0.62 + 0.38 * darker_creases[..., None])

    # Keep wheel and intake cavities convincingly dark; these regions are already low-luma in the clay render.
    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    car_luma = luma[mask > 0.15]
    dark_threshold = np.percentile(car_luma, 32.0) if car_luma.size else 128.0
    dark_parts = ((luma < dark_threshold) & (mask > 0.18)).astype(np.float32)
    body_ramp = body_ramp * (1.0 - dark_parts[..., None] * 0.42)

    accents = _accent_lights(width, height, view_index, mask, accent)
    bg = _background(width, height, bg_base)
    car = np.clip(body_ramp + accents, 0, 255)
    out = bg * (1.0 - mask[..., None]) + car * mask[..., None]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB").save(output_path)


def main() -> None:
    args = parse_args()
    body = _rgb_triplet(args.body_color)
    highlight = _rgb_triplet(args.highlight_color)
    bg_base = _rgb_triplet(args.background)
    manifest = _load_manifest(args.input_manifest)

    out_manifest: dict[str, Any] = {
        "type": "render_manifest",
        "source_manifest": str(args.input_manifest),
        "material_init": {
            "body_color": args.body_color,
            "highlight_color": args.highlight_color,
            "background": args.background,
            "accent": args.accent,
        },
        "views": [],
    }

    for index, view in enumerate(manifest.get("views", [])):
        files = dict(view.get("files", {}))
        view_id = view.get("view_id", f"view_{index:02d}")
        rgb_out = args.output_dir / view_id / "rgb.png"
        _make_init(
            files["rgb"],
            files["mask"],
            rgb_out,
            body=body,
            highlight=highlight,
            bg_base=bg_base,
            accent=args.accent,
            view_index=index,
        )
        files["rgb"] = str(rgb_out)
        out_view = dict(view)
        out_view["files"] = files
        out_manifest["views"].append(out_view)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
