from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .manifest import write_manifest


def _shade_color(base: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * factor))) for channel in base)


def _make_view(view_index: int, total_views: int, size: int) -> dict[str, Image.Image]:
    angle = (math.tau * view_index) / max(total_views, 1)
    cx = size // 2 + int(math.sin(angle) * size * 0.06)
    cy = size // 2 - int(math.cos(angle) * size * 0.025)
    radius = int(size * 0.25)
    canvas = Image.new("RGB", (size, size), (242, 244, 246))
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.ellipse(
        [cx - radius, cy + radius // 2, cx + radius, cy + radius],
        fill=(0, 0, 0, 52),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(2, size // 42)))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow).convert("RGB")

    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    body = [
        cx - radius,
        cy - int(radius * 0.92),
        cx + radius,
        cy + int(radius * 0.78),
    ]
    mask_draw.ellipse(body, fill=255)
    ear_offset = int(math.sin(angle) * radius * 0.2)
    mask_draw.polygon(
        [
            (cx - radius // 2 + ear_offset, cy - radius // 2),
            (cx - radius // 5 + ear_offset, cy - int(radius * 1.45)),
            (cx, cy - radius // 2),
        ],
        fill=255,
    )
    mask_draw.polygon(
        [
            (cx + radius // 2 + ear_offset, cy - radius // 2),
            (cx + radius // 5 + ear_offset, cy - int(radius * 1.45)),
            (cx, cy - radius // 2),
        ],
        fill=255,
    )

    object_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(object_layer)
    base = (216, 84, 50)
    highlight = _shade_color(base, 1.22)
    shade = _shade_color(base, 0.62)
    draw.ellipse(body, fill=base + (255,), outline=shade + (255,), width=max(2, size // 110))
    draw.polygon(
        [
            (cx - radius // 2 + ear_offset, cy - radius // 2),
            (cx - radius // 5 + ear_offset, cy - int(radius * 1.45)),
            (cx, cy - radius // 2),
        ],
        fill=highlight + (255,),
        outline=shade + (255,),
    )
    draw.polygon(
        [
            (cx + radius // 2 + ear_offset, cy - radius // 2),
            (cx + radius // 5 + ear_offset, cy - int(radius * 1.45)),
            (cx, cy - radius // 2),
        ],
        fill=_shade_color(base, 0.86) + (255,),
        outline=shade + (255,),
    )
    draw.ellipse(
        [cx - radius // 3, cy - radius // 4, cx - radius // 6, cy - radius // 12],
        fill=(25, 32, 38, 255),
    )
    draw.ellipse(
        [cx + radius // 6, cy - radius // 4, cx + radius // 3, cy - radius // 12],
        fill=(25, 32, 38, 255),
    )
    draw.polygon(
        [(cx - radius // 8, cy + radius // 16), (cx + radius // 8, cy + radius // 16), (cx, cy + radius // 5)],
        fill=(250, 236, 210, 255),
    )
    draw.ellipse(
        [cx - radius // 3, cy - radius // 2, cx + radius // 3, cy + radius // 3],
        outline=(255, 224, 166, 180),
        width=max(2, size // 80),
    )
    rgb = Image.alpha_composite(canvas.convert("RGBA"), object_layer).convert("RGB")

    mask_np = np.asarray(mask, dtype=np.float32) / 255.0
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt(((xx - cx) / max(radius, 1)) ** 2 + ((yy - cy) / max(radius, 1)) ** 2)
    depth_arr = np.clip((1.15 - dist) * 210, 0, 255) * mask_np
    depth = Image.fromarray(depth_arr.astype(np.uint8), "L")

    edge = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(max(3, size // 180 * 2 + 1)))
    normal_arr = np.zeros((size, size, 3), dtype=np.uint8)
    nx = np.clip(((xx - cx) / max(radius, 1)) * 90 + 128, 0, 255)
    ny = np.clip(((yy - cy) / max(radius, 1)) * 90 + 128, 0, 255)
    nz = np.clip(255 - dist * 80, 96, 255)
    normal_arr[..., 0] = (nx * mask_np + 128 * (1 - mask_np)).astype(np.uint8)
    normal_arr[..., 1] = (ny * mask_np + 128 * (1 - mask_np)).astype(np.uint8)
    normal_arr[..., 2] = (nz * mask_np + 255 * (1 - mask_np)).astype(np.uint8)
    normal = Image.fromarray(normal_arr, "RGB")
    return {"rgb": rgb, "depth": depth.convert("RGB"), "edge": edge.convert("RGB"), "normal": normal, "mask": mask.convert("RGB")}


def create_sample_renders(output_dir: str | Path, *, views: int = 4, resolution: int = 512) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"type": "render_manifest", "source": "synthetic_sample", "views": []}
    for index in range(views):
        view_id = f"view_{index:02d}"
        view_dir = output / view_id
        view_dir.mkdir(parents=True, exist_ok=True)
        channels = _make_view(index, views, resolution)
        files: dict[str, str] = {}
        for channel, image in channels.items():
            path = view_dir / f"{channel}.png"
            image.save(path)
            files[channel] = str(path)
        manifest["views"].append({"view_id": view_id, "azimuth_deg": round(index * 360 / views, 3), "files": files})
    return write_manifest(output / "manifest.json", manifest)
