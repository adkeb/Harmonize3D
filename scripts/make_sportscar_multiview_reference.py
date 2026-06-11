#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create consistent transparent multiview sports-car reference images.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument(
        "--line-width",
        type=int,
        default=7,
        help="Structural outline width at final resolution.",
    )
    return parser.parse_args()


def wheel(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float) -> None:
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(18, 19, 22, 255), outline=(214, 224, 230, 255), width=5)
    draw.ellipse((cx - r * 0.58, cy - r * 0.58, cx + r * 0.58, cy + r * 0.58), fill=(58, 63, 70, 255))
    for i in range(8):
        angle = i * 3.14159 / 4
        x = cx + r * 0.52 * __import__("math").cos(angle)
        y = cy + r * 0.52 * __import__("math").sin(angle)
        draw.line((cx, cy, x, y), fill=(220, 230, 235, 255), width=3)


def side_view(size: int, line_width: int, flip: bool = False) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    body = [
        (size * 0.08, size * 0.66),
        (size * 0.16, size * 0.59),
        (size * 0.33, size * 0.53),
        (size * 0.48, size * 0.42),
        (size * 0.63, size * 0.43),
        (size * 0.78, size * 0.52),
        (size * 0.92, size * 0.62),
        (size * 0.96, size * 0.69),
        (size * 0.84, size * 0.73),
        (size * 0.19, size * 0.73),
    ]
    canopy = [
        (size * 0.40, size * 0.52),
        (size * 0.52, size * 0.35),
        (size * 0.67, size * 0.42),
        (size * 0.75, size * 0.55),
        (size * 0.57, size * 0.56),
    ]
    if flip:
        body = [(size - x, y) for x, y in body]
        canopy = [(size - x, y) for x, y in canopy]
    d.polygon(body, fill=(235, 239, 241, 255), outline=(34, 40, 46, 255))
    d.line(body + [body[0]], fill=(34, 40, 46, 255), width=line_width, joint="curve")
    d.polygon(canopy, fill=(36, 48, 58, 255), outline=(190, 205, 214, 255))
    d.line(canopy + [canopy[0]], fill=(24, 30, 36, 255), width=max(3, line_width - 2))
    wing = [(size * 0.72, size * 0.47), (size * 0.91, size * 0.39), (size * 0.94, size * 0.45), (size * 0.78, size * 0.52)]
    if flip:
        wing = [(size - x, y) for x, y in wing]
    d.polygon(wing, fill=(210, 218, 224, 255), outline=(32, 36, 42, 255))
    d.line(
        (size * (0.21 if not flip else 0.79), size * 0.64, size * (0.80 if not flip else 0.20), size * 0.58),
        fill=(82, 95, 106, 255),
        width=max(3, line_width - 2),
    )
    intake = [
        (size * 0.38, size * 0.63),
        (size * 0.52, size * 0.58),
        (size * 0.67, size * 0.61),
        (size * 0.58, size * 0.68),
        (size * 0.41, size * 0.69),
    ]
    if flip:
        intake = [(size - x, y) for x, y in intake]
    d.polygon(intake, fill=(30, 36, 42, 255), outline=(170, 184, 194, 255))
    wheel(d, size * (0.29 if not flip else 0.71), size * 0.72, size * 0.075)
    wheel(d, size * (0.76 if not flip else 0.24), size * 0.72, size * 0.082)
    d.polygon([(size * 0.13, size * 0.62), (size * 0.27, size * 0.58), (size * 0.25, size * 0.65)], fill=(0, 212, 245, 255))
    d.rounded_rectangle((size * 0.67, size * 0.61, size * 0.90, size * 0.66), radius=10, fill=(226, 34, 72, 255))
    return img


def front_or_back(size: int, line_width: int, back: bool = False) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    body = [
        (size * 0.11, size * 0.67),
        (size * 0.24, size * 0.53),
        (size * 0.41, size * 0.46),
        (size * 0.59, size * 0.46),
        (size * 0.76, size * 0.53),
        (size * 0.89, size * 0.67),
        (size * 0.73, size * 0.75),
        (size * 0.27, size * 0.75),
    ]
    d.polygon(body, fill=(235, 239, 241, 255), outline=(34, 40, 46, 255))
    d.line(body + [body[0]], fill=(34, 40, 46, 255), width=line_width)
    d.polygon([(size * 0.39, size * 0.47), (size * 0.47, size * 0.36), (size * 0.53, size * 0.36), (size * 0.61, size * 0.47)], fill=(35, 48, 58, 255), outline=(180, 195, 205, 255))
    d.line((size * 0.5, size * 0.38, size * 0.5, size * 0.74), fill=(110, 120, 128, 255), width=max(3, line_width - 4))
    if back:
        d.rounded_rectangle((size * 0.24, size * 0.59, size * 0.43, size * 0.63), radius=10, fill=(226, 34, 72, 255))
        d.rounded_rectangle((size * 0.57, size * 0.59, size * 0.76, size * 0.63), radius=10, fill=(226, 34, 72, 255))
        d.rectangle((size * 0.34, size * 0.70, size * 0.66, size * 0.76), fill=(25, 28, 32, 255))
        d.rectangle((size * 0.19, size * 0.44, size * 0.81, size * 0.50), fill=(210, 218, 224, 255), outline=(34, 40, 46, 255))
    else:
        d.polygon([(size * 0.22, size * 0.59), (size * 0.42, size * 0.55), (size * 0.38, size * 0.62)], fill=(0, 212, 245, 255))
        d.polygon([(size * 0.78, size * 0.59), (size * 0.58, size * 0.55), (size * 0.62, size * 0.62)], fill=(0, 212, 245, 255))
        d.polygon([(size * 0.41, size * 0.68), (size * 0.59, size * 0.68), (size * 0.55, size * 0.75), (size * 0.45, size * 0.75)], fill=(28, 32, 36, 255))
    wheel(d, size * 0.25, size * 0.73, size * 0.056)
    wheel(d, size * 0.75, size * 0.73, size * 0.056)
    return img


def make_sheet(paths: dict[str, Path], output: Path, size: int) -> None:
    tile = size // 2
    canvas = Image.new("RGBA", (tile * 2, tile * 2), (245, 247, 249, 255))
    for idx, key in enumerate(("front", "left", "back", "right")):
        image = Image.open(paths[key]).convert("RGBA").resize((tile, tile), Image.Resampling.LANCZOS)
        canvas.alpha_composite(image, ((idx % 2) * tile, (idx // 2) * tile))
    canvas.convert("RGB").save(output)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scale = 3
    size = args.size * scale
    line_width = args.line_width * scale
    views = {
        "front": front_or_back(size, line_width, back=False),
        "left": side_view(size, line_width, flip=False),
        "back": front_or_back(size, line_width, back=True),
        "right": side_view(size, line_width, flip=True),
    }
    paths: dict[str, Path] = {}
    for name, image in views.items():
        image = image.resize((args.size, args.size), Image.Resampling.LANCZOS)
        path = args.output_dir / f"{name}.png"
        image.save(path)
        paths[name] = path
    make_sheet(paths, args.output_dir / "reference_sheet.png", args.size)
    print(args.output_dir)


if __name__ == "__main__":
    main()
