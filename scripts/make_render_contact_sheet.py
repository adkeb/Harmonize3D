#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a labeled contact sheet from rendered view images.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="view_*/rgb.png")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No images matched {args.input_dir / args.pattern}")

    label_h = max(36, args.tile // 14)
    tiles = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((args.tile, args.tile), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (args.tile, args.tile + label_h), (248, 248, 248))
        tile.paste(image, ((args.tile - image.width) // 2, 0))
        draw = ImageDraw.Draw(tile)
        name = f"{args.label} {path.parent.name}".strip()
        draw.text((16, args.tile + max(8, label_h // 4)), name, fill=(20, 20, 20))
        tiles.append(tile)

    rows = (len(tiles) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * args.tile, rows * (args.tile + label_h)), (232, 232, 232))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % args.cols) * args.tile, (i // args.cols) * (args.tile + label_h)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
