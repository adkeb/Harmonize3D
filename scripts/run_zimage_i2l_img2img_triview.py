#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


DEFAULT_VIEWS = [
    ("left", "view_left_30"),
    ("locked", "view_locked"),
    ("right", "view_right_30"),
]

DEFAULT_PROMPT = (
    "glossy red racecar matching the exact 3D model silhouette, isolated on clean white studio background, "
    "black glass canopy, black tires, carbon fiber aero splitter, premium hard-surface product render"
)

DEFAULT_NEGATIVE_PROMPT = (
    "red background, colored background, gray clay, white plastic, pink plastic, line art, wireframe, "
    "copied control image, blurry, low quality, warped wheels, extra parts, text, watermark"
)


def _workspace_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


def _require(path: str | Path) -> Path:
    p = _workspace_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _preprocess_white_input(
    white_renders: Path,
    view_dir: str,
    output: Path,
    threshold: int,
    blur_radius: float,
    background: tuple[int, int, int],
) -> Path:
    rgb = Image.open(_require(white_renders / view_dir / "rgb.png")).convert("RGB")
    mask = Image.open(_require(white_renders / view_dir / "mask.png")).convert("L")
    alpha = mask.point(lambda value: 255 if value > threshold else 0)
    if blur_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(blur_radius))
    canvas = Image.new("RGB", rgb.size, background)
    image = Image.composite(rgb, canvas, alpha)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    alpha.save(output.with_name(output.stem + "_alpha.png"))
    return output


def _run_probe(args: argparse.Namespace, label: str, view_dir: str, input_image: Path) -> Path:
    view_output = _workspace_path(args.output) / f"probe_{label}"
    sample = view_output / f"sample_00_seed_{args.seed}.png"
    if sample.exists() and not args.force:
        return sample

    cmd = [
        sys.executable,
        "scripts/probe_zimage_i2l_lora.py",
        "--output",
        str(view_output),
        "--lora-output",
        str(_workspace_path(args.lora_output)),
        "--lora-scale",
        str(args.lora_scale),
        "--generate",
        "--samples",
        "1",
        "--steps",
        str(args.steps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--seed",
        str(args.seed),
        "--input-image",
        str(input_image),
        "--denoising-strength",
        str(args.denoising_strength),
        "--offload-device",
        args.offload_device,
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
    ]
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if args.control_image_mode:
        control_image = _require(_workspace_path(args.white_renders) / view_dir / args.control_image_name)
        cmd.extend(
            [
                "--controlnet-file",
                str(_workspace_path(args.controlnet_file)),
                "--control-image",
                str(control_image),
                "--control-mode",
                args.control_image_mode,
                "--control-scale",
                str(args.control_scale),
                "--control-start",
                str(args.control_start),
                "--control-end",
                str(args.control_end),
                "--patch-lite-controlnet",
            ]
        )

    view_output.mkdir(parents=True, exist_ok=True)
    log_path = view_output / "batch_run.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT, check=True)
    if not sample.exists():
        raise FileNotFoundError(str(sample))
    return sample


def _make_contact(items: list[tuple[str, Path]], output: Path, tile: int) -> Path:
    pad = 16
    label_h = 28
    canvas = Image.new("RGB", (pad + len(items) * (tile + pad), pad * 2 + label_h + tile), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        x = pad + index * (tile + pad)
        y = pad
        draw.text((x, y), label, fill=(20, 20, 20))
        canvas.paste(image, (x, y + label_h + (tile - image.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def run(args: argparse.Namespace) -> dict[str, object]:
    output = _workspace_path(args.output)
    preprocess_dir = output / "inputs"
    white_renders = _workspace_path(args.white_renders)

    selected = set(args.views or [label for label, _ in DEFAULT_VIEWS])
    rendered: list[tuple[str, Path]] = []
    inputs: list[dict[str, str]] = []
    for label, view_dir in DEFAULT_VIEWS:
        if label not in selected:
            continue
        input_path = _preprocess_white_input(
            white_renders,
            view_dir,
            preprocess_dir / f"{label}_white_bg.png",
            threshold=args.mask_threshold,
            blur_radius=args.mask_blur,
            background=tuple(args.background),
        )
        sample = _run_probe(args, label, view_dir, input_path)
        final_path = output / f"{label}.png"
        shutil.copyfile(sample, final_path)
        rendered.append((label, final_path))
        inputs.append({"label": label, "view": view_dir, "input": str(input_path), "sample": str(final_path)})

    if not rendered:
        raise ValueError("No views selected")
    contact = _make_contact(rendered, output / "triview_contact.png", tile=args.contact_tile)
    report = {
        "type": "zimage_i2l_img2img_triview",
        "output": str(output),
        "contact_sheet": str(contact),
        "views": inputs,
        "selected_views": sorted(selected),
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "denoising_strength": args.denoising_strength,
        "lora_scale": args.lora_scale,
        "mask_threshold": args.mask_threshold,
        "mask_blur": args.mask_blur,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "control_image_mode": args.control_image_mode,
        "control_scale": args.control_scale,
        "control_start": args.control_start,
        "control_end": args.control_end,
    }
    report_path = output / "triview_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the current best Z-Image-i2L img2img tri-view render recipe.")
    parser.add_argument("--output", default="outputs/flux2_klein_high_quality_car_reference/zimage_i2l_img2img_whitebg_batch_512")
    parser.add_argument("--white-renders", default="outputs/flux2_klein_high_quality_car_reference/white_renders")
    parser.add_argument("--views", nargs="+", choices=[label for label, _ in DEFAULT_VIEWS], default=None)
    parser.add_argument("--lora-output", default="outputs/flux2_klein_high_quality_car_reference/zimage_i2l_anchor_probe/zimage_i2l_style_lora.safetensors")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--denoising-strength", type=float, default=0.68)
    parser.add_argument("--lora-scale", type=float, default=0.85)
    parser.add_argument("--mask-threshold", type=int, default=170)
    parser.add_argument("--mask-blur", type=float, default=1.2)
    parser.add_argument("--background", type=int, nargs=3, default=(248, 248, 246))
    parser.add_argument("--contact-tile", type=int, default=360)
    parser.add_argument("--offload-device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--control-image-mode", default="")
    parser.add_argument("--control-image-name", default="depth.png")
    parser.add_argument("--controlnet-file", default="models/Z-Image-Fun-Controlnet-Union-2.1/Z-Image-Fun-Controlnet-Union-2.1-lite.safetensors")
    parser.add_argument("--control-scale", type=float, default=0.25)
    parser.add_argument("--control-start", type=float, default=0.0)
    parser.add_argument("--control-end", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print("Wrote report:", report["report"])
    print("Contact:", report["contact_sheet"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
