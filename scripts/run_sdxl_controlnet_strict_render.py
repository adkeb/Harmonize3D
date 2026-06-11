#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Render a white model with SDXL img2img plus Canny and Depth ControlNet.")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--canny-controlnet", type=Path, required=True)
    parser.add_argument("--depth-controlnet", type=Path, required=True)
    parser.add_argument("--init-image", type=Path, required=True)
    parser.add_argument("--depth-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--strength", type=float, default=0.68)
    parser.add_argument("--canny-scale", type=float, default=1.25)
    parser.add_argument("--depth-scale", type=float, default=0.95)
    parser.add_argument("--canny-low", type=int, default=80)
    parser.add_argument("--canny-high", type=int, default=180)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)


def make_canny(image: Image.Image, low: int, high: int) -> Image.Image:
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, low, high)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2), "RGB")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    size = (args.width, args.height)

    init_image = load_rgb(args.init_image, size)
    depth_image = load_rgb(args.depth_image, size)
    canny_image = make_canny(init_image, args.canny_low, args.canny_high)
    canny_path = args.output_dir / "control_canny.png"
    depth_path = args.output_dir / "control_depth.png"
    canny_image.save(canny_path)
    depth_image.save(depth_path)

    from diffusers import ControlNetModel, EulerAncestralDiscreteScheduler
    if args.control_only:
        from diffusers import StableDiffusionXLControlNetPipeline as Pipeline
    else:
        from diffusers import StableDiffusionXLControlNetImg2ImgPipeline as Pipeline

    start = time.time()
    dtype = torch.float16
    controlnets = [
        ControlNetModel.from_pretrained(
            str(args.canny_controlnet),
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
        ),
        ControlNetModel.from_pretrained(
            str(args.depth_controlnet),
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
        ),
    ]
    pipe = Pipeline.from_pretrained(
        str(args.base_model),
        controlnet=controlnets,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )
    pipe.to(args.device)
    pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    outputs = []
    for index in range(args.candidates):
        seed = args.seed + index * 997
        generator = torch.Generator(device=args.device.split(":")[0]).manual_seed(seed)
        call_kwargs = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "controlnet_conditioning_scale": [args.canny_scale, args.depth_scale],
            "control_guidance_start": [0.0, 0.0],
            "control_guidance_end": [0.95, 0.9],
            "generator": generator,
            "width": args.width,
            "height": args.height,
        }
        if args.control_only:
            call_kwargs["image"] = [canny_image, depth_image]
        else:
            call_kwargs["image"] = init_image
            call_kwargs["control_image"] = [canny_image, depth_image]
            call_kwargs["strength"] = args.strength
        image = pipe(
            **call_kwargs
        ).images[0]
        path = args.output_dir / f"candidate_{index:02d}.png"
        image.save(path)
        outputs.append({"index": index, "seed": seed, "file": str(path)})

    metadata = {
        "base_model": str(args.base_model),
        "canny_controlnet": str(args.canny_controlnet),
        "depth_controlnet": str(args.depth_controlnet),
        "init_image": str(args.init_image),
        "depth_image": str(args.depth_image),
        "control_canny": str(canny_path),
        "control_depth": str(depth_path),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "canny_scale": args.canny_scale,
        "depth_scale": args.depth_scale,
        "control_only": args.control_only,
        "outputs": outputs,
        "elapsed_seconds": time.time() - start,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
