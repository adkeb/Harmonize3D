#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a white mesh with Hunyuan3D 2.1 shape pipeline only.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


def main():
    args = parse_args()
    hy3dshape_root = args.repo_root / "hy3dshape"
    sys.path.insert(0, str(hy3dshape_root))

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGBA")
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    start = time.time()
    print("Loading Hunyuan3D shape model...", flush=True)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(args.model_path),
        subfolder="hunyuan3d-dit-v2-1",
        variant="fp16",
        use_safetensors=False,
        device=args.device,
        dtype=dtype,
    )

    print("Generating shape mesh...", flush=True)
    mesh = pipeline(
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        mc_algo="mc",
        output_type="trimesh",
        enable_pbar=True,
    )[0]

    mesh.export(args.output)
    elapsed = time.time() - start
    metadata = {
        "source_image": str(args.image),
        "model_path": str(args.model_path),
        "output": str(args.output),
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "octree_resolution": args.octree_resolution,
        "num_chunks": args.num_chunks,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "elapsed_seconds": elapsed,
    }
    if torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(0)
        metadata["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    if args.metadata:
        args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
