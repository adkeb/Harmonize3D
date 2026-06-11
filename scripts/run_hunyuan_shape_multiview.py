#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a white mesh with Hunyuan3D 2mv from 1-4 reference views.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Root of Tencent Hunyuan3D code. Use Hunyuan3D-2 for 2mv weights, Hunyuan3D-2.1 for 2.1 weights.",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-mv-fast")
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--back", type=Path)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--octree-resolution", type=int, default=512)
    parser.add_argument("--num-chunks", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--use-safetensors", action="store_true")
    return parser.parse_args()


def load_rgba(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    return Image.open(path).convert("RGBA")


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo_root))
    sys.path.insert(0, str(args.repo_root / "hy3dshape"))

    pipeline_family = "hy3dshape"
    try:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline, export_to_trimesh
    except ModuleNotFoundError:
        pipeline_family = "hy3dgen"
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.shapegen.pipelines import export_to_trimesh

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    images = {
        "front": load_rgba(args.front),
        "back": load_rgba(args.back),
        "left": load_rgba(args.left),
        "right": load_rgba(args.right),
    }
    images = {key: value for key, value in images.items() if value is not None}
    if not images:
        raise RuntimeError("At least one reference view is required.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    print("Loading Hunyuan3D 2mv shape model...", flush=True)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(args.model_path),
        subfolder=args.subfolder,
        variant="fp16",
        use_safetensors=args.use_safetensors,
        device=args.device,
        dtype=dtype,
    )

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    print(f"Generating multiview shape mesh with {len(images)} views using {pipeline_family}...", flush=True)
    outputs = pipeline(
        image=images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        output_type="trimesh" if pipeline_family == "hy3dgen" else "mesh",
    )
    mesh = outputs[0] if pipeline_family == "hy3dgen" else export_to_trimesh(outputs)[0]
    mesh.export(args.output)

    metadata = {
        "views": {key: str(path) for key, path in {
            "front": args.front,
            "back": args.back,
            "left": args.left,
            "right": args.right,
        }.items() if path is not None},
        "model_path": str(args.model_path),
        "subfolder": args.subfolder,
        "output": str(args.output),
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "octree_resolution": args.octree_resolution,
        "num_chunks": args.num_chunks,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "use_safetensors": args.use_safetensors,
        "pipeline_family": pipeline_family,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "elapsed_seconds": time.time() - start,
    }
    if torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(0)
        metadata["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    if args.metadata:
        args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
