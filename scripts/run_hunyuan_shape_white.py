#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

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
    parser.add_argument("--preprocessed-image", type=Path, default=None)
    parser.add_argument("--no-remove-background", action="store_true")
    return parser.parse_args()


def _has_useful_alpha(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False
    alpha = image.getchannel("A")
    low, high = alpha.getextrema()
    return low < 245 and high > 10


def _crop_and_pad_to_square(image: Image.Image, *, padding_ratio: float = 0.08) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return rgba
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    pad = round(max(width, height) * padding_ratio)
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(rgba.width, bbox[2] + pad)
    bottom = min(rgba.height, bbox[3] + pad)
    cropped = rgba.crop((left, top, right, bottom))
    side = max(cropped.width, cropped.height)
    square = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    square.alpha_composite(cropped, ((side - cropped.width) // 2, (side - cropped.height) // 2))
    return square


def preprocess_reference_image(
    image_path: Path,
    output_path: Path,
    *,
    repo_root: Path,
    remove_background: bool = True,
) -> dict[str, object]:
    hy3dshape_root = repo_root / "hy3dshape"
    if str(hy3dshape_root) not in sys.path:
        sys.path.insert(0, str(hy3dshape_root))

    original = Image.open(image_path)
    image = original.convert("RGBA")
    used_background_remover = False
    if remove_background and not _has_useful_alpha(image):
        from hy3dshape.rembg import BackgroundRemover

        remover = BackgroundRemover()
        image = remover(original.convert("RGB")).convert("RGBA")
        used_background_remover = True

    preprocessed = _crop_and_pad_to_square(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessed.save(output_path)
    alpha = preprocessed.getchannel("A")
    alpha_bbox = alpha.getbbox()
    foreground_ratio = 0.0
    if alpha_bbox:
        histogram = alpha.histogram()
        foreground_pixels = sum(histogram[11:])
        foreground_ratio = foreground_pixels / max(1, preprocessed.width * preprocessed.height)
    return {
        "original_size": list(original.size),
        "preprocessed_size": list(preprocessed.size),
        "used_background_remover": used_background_remover,
        "foreground_ratio": round(float(foreground_ratio), 6),
        "alpha_bbox": list(alpha_bbox) if alpha_bbox else [],
    }


def mesh_sanity(mesh) -> dict[str, object]:
    extents = [float(value) for value in getattr(mesh, "extents", [])]
    max_extent = max(extents) if extents else 0.0
    min_extent = min((value for value in extents if value > 1e-6), default=0.0)
    bbox_aspect_ratio = max_extent / min_extent if min_extent else 0.0
    components = []
    try:
        components = list(mesh.split(only_watertight=False))
    except Exception:
        components = []
    face_count = int(len(mesh.faces))
    component_face_counts = [int(len(component.faces)) for component in components]
    largest_component_ratio = max(component_face_counts, default=face_count) / max(face_count, 1)
    flags = []
    if int(len(mesh.vertices)) < 20000 or face_count < 30000:
        flags.append("low_mesh_density")
    if bbox_aspect_ratio > 14:
        flags.append("extreme_bbox_aspect")
    if len(component_face_counts) > 30 and largest_component_ratio < 0.55:
        flags.append("fragmented_mesh")
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": face_count,
        "bounds": [[float(value) for value in row] for row in getattr(mesh, "bounds", [])],
        "extents": extents,
        "bbox_aspect_ratio": round(float(bbox_aspect_ratio), 6),
        "component_count": len(component_face_counts) or 1,
        "largest_component_face_ratio": round(float(largest_component_ratio), 6),
        "flags": flags,
        "status": "needs_review" if flags else "pass",
    }


def main():
    args = parse_args()
    hy3dshape_root = args.repo_root / "hy3dshape"
    sys.path.insert(0, str(hy3dshape_root))

    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)

    preprocessed_path = args.preprocessed_image or args.output.with_name("shape_input_preprocessed.png")
    preprocess_metadata = preprocess_reference_image(
        args.image,
        preprocessed_path,
        repo_root=args.repo_root,
        remove_background=not args.no_remove_background,
    )
    image = Image.open(preprocessed_path).convert("RGBA")
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
    sanity = mesh_sanity(mesh)
    metadata = {
        "source_image": str(args.image),
        "preprocessed_image": str(preprocessed_path),
        "preprocess": preprocess_metadata,
        "model_path": str(args.model_path),
        "output": str(args.output),
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "octree_resolution": args.octree_resolution,
        "num_chunks": args.num_chunks,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "vertices": sanity["vertices"],
        "faces": sanity["faces"],
        "mesh_sanity": sanity,
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
