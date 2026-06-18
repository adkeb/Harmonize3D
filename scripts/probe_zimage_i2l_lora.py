#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


DEFAULT_STYLE_IMAGES = [
    "outputs/flux2_klein_high_quality_car_reference/racecar_reference_image2.png",
    "outputs/flux2_klein_high_quality_car_reference/flux2_feature_consistency_position_lock_hardmask/feature_reference_anchor.png",
    "outputs/flux2_klein_high_quality_car_reference/flux2_feature_consistency_position_lock_hardmask/final_view_locked.png",
    "outputs/flux2_klein_high_quality_car_reference/flux2_feature_consistency_position_lock_hardmask/final_view_left_30.png",
    "outputs/flux2_klein_high_quality_car_reference/flux2_feature_consistency_position_lock_hardmask/final_view_right_30.png",
]


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


def _glob_required(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        paths = sorted(str(p) for p in Path.cwd().glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths


def _torch_dtype(name: str) -> Any:
    import torch

    lookup = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return lookup.get(name.lower(), torch.bfloat16)


def _vram_config(dtype: Any, device: str, offload_device: str) -> dict[str, Any]:
    return {
        "offload_dtype": dtype,
        "offload_device": offload_device,
        "onload_dtype": dtype,
        "onload_device": device,
        "preparing_dtype": dtype,
        "preparing_device": device,
        "computation_dtype": dtype,
        "computation_device": device,
    }


def _prepare_control_image(path: Path, size: tuple[int, int], mode: str) -> Image.Image:
    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    if mode in {"depth-neutral", "depth_neutral"}:
        depth = ImageOps.autocontrast(image.convert("L"))
        mask = depth.point(lambda value: 255 if value > 8 else 0)
        neutral = Image.new("L", size, 232)
        softened = depth.point(lambda value: min(220, 72 + int(value * 0.55)))
        neutral.paste(softened, mask=mask)
        return neutral.convert("RGB")
    if mode in {"depth", "edge", "canny", "mask", "gray", "skeleton"}:
        image = ImageOps.autocontrast(image.convert("L")).convert("RGB")
    return image


def _make_contact_sheet(items: list[dict[str, str]], output: Path, tile: int = 256) -> str:
    if not items:
        return ""
    pad = 10
    label_h = 30
    cols = min(4, len(items))
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (pad + cols * (tile + pad), pad + rows * (tile + label_h + pad)), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(items):
        image = Image.open(item["file"]).convert("RGB")
        image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        x = pad + (index % cols) * (tile + pad)
        y = pad + (index // cols) * (tile + label_h + pad)
        draw.text((x, y + 8), item["label"], fill=(238, 238, 238))
        canvas.paste(image, (x, y + label_h + (tile - image.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return str(output)


def _install_diffsynth_path(path: Path) -> None:
    source = str(path)
    if source not in sys.path:
        sys.path.insert(0, source)


def _install_transformers_compat() -> None:
    import transformers

    if not hasattr(transformers, "DINOv3ViTImageProcessor") and hasattr(transformers, "DINOv3ViTImageProcessorFast"):
        transformers.DINOv3ViTImageProcessor = transformers.DINOv3ViTImageProcessorFast


def _patch_lite_controlnet_config(controlnet_path: Path) -> str:
    from diffsynth.configs import MODEL_CONFIGS
    from diffsynth.core.loader.file import hash_model_file

    model_hash = hash_model_file(str(controlnet_path))
    if not any(config.get("model_hash") == model_hash for config in MODEL_CONFIGS):
        MODEL_CONFIGS.insert(
            0,
            {
                "model_hash": model_hash,
                "model_name": "z_image_controlnet",
                "model_class": "diffsynth.models.z_image_controlnet.ZImageControlNet",
                "extra_kwargs": {"control_layers_places": (0, 10, 20), "control_in_dim": 33, "n_refiner_layers": 2},
            },
        )
    return model_hash


def _siglip_state_dict(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    state_dict = load_file(str(path), device="cpu")
    if state_dict and not next(iter(state_dict)).startswith("vision_model."):
        state_dict = {f"vision_model.{key}": value for key, value in state_dict.items()}
    return state_dict


def _load_pipeline(args: argparse.Namespace, dtype: Any, load_i2l_encoders: bool):
    from diffsynth.pipelines.z_image import ModelConfig, ZImagePipeline

    base_model = _workspace_path(args.base_model)
    device = args.device
    vram = _vram_config(dtype, device, args.offload_device)
    model_configs = [
        ModelConfig(path=_glob_required(str(base_model / "transformer" / "*.safetensors")), **vram),
        ModelConfig(path=_glob_required(str(base_model / "text_encoder" / "*.safetensors")), **vram),
        ModelConfig(path=str(_require(base_model / "vae" / "diffusion_pytorch_model.safetensors")), **vram),
    ]
    if load_i2l_encoders:
        siglip_model = _require(args.siglip_model)
        model_configs.extend(
            [
                ModelConfig(path=str(siglip_model), state_dict=_siglip_state_dict(siglip_model), **vram),
                ModelConfig(path=str(_require(args.dinov3_model)), **vram),
                ModelConfig(path=str(_require(args.i2l_model)), **vram),
            ]
        )
    controlnet_file = _workspace_path(args.controlnet_file) if args.controlnet_file else None
    control_hash = ""
    if controlnet_file and controlnet_file.exists() and args.control_image:
        if args.patch_lite_controlnet:
            control_hash = _patch_lite_controlnet_config(controlnet_file)
        model_configs.append(ModelConfig(path=str(controlnet_file), **vram))

    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=str(_require(base_model / "tokenizer"))),
        vram_limit=args.vram_limit,
    )
    return pipe, control_hash


def _load_or_make_lora(pipe: Any, args: argparse.Namespace, dtype: Any, lora_path: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file
    from diffsynth.pipelines.z_image import ZImageUnit_Image2LoRAEncode, ZImageUnit_Image2LoRADecode

    if lora_path.exists() and not args.force_lora:
        return load_file(str(lora_path), device="cpu")

    style_paths = [_require(path) for path in args.style_images]
    images = [Image.open(path).convert("RGB") for path in style_paths]
    with torch.no_grad():
        embs = ZImageUnit_Image2LoRAEncode().process(pipe, image2lora_images=images)
        lora = ZImageUnit_Image2LoRADecode().process(pipe, **embs)["lora"]
    lora_cpu = {key: value.detach().to("cpu", dtype=dtype) for key, value in lora.items()}
    lora_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora_cpu, str(lora_path))
    return lora


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output = _workspace_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _install_diffsynth_path(_workspace_path(args.diffsynth_path))
    _install_transformers_compat()

    import torch
    from diffsynth.core.loader.file import hash_model_file
    from diffsynth.utils.controlnet import ControlNetInput

    dtype = _torch_dtype(args.dtype)
    lora_path = _workspace_path(args.lora_output) if args.lora_output else output / "zimage_i2l_style_lora.safetensors"
    lora_exists = lora_path.exists() and not args.force_lora
    load_i2l_encoders = not lora_exists
    pipe, control_hash = _load_pipeline(args, dtype, load_i2l_encoders)
    lora = _load_or_make_lora(pipe, args, dtype, lora_path)
    if args.lora_scale != 1.0:
        lora = {key: value * args.lora_scale for key, value in lora.items()}

    contact_items: list[dict[str, str]] = []
    for index, path in enumerate(args.style_images):
        contact_items.append({"label": f"style {index}", "file": str(_workspace_path(path))})

    generated: list[dict[str, Any]] = []
    size = (args.width, args.height)
    input_image = None
    if args.input_image:
        input_image = Image.open(_require(args.input_image)).convert("RGB").resize(size, Image.Resampling.LANCZOS)
        input_path = output / "input_image.png"
        input_image.save(input_path)
        contact_items.append({"label": "input", "file": str(input_path)})
    control_image = None
    controlnet_inputs = None
    if args.control_image:
        control_image = _prepare_control_image(_require(args.control_image), size, args.control_mode)
        control_path = output / "control_image.png"
        control_image.save(control_path)
        contact_items.append({"label": "control", "file": str(control_path)})
        controlnet_inputs = [
            ControlNetInput(
                image=control_image,
                scale=args.control_scale,
                start=args.control_start,
                end=args.control_end,
            )
        ]

    if args.generate:
        for index in range(args.samples):
            seed = args.seed + index
            with torch.no_grad():
                image = pipe(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    seed=seed,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.steps,
                    positive_only_lora=lora,
                    sigma_shift=args.sigma_shift,
                    height=args.height,
                    width=args.width,
                    input_image=input_image,
                    denoising_strength=args.denoising_strength,
                    controlnet_inputs=controlnet_inputs,
                )
            image_path = output / f"sample_{index:02d}_seed_{seed}.png"
            image.save(image_path)
            generated.append({"seed": seed, "file": str(image_path)})
            contact_items.append({"label": f"sample {index}", "file": str(image_path)})

    contact = _make_contact_sheet(contact_items, output / "zimage_i2l_contact.png")
    hashes = {
        "dit": hash_model_file(_glob_required(str(_workspace_path(args.base_model) / "transformer" / "*.safetensors"))),
        "text_encoder": hash_model_file(_glob_required(str(_workspace_path(args.base_model) / "text_encoder" / "*.safetensors"))),
        "vae": hash_model_file(str(_workspace_path(args.base_model) / "vae" / "diffusion_pytorch_model.safetensors")),
        "i2l": hash_model_file(str(_workspace_path(args.i2l_model))),
        "siglip": hash_model_file(str(_workspace_path(args.siglip_model))),
        "dinov3": hash_model_file(str(_workspace_path(args.dinov3_model))),
    }
    if args.control_image and args.controlnet_file:
        hashes["controlnet"] = control_hash or hash_model_file(str(_workspace_path(args.controlnet_file)))

    report = {
        "type": "zimage_i2l_lora_probe",
        "status": "complete",
        "elapsed_seconds": round(time.time() - started, 3),
        "output": str(output),
        "lora": str(lora_path),
        "lora_tensor_count": len(lora),
        "lora_scale": args.lora_scale,
        "reused_lora": lora_exists,
        "loaded_i2l_encoders": load_i2l_encoders,
        "generated": generated,
        "contact_sheet": contact,
        "style_images": [str(_workspace_path(path)) for path in args.style_images],
        "input_image": str(_workspace_path(args.input_image)) if args.input_image else "",
        "denoising_strength": args.denoising_strength,
        "control_image": str(_workspace_path(args.control_image)) if args.control_image else "",
        "control_start": args.control_start,
        "control_end": args.control_end,
        "model_hashes": hashes,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "steps": args.steps,
        "width": args.width,
        "height": args.height,
        "cfg_scale": args.cfg_scale,
        "sigma_shift": args.sigma_shift,
        "device": args.device,
        "dtype": args.dtype,
        "offload_device": args.offload_device,
    }
    report_path = output / "zimage_i2l_lora_probe_report.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and test a Z-Image-i2L positive-only appearance LoRA.")
    parser.add_argument("--output", default="outputs/flux2_klein_high_quality_car_reference/zimage_i2l_lora_probe")
    parser.add_argument("--diffsynth-path", default="tools/DiffSynth-Studio")
    parser.add_argument("--base-model", default="models/Z-Image")
    parser.add_argument("--i2l-model", default="models/Z-Image-i2L/model.safetensors")
    parser.add_argument("--siglip-model", default="models/General-Image-Encoders/SigLIP2-G384/model.safetensors")
    parser.add_argument("--dinov3-model", default="models/General-Image-Encoders/DINOv3-7B/model.safetensors")
    parser.add_argument("--controlnet-file", default="")
    parser.add_argument("--input-image", default="")
    parser.add_argument("--denoising-strength", type=float, default=1.0)
    parser.add_argument("--control-image", default="")
    parser.add_argument("--control-mode", default="depth")
    parser.add_argument("--control-scale", type=float, default=0.7)
    parser.add_argument("--control-start", type=float, default=0.0)
    parser.add_argument("--control-end", type=float, default=1.0)
    parser.add_argument("--patch-lite-controlnet", action="store_true")
    parser.add_argument("--style-images", nargs="+", default=DEFAULT_STYLE_IMAGES)
    parser.add_argument("--lora-output", default="")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--force-lora", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--prompt", default="premium red racecar studio product render, glossy red bodywork, black glass canopy, carbon fiber aerodynamic details, realistic hard-surface car design, clean white studio lighting")
    parser.add_argument("--negative-prompt", default="gray clay, white plastic, line art, wireframe, copied control image, blurry, low quality, warped wheels, extra parts, text, watermark")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--sigma-shift", type=float, default=8.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--offload-device", default="cpu")
    parser.add_argument("--vram-limit", type=float, default=None)
    args = parser.parse_args()
    report = run(args)
    print("Wrote report:", report["report"])
    print("LoRA:", report["lora"])
    print("Contact:", report["contact_sheet"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
