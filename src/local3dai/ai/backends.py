from __future__ import annotations

import hashlib
import inspect
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from local3dai.ai.geometry import geometry_lock_render, make_canny_control
from local3dai.manifest import read_manifest, write_manifest


def _open_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _prompt_palette(prompt: str, seed: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    digest = hashlib.sha256(f"{prompt}|{seed}".encode("utf-8")).digest()
    a = tuple(64 + value % 160 for value in digest[:3])
    b = tuple(32 + value % 190 for value in digest[3:6])
    return a, b


def _resize_like(image: Image.Image, ref: Image.Image) -> Image.Image:
    if image.size == ref.size:
        return image
    return image.resize(ref.size, Image.Resampling.LANCZOS)


def _chunk_tokens(token_ids: list[int], chunk_size: int) -> list[list[int]]:
    if not token_ids:
        return [[]]
    return [token_ids[i : i + chunk_size] for i in range(0, len(token_ids), chunk_size)]


def _encode_prompt_tokens(tokenizer: Any, prompt: str) -> list[int]:
    raw_tokenizer = getattr(tokenizer, "_tokenizer", None)
    if raw_tokenizer is not None:
        return list(raw_tokenizer.encode(prompt, add_special_tokens=False).ids)
    return list(tokenizer.encode(prompt, add_special_tokens=False))


def _token_count(tokenizer: Any, prompt: str) -> int:
    return len(_encode_prompt_tokens(tokenizer, prompt))


def _build_padded_chunk(tokenizer: Any, token_chunk: list[int]) -> list[int]:
    max_length = int(tokenizer.model_max_length)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos
    token_chunk = token_chunk[: max_length - 2]
    input_ids = [bos] + token_chunk + [eos]
    input_ids.extend([pad] * (max_length - len(input_ids)))
    return input_ids


def _encode_chunked_prompt_for_encoder(
    *,
    tokenizer: Any,
    text_encoder: Any,
    prompt: str,
    chunk_count: int,
    device: Any,
    torch: Any,
) -> tuple[Any, Any | None]:
    chunk_size = int(tokenizer.model_max_length) - 2
    token_ids = _encode_prompt_tokens(tokenizer, prompt)
    chunks = _chunk_tokens(token_ids, chunk_size)
    chunks.extend([[] for _ in range(max(0, chunk_count - len(chunks)))])
    chunks = chunks[:chunk_count]
    ids = [_build_padded_chunk(tokenizer, chunk) for chunk in chunks]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = text_encoder(input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[-2]
    hidden = hidden.reshape(1, chunk_count * int(tokenizer.model_max_length), hidden.shape[-1])

    pooled = None
    first_output = outputs[0]
    if getattr(first_output, "ndim", 0) == 2:
        pooled = first_output.mean(dim=0, keepdim=True)
    return hidden, pooled


def _encode_long_sdxl_prompt(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str,
    device: Any,
    torch: Any,
) -> dict[str, Any]:
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2] if pipe.tokenizer is not None else [pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2] if pipe.text_encoder is not None else [pipe.text_encoder_2]
    chunk_size = int(tokenizers[0].model_max_length) - 2

    token_counts = {
        "prompt": max(_token_count(tokenizer, prompt) for tokenizer in tokenizers),
        "negative_prompt": max(_token_count(tokenizer, negative_prompt or "") for tokenizer in tokenizers),
    }
    chunk_count = max(1, (max(token_counts.values()) + chunk_size - 1) // chunk_size)

    prompt_embeds_list = []
    negative_embeds_list = []
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        prompt_embeds, pooled = _encode_chunked_prompt_for_encoder(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=prompt,
            chunk_count=chunk_count,
            device=device,
            torch=torch,
        )
        negative_embeds, negative_pooled = _encode_chunked_prompt_for_encoder(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=negative_prompt or "",
            chunk_count=chunk_count,
            device=device,
            torch=torch,
        )
        prompt_embeds_list.append(prompt_embeds)
        negative_embeds_list.append(negative_embeds)
        if pooled is not None:
            pooled_prompt_embeds = pooled
        if negative_pooled is not None:
            negative_pooled_prompt_embeds = negative_pooled

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    negative_prompt_embeds = torch.concat(negative_embeds_list, dim=-1)
    target_dtype = pipe.text_encoder_2.dtype if getattr(pipe, "text_encoder_2", None) is not None else pipe.unet.dtype
    prompt_embeds = prompt_embeds.to(dtype=target_dtype, device=device)
    negative_prompt_embeds = negative_prompt_embeds.to(dtype=target_dtype, device=device)
    if pooled_prompt_embeds is None:
        pooled_prompt_embeds = torch.zeros((1, prompt_embeds.shape[-1]), dtype=target_dtype, device=device)
    else:
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=target_dtype, device=device)
    if negative_pooled_prompt_embeds is None:
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    else:
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(dtype=target_dtype, device=device)

    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        "long_prompt_chunk_count": chunk_count,
        "long_prompt_token_counts": token_counts,
    }


class MockImageBackend:
    name = "mock"

    def generate(
        self,
        render_manifest_path: str | Path,
        output_dir: str | Path,
        *,
        prompt: str,
        candidates_per_view: int,
        seed: int,
        **_: Any,
    ) -> Path:
        manifest = read_manifest(render_manifest_path)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {
            "type": "ai_manifest",
            "backend": self.name,
            "prompt": prompt,
            "candidates": [],
        }
        rng = random.Random(seed)
        for view in manifest.get("views", []):
            files = view.get("files", {})
            rgb = _open_rgb(files["rgb"])
            depth = _resize_like(_open_rgb(files["depth"]).convert("L"), rgb)
            edge = _resize_like(_open_rgb(files["edge"]).convert("L"), rgb)
            mask = _resize_like(_open_rgb(files["mask"]).convert("L"), rgb)
            for candidate_index in range(candidates_per_view):
                candidate_seed = rng.randint(0, 2**31 - 1)
                image = self._stylize(rgb, depth, edge, mask, prompt=prompt, seed=candidate_seed)
                view_dir = output / view["view_id"]
                view_dir.mkdir(parents=True, exist_ok=True)
                image_path = view_dir / f"candidate_{candidate_index:02d}.png"
                image.save(image_path)
                result["candidates"].append(
                    {
                        "view_id": view["view_id"],
                        "candidate_id": f"{view['view_id']}_{candidate_index:02d}",
                        "seed": candidate_seed,
                        "file": str(image_path),
                        "source_files": files,
                    }
                )
        return write_manifest(output / "manifest.json", result)

    def _stylize(self, rgb: Image.Image, depth: Image.Image, edge: Image.Image, mask: Image.Image, *, prompt: str, seed: int) -> Image.Image:
        color_a, color_b = _prompt_palette(prompt, seed)
        base = ImageOps.autocontrast(rgb, cutoff=1)
        base = ImageEnhance.Color(base).enhance(1.12)
        base = ImageEnhance.Contrast(base).enhance(1.08)

        w, h = base.size
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        ramp = (0.58 * xx + 0.42 * yy)
        c1 = np.array(color_a, dtype=np.float32)
        c2 = np.array(color_b, dtype=np.float32)
        grad = (c1 * (1 - ramp[..., None]) + c2 * ramp[..., None]).astype(np.uint8)
        grad_img = Image.fromarray(grad, "RGB").filter(ImageFilter.GaussianBlur(0.8))

        depth_rgb = ImageOps.colorize(depth, black=(20, 24, 30), white=(255, 245, 220))
        composed = Image.blend(base, grad_img, 0.24)
        composed = Image.blend(composed, depth_rgb, 0.18)

        edge_mask = ImageOps.invert(ImageOps.autocontrast(edge)).filter(ImageFilter.GaussianBlur(0.35))
        ink = Image.new("RGB", base.size, (15, 18, 22))
        composed = Image.composite(ink, composed, ImageOps.invert(edge_mask).point(lambda px: 255 if px > 210 else 0))
        mask_blur = mask.filter(ImageFilter.GaussianBlur(1.2))
        background = Image.new("RGB", base.size, (236, 238, 241))
        result = Image.composite(composed, background, mask_blur)
        result = ImageEnhance.Sharpness(result).enhance(1.25)
        return result


class DiffusersImageBackend:
    name = "diffusers"

    def generate(
        self,
        render_manifest_path: str | Path,
        output_dir: str | Path,
        *,
        prompt: str,
        candidates_per_view: int,
        seed: int,
        model_ref: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        variant: str | None = None,
        attention_slicing: bool = False,
        steps: int = 24,
        guidance_scale: float = 3.5,
        strength: float = 0.62,
        **_: Any,
    ) -> Path:
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Diffusers backend requires torch and diffusers. Run scripts/install_ai_cuda130.sh, "
                "then set ai.default_backend to diffusers-flux or pass --backend diffusers-flux."
            ) from exc

        if steps < 1:
            raise ValueError("Diffusers image-to-image rendering requires steps >= 1.")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Diffusers image-to-image rendering requires 0.0 <= strength <= 1.0.")
        if int(steps * strength) < 1:
            raise ValueError("Diffusers image-to-image rendering requires steps * strength >= 1.")

        torch_dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype.lower(), torch.bfloat16)
        load_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if variant:
            load_kwargs["variant"] = variant
        pipe = AutoPipelineForImage2Image.from_pretrained(model_ref, **load_kwargs)
        pipe.to(device)
        if attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        call_params = inspect.signature(pipe.__call__).parameters
        manifest = read_manifest(render_manifest_path)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {
            "type": "ai_manifest",
            "backend": self.name,
            "model_ref": model_ref,
            "variant": variant or "",
            "prompt": prompt,
            "candidates": [],
        }
        for view in manifest.get("views", []):
            files = view.get("files", {})
            init_image = _open_rgb(files["rgb"])
            for candidate_index in range(candidates_per_view):
                candidate_seed = seed + candidate_index + len(result["candidates"]) * 997
                generator = torch.Generator(device=device.split(":")[0]).manual_seed(candidate_seed)
                kwargs: dict[str, Any] = {
                    "prompt": prompt,
                    "image": init_image,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance_scale,
                    "strength": strength,
                    "generator": generator,
                }
                kwargs = {key: value for key, value in kwargs.items() if key in call_params}
                image = pipe(**kwargs).images[0]
                view_dir = output / view["view_id"]
                view_dir.mkdir(parents=True, exist_ok=True)
                image_path = view_dir / f"candidate_{candidate_index:02d}.png"
                image.save(image_path)
                result["candidates"].append(
                    {
                        "view_id": view["view_id"],
                        "candidate_id": f"{view['view_id']}_{candidate_index:02d}",
                        "seed": candidate_seed,
                        "file": str(image_path),
                        "source_files": files,
                    }
                )
        return write_manifest(output / "manifest.json", result)


class SDXLControlNetGeometryBackend:
    name = "sdxl-controlnet-geometry"

    def generate(
        self,
        render_manifest_path: str | Path,
        output_dir: str | Path,
        *,
        prompt: str,
        candidates_per_view: int,
        seed: int,
        model_ref: str = "",
        model_config: dict[str, Any] | None = None,
        negative_prompt: str = "",
        device: str = "cuda:0",
        dtype: str = "float16",
        variant: str | None = "fp16",
        steps: int = 42,
        guidance_scale: float = 8.0,
        strength: float = 0.68,
        width: int = 1536,
        height: int = 1536,
        control_only: bool = True,
        canny_scale: float = 2.85,
        depth_scale: float = 0.55,
        canny_low: int = 80,
        canny_high: int = 180,
        geometry_lock: bool = True,
        **_: Any,
    ) -> Path:
        try:
            import torch
            from diffusers import ControlNetModel, EulerAncestralDiscreteScheduler
            if control_only:
                from diffusers import StableDiffusionXLControlNetPipeline as Pipeline
            else:
                from diffusers import StableDiffusionXLControlNetImg2ImgPipeline as Pipeline
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "SDXL ControlNet backend requires torch, diffusers, transformers, and accelerate. "
                "Run scripts/install_ai_cuda130.sh and keep the SDXL/ControlNet weights in models/."
            ) from exc

        cfg = model_config or {}
        base_model = model_ref or cfg.get("base_model") or cfg.get("local_path") or cfg.get("model_id")
        canny_controlnet = cfg.get("canny_controlnet", "")
        depth_controlnet = cfg.get("depth_controlnet", "")
        if not base_model:
            raise RuntimeError("SDXL ControlNet backend needs a base model path or model_ref.")
        for label, value in {
            "base_model": base_model,
            "canny_controlnet": canny_controlnet,
            "depth_controlnet": depth_controlnet,
        }.items():
            if not value:
                raise RuntimeError(f"SDXL ControlNet backend is missing {label} in model config.")
            if Path(value).is_absolute() and not Path(value).exists():
                raise RuntimeError(f"Configured {label} does not exist: {value}")

        if steps < 1:
            raise ValueError("SDXL ControlNet rendering requires steps >= 1.")
        if width < 64 or height < 64:
            raise ValueError("SDXL ControlNet rendering requires width/height >= 64.")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("SDXL ControlNet img2img mode requires 0.0 <= strength <= 1.0.")

        torch_dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype.lower(), torch.float16)

        controlnets = [
            ControlNetModel.from_pretrained(
                str(canny_controlnet),
                torch_dtype=torch_dtype,
                variant=variant,
                use_safetensors=True,
            ),
            ControlNetModel.from_pretrained(
                str(depth_controlnet),
                torch_dtype=torch_dtype,
                variant=variant,
                use_safetensors=True,
            ),
        ]
        pipe = Pipeline.from_pretrained(
            str(base_model),
            controlnet=controlnets,
            torch_dtype=torch_dtype,
            variant=variant,
            use_safetensors=True,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
        )
        pipe.to(device)
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        elif hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        tokenizers = [pipe.tokenizer, pipe.tokenizer_2] if pipe.tokenizer is not None else [pipe.tokenizer_2]
        chunk_size = int(tokenizers[0].model_max_length) - 2
        prompt_token_count = max(_token_count(tokenizer, prompt) for tokenizer in tokenizers)
        negative_token_count = max(_token_count(tokenizer, negative_prompt or "") for tokenizer in tokenizers)
        use_long_prompt = max(prompt_token_count, negative_token_count) > chunk_size
        long_prompt_kwargs: dict[str, Any] = {}
        long_prompt_meta: dict[str, Any] = {
            "enabled": False,
            "chunk_size": chunk_size,
            "chunk_count": 1,
            "token_counts": {
                "prompt": prompt_token_count,
                "negative_prompt": negative_token_count,
            },
        }
        if use_long_prompt:
            execution_device = getattr(pipe, "_execution_device", None)
            if execution_device is None:
                execution_device = torch.device(device)
            encoded_prompt = _encode_long_sdxl_prompt(
                pipe,
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                device=execution_device,
                torch=torch,
            )
            long_prompt_kwargs = {
                "prompt_embeds": encoded_prompt["prompt_embeds"],
                "negative_prompt_embeds": encoded_prompt["negative_prompt_embeds"],
                "pooled_prompt_embeds": encoded_prompt["pooled_prompt_embeds"],
                "negative_pooled_prompt_embeds": encoded_prompt["negative_pooled_prompt_embeds"],
            }
            long_prompt_meta = {
                "enabled": True,
                "chunk_size": chunk_size,
                "chunk_count": encoded_prompt["long_prompt_chunk_count"],
                "token_counts": encoded_prompt["long_prompt_token_counts"],
            }

        manifest = read_manifest(render_manifest_path)
        output = Path(output_dir)
        controls_dir = output / "controls"
        direct_dir = output / "direct"
        output.mkdir(parents=True, exist_ok=True)
        controls_dir.mkdir(parents=True, exist_ok=True)
        direct_dir.mkdir(parents=True, exist_ok=True)

        result: dict[str, Any] = {
            "type": "ai_manifest",
            "backend": self.name,
            "base_model": str(base_model),
            "canny_controlnet": str(canny_controlnet),
            "depth_controlnet": str(depth_controlnet),
            "variant": variant or "",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "geometry_lock": geometry_lock,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "canny_scale": canny_scale,
            "depth_scale": depth_scale,
            "control_only": control_only,
            "long_prompt": long_prompt_meta,
            "candidates": [],
        }
        size = (width, height)
        for view in manifest.get("views", []):
            files = view.get("files", {})
            init_image = _open_rgb(files["rgb"]).resize(size, Image.Resampling.LANCZOS)
            depth_image = _open_rgb(files["depth"]).resize(size, Image.Resampling.LANCZOS)
            canny_image = make_canny_control(init_image, low=canny_low, high=canny_high)
            canny_path = controls_dir / f"{view['view_id']}_canny.png"
            depth_path = controls_dir / f"{view['view_id']}_depth.png"
            canny_image.save(canny_path)
            depth_image.save(depth_path)

            for candidate_index in range(candidates_per_view):
                candidate_seed = seed + candidate_index + len(result["candidates"]) * 997
                generator = torch.Generator(device=device.split(":")[0]).manual_seed(candidate_seed)
                call_kwargs: dict[str, Any] = {
                    "num_inference_steps": steps,
                    "guidance_scale": guidance_scale,
                    "controlnet_conditioning_scale": [canny_scale, depth_scale],
                    "control_guidance_start": [0.0, 0.0],
                    "control_guidance_end": [0.95, 0.9],
                    "generator": generator,
                    "width": width,
                    "height": height,
                }
                if long_prompt_kwargs:
                    call_kwargs.update(long_prompt_kwargs)
                else:
                    call_kwargs["prompt"] = prompt
                    call_kwargs["negative_prompt"] = negative_prompt
                if control_only:
                    call_kwargs["image"] = [canny_image, depth_image]
                else:
                    call_kwargs["image"] = init_image
                    call_kwargs["control_image"] = [canny_image, depth_image]
                    call_kwargs["strength"] = strength
                image = pipe(**call_kwargs).images[0]
                view_dir = output / view["view_id"]
                view_dir.mkdir(parents=True, exist_ok=True)
                direct_path = direct_dir / f"{view['view_id']}_{candidate_index:02d}.png"
                image.save(direct_path)
                image_path = view_dir / f"candidate_{candidate_index:02d}.png"
                if geometry_lock:
                    geometry_lock_render(
                        source_rgb=files["rgb"],
                        source_mask=files["mask"],
                        source_edge=files["edge"],
                        ai_image=direct_path,
                        output=image_path,
                    )
                else:
                    image.save(image_path)
                result["candidates"].append(
                    {
                        "view_id": view["view_id"],
                        "candidate_id": f"{view['view_id']}_{candidate_index:02d}",
                        "seed": candidate_seed,
                        "file": str(image_path),
                        "direct_file": str(direct_path),
                        "control_canny": str(canny_path),
                        "control_depth": str(depth_path),
                        "source_files": files,
                    }
                )
        return write_manifest(output / "manifest.json", result)


class HiDreamO1ImageBackend:
    name = "hidream-o1-image"

    def generate(
        self,
        render_manifest_path: str | Path,
        output_dir: str | Path,
        *,
        prompt: str,
        candidates_per_view: int,
        seed: int,
        model_ref: str = "",
        model_config: dict[str, Any] | None = None,
        device: str = "cuda:0",
        steps: int = 50,
        guidance_scale: float = 5.0,
        width: int = 1024,
        height: int = 1024,
        **_: Any,
    ) -> Path:
        try:
            import torch
            from transformers import AutoProcessor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("HiDream-O1-Image backend requires torch and transformers.") from exc

        cfg = model_config or {}
        repo_root = Path(cfg.get("repo_root", "tools/HiDream-O1-Image")).resolve()
        model_path = Path(model_ref or cfg.get("local_path") or cfg.get("model_path") or "models/HiDream-O1-Image").resolve()
        model_type = str(cfg.get("model_type", "full"))
        shift = float(cfg.get("shift", 3.0))
        keep_original_aspect = bool(cfg.get("keep_original_aspect", False))
        reference_channels = list(cfg.get("reference_channels", ["rgb"]))
        if model_type != "full":
            raise ValueError("This backend is configured for HiDream-O1-Image full; set model_type='full'.")
        if not repo_root.exists():
            raise RuntimeError(f"HiDream-O1-Image repo_root does not exist: {repo_root}")
        if not model_path.exists():
            raise RuntimeError(f"HiDream-O1-Image model path does not exist: {model_path}")

        sys.path.insert(0, str(repo_root))
        try:
            from inference import add_special_tokens, get_tokenizer
            from models.pipeline import generate_image
            from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(f"Failed to import HiDream-O1-Image official pipeline from {repo_root}.") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("HiDream-O1-Image inference requires CUDA.")

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        manifest = read_manifest(render_manifest_path)
        result: dict[str, Any] = {
            "type": "ai_manifest",
            "backend": self.name,
            "repo_root": str(repo_root),
            "model_path": str(model_path),
            "model_type": model_type,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "shift": shift,
            "reference_channels": reference_channels,
            "candidates": [],
        }

        device_map = "cuda" if device.startswith("cuda") else device
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            local_files_only=True,
        ).eval()
        tokenizer = get_tokenizer(processor)
        add_special_tokens(tokenizer)

        for view in manifest.get("views", []):
            files = view.get("files", {})
            ref_image_paths = [files[channel] for channel in reference_channels if channel in files]
            if not ref_image_paths:
                ref_image_paths = [files["rgb"]]
            for candidate_index in range(candidates_per_view):
                candidate_seed = seed + candidate_index + len(result["candidates"]) * 997
                image = generate_image(
                    model=model,
                    processor=processor,
                    prompt=prompt,
                    ref_image_paths=ref_image_paths,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    shift=shift,
                    timesteps_list=None,
                    scheduler_name="default",
                    seed=candidate_seed,
                    keep_original_aspect=keep_original_aspect,
                )
                view_dir = output / view["view_id"]
                view_dir.mkdir(parents=True, exist_ok=True)
                image_path = view_dir / f"candidate_{candidate_index:02d}.png"
                image.save(image_path)
                result["candidates"].append(
                    {
                        "view_id": view["view_id"],
                        "candidate_id": f"{view['view_id']}_{candidate_index:02d}",
                        "seed": candidate_seed,
                        "file": str(image_path),
                        "source_files": files,
                        "reference_files": ref_image_paths,
                    }
                )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return write_manifest(output / "manifest.json", result)


def build_backend(name: str) -> MockImageBackend | DiffusersImageBackend | SDXLControlNetGeometryBackend | HiDreamO1ImageBackend:
    normalized = name.strip().lower()
    if normalized == "mock":
        return MockImageBackend()
    if normalized in {"diffusers", "diffusers-flux", "diffusers-img2img"}:
        return DiffusersImageBackend()
    if normalized in {"sdxl-controlnet-geometry", "controlnet-geometry", "diffusers-sdxl-controlnet"}:
        return SDXLControlNetGeometryBackend()
    if normalized in {"hidream-o1-image", "hidream-o1", "hidream"}:
        return HiDreamO1ImageBackend()
    raise ValueError(f"Unknown AI backend: {name}")
