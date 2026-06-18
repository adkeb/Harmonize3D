# Dual-Image Staged Weight Control Research

Date: 2026-06-13

## Decision

Pause the current Flux2 Klein ordinary multi-reference tuning path. The observed failure mode is structural, not just a prompt issue:

- Overweighting the appearance reference copies the reference perspective and deforms the mesh-conditioned view.
- Overweighting white-model structure images leaks lines, gray clay texture, and mask/edge artifacts into the final render.

Do not continue designing around the local `FluxControlNetPipeline` path. It is available in the local diffusers install, but the user explicitly asked to remove that old local pipeline from the model plan. Keep it out of the next-stage implementation candidate list.

The recommended direction is a newer lightweight dual-branch system:

```text
3D mesh render buffers
  -> structure branch: depth / canny / gray / mask via ControlNet-like internal condition
  -> appearance branch: reference image -> LoRA / image adapter / style embedding
  -> staged scheduler:
       early steps: high structure, low appearance
       middle steps: structure decays, appearance rises
       late steps: low edge, moderate depth/normal, stronger material/style
```

## Model Shortlist

Cutoff for "recent": 2025-12-13 to 2026-06-13.

### 1. Z-Image-Turbo + Fun ControlNet Union + Z-Image-i2L

Status: primary candidate.

Why:

- Z-Image-Turbo-Fun-Controlnet-Union-2.1 has 2026.01 and 2026.02 updates.
- It supports Canny, Depth, Pose, MLSD, HED, Scribble, and Gray control.
- Lite versions apply control to fewer layers, intended to reduce artifacts and suit lower-spec machines.
- It uses `control_context_scale`, which maps well to staged structure-weight schedules.
- Z-Image-i2L converts an image reference into a LoRA-like appearance/style adapter, separating appearance from structure.
- License is Apache-2.0 on the checked model cards.

Evidence:

- Z-Image ControlNet model card lists 2026.02.26 2602 update with Gray Control, 2026.01.12 2601 update with Scribble Control and lite models, and 2025.12.22 8-step distillation.
- The same card lists control conditions and notes the optimal `control_context_scale` range.
- Z-Image-i2L model card describes image-to-LoRA for style preservation and recommends positive-only LoRA use.

Links:

- https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
- https://huggingface.co/DiffSynth-Studio/Z-Image-i2L

Expected local experiment:

- Start with the lite 2602 8-step ControlNet Union if available or downloadable via direct non-proxy route.
- Use depth or gray control as the first structure channel; add canny only in early steps.
- Convert the car reference image to a LoRA via i2L or use a standard LoRA/reference adapter path.

### 2. FLUX.2 Klein 4B

Status: keep as fallback only.

Why:

- It is recent, 4B, Apache-2.0, and already local.
- It supports multi-reference editing and runs around 13GB VRAM.

Why not primary:

- It does not expose independent structure-reference and appearance-reference weights in the current pipeline.
- Passing depth/edge/reference as ordinary `image` inputs causes exactly the conflict observed in this project.

Links:

- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B

### 3. TeleStyle

Status: style-transfer fallback, not a structure-control backend.

Why:

- 2026.01 release line.
- Content + style two-image workflow is conceptually close to dual-image guidance.
- LoRA weights are small.

Why not primary:

- It is not designed for depth/canny/mesh render buffers as the main structure condition.

Links:

- https://huggingface.co/Tele-AI/TeleStyle

### 4. Qwen-Image-2512 Fun ControlNet Union

Status: heavy backup.

Why:

- Strong control conditions and Apache-2.0.

Why not primary:

- Qwen-Image base is around 20B, outside the "not too large" preference.

Links:

- https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union

### Rechecked Alternatives

Direct HF API search via `curl --noproxy '*'` did not surface a better recent lightweight candidate.

| Candidate | Last modified | Known size | License signal | Decision |
|---|---:|---:|---|---|
| `mzbac/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps-8bit` | 2025-12-24 | 3.91 GiB | no license tag | Not better than official lite 2602; unclear license. |
| `neuralvfx/Z-Image-SAM-ControlNet` | 2026-03-30 | 23.98 GiB | Apache-2.0 | Too large for this stage. |
| `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` | 2026-02-13 | 15.36 GiB control files | other | Not suitable as default; also targets FLUX.2-dev, not local Klein. |
| `B4100/FLUX.2-dev-Fun-Controlnet-Union` | 2026-04-20 | 15.36 GiB control files | other | Mirror-like candidate; same concerns. |
| `JasonXF/Flux2-dev-controlnet-lora-weights` | 2026-05-28 | 505 GiB known blobs | Apache-2.0 | Dataset/checkpoint dump scale; not practical. |
| `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` | 2026-02-24 | 1.88 GiB lite control file | Apache-2.0 | Non-turbo base is still about 19.14 GiB and needs more steps. |

This keeps the primary recommendation unchanged: the first real generation test should use official Z-Image Turbo + official 2602 lite ControlNet if the base-model size is acceptable. Otherwise, continue searching rather than falling back to Flux2 Klein ordinary multi-reference mode.

## Scheduling Mechanism

Use a 3-phase schedule. Do not use 3-step Flux2 Klein style smoke tests for this research question; there are too few denoising steps to separate layout from material refinement.

### Recommended Schedule

For Z-Image 8-step turbo variants:

| Step range | Structure control | Appearance reference / LoRA | Notes |
|---|---:|---:|---|
| 0-2 | 0.85-1.00 | 0.00-0.15 | Establish silhouette, depth, large parts. Prefer depth/gray. |
| 3-5 | 0.65 -> 0.35 | 0.20 -> 0.55 | Begin material transfer. Keep canny decreasing to avoid line leakage. |
| 6-8 | 0.15-0.30 | 0.55-0.75 | Preserve paint/glass/carbon. Edge nearly off; depth/gray can stay low. |

For non-turbo 20-32 step variants:

| Step ratio | Structure control | Appearance reference / LoRA |
|---|---:|---:|
| 0.00-0.35 | high | off/very low |
| 0.35-0.70 | decay | rising |
| 0.70-1.00 | low | medium/high |

## Implementation Implications

The next backend should not encode structure as ordinary reference images. It should expose separate concepts:

```json
{
  "structure_conditions": [
    {"type": "depth", "file": "...", "schedule": [[0.0, 0.95], [0.5, 0.45], [1.0, 0.2]]},
    {"type": "canny", "file": "...", "schedule": [[0.0, 0.75], [0.45, 0.15], [0.65, 0.0]]}
  ],
  "appearance_conditions": [
    {"type": "lora_or_ip_adapter", "file": "...", "schedule": [[0.0, 0.0], [0.4, 0.35], [1.0, 0.7]]}
  ]
}
```

Required report fields:

- `structure_schedule`
- `appearance_schedule`
- `structure_condition_types`
- `appearance_condition_type`
- `num_inference_steps`
- `edge_leak_score`
- `structure_scores`
- `appearance_scores`
- `manual_visual_review`

## Validation Gates

Add new gates before declaring success:

- Structure: silhouette IoU and edge chamfer pass for every view, not just the problem view.
- No line leakage: generated image should not contain high-contrast structure-map lines that align too strongly with the input canny map.
- Appearance: red paint, black glass, carbon, and wheel style remain consistent without copying the reference perspective.
- View agnostic: no view-id-specific logic.
- Contact sheets: white render, structure conditions, direct output, final output, and reference image must be shown together.

## Source Notes

- Diffusers ControlNet API documents `controlnet_conditioning_scale`, `control_guidance_start`, and `control_guidance_end`, which support start/end staged structure control.
- Diffusers IP-Adapter docs describe image-prompt cross-attention and `set_ip_adapter_scale()`, suitable for appearance reference influence.
- Diffusers callbacks can modify pipeline behavior after denoising steps, useful for step-dependent schedule logic.
- Z-Image ControlNet's model card specifically mentions lite models, multi-control support, `control_context_scale`, and step/scale testing.

## Current Open Item

The stalled sub-agent was shut down and the probe was run locally.

## Direct Non-Proxy Availability Check

Proxy environment at check time:

```text
HTTP_PROXY=
HTTPS_PROXY=
NO_PROXY=
```

Direct `curl --noproxy '*'` HEAD/API checks against `hf-mirror.com` succeeded for the relevant files. Disk free space on `/root/sakura/work/Harmonize3D` was about 680 GiB.

| Model/file | Last modified | Size | Notes |
|---|---:|---:|---|
| `Tongyi-MAI/Z-Image-Turbo` | 2026-01-30 | 30.64 GiB | Full diffusers repo. This is the heavy part. |
| `Tongyi-MAI/Z-Image` | 2026-01-28 | 19.14 GiB | Non-turbo base; smaller than Turbo repo but not the 8-step path. |
| `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors` | 2026-02-26 | 1.88 GiB | Best first ControlNet candidate: recent, lite, 8-step, includes Gray/Canny/Depth. |
| `DiffSynth-Studio/Z-Image-i2L/model.safetensors` | 2026-01-28 | 3.00 GiB | Image-to-LoRA appearance branch candidate. |

Do not start a large base-model download just to compensate for a weak interface. The interface must support separated structure/appearance control first.

## Local Diffusers Z-Image Interface Check

The local diffusers install includes:

- `ZImagePipeline`
- `ZImageControlNetPipeline`
- `ZImageControlNetInpaintPipeline`
- `ZImageControlNetModel`

Observed interface:

```text
ZImageControlNetPipeline.__call__(
  prompt,
  control_image,
  controlnet_conditioning_scale=0.75,
  latents=None,
  callback_on_step_end=...,
)
```

Important findings:

- Structure can be passed as `control_image`, which is the correct internal ControlNet branch rather than an ordinary reference image.
- `ZImageControlNetPipeline` supports LoRA loading and adapter weights through `load_lora_weights()` and `set_adapters()`.
- The exposed ControlNet scale is fixed for the whole call; there is no built-in `control_guidance_start/end` parameter in this Z-Image pipeline.
- The denoise callback exposes `latents` and `prompt_embeds`, but not `controlnet_conditioning_scale`.
- A true schedule is still possible by wrapping or subclassing the ControlNet forward path, because `ZImageControlNetModel.forward(..., conditioning_scale=...)` receives the scale at every denoise step.

Practical staged implementation for the next actual model test:

1. Load Z-Image-Turbo + the lite 2602 ControlNet.
2. Wrap `pipe.controlnet.forward` so each denoise call overrides `conditioning_scale` from a generic schedule, not a view-id branch.
3. Use depth/gray as the main structure control, with canny only early or in a separate low-weight pass if the implementation supports multiple conditions.
4. Load an appearance LoRA generated by i2L or a compatible appearance adapter, set initial adapter weight before generation, then raise it through `callback_on_step_end` for subsequent steps.
5. Use at least 8 steps for Turbo and 20-32 steps for non-turbo; 3-step smoke tests are not meaningful for staged weighting.

## Z-Image Staged Control Dry Run

Script:

```bash
PYTHONPATH=src .venv/bin/python3 scripts/probe_zimage_staged_control.py \
  --output outputs/flux2_klein_high_quality_car_reference/zimage_staged_control_probe \
  --steps 8 --width 768 --height 768
```

Artifacts:

- Report: `outputs/flux2_klein_high_quality_car_reference/zimage_staged_control_probe/zimage_staged_control_probe_report.json`
- Control preview contact: `outputs/flux2_klein_high_quality_car_reference/zimage_staged_control_probe/zimage_control_previews_contact.png`

Dry-run result:

```json
{
  "status": "dry_run",
  "ready_to_generate": false,
  "view_ids": ["view_locked", "view_left_30", "view_right_30"],
  "structure_channels": ["depth", "gray", "canny"],
  "missing_manifest_channels": 0,
  "zimage_interface": {
    "available": true,
    "has_control_image": true,
    "has_controlnet_conditioning_scale": true,
    "controlnet_forward_has_conditioning_scale": true,
    "supports_lora_loader": true,
    "supports_adapter_weights": true,
    "has_native_control_guidance_start_end": false
  }
}
```

The dry run generated control previews for every manifest view and every requested structure channel. It does not branch on fixed view IDs. The schedule wrapper self-test passed, proving the proposed ControlNet forward wrapper can override per-step structure scale without relying on the pipeline's static `controlnet_conditioning_scale` argument.

Generated 8-step schedules:

| Step | Structure scale | Appearance scale |
|---:|---:|---:|
| 0 | 0.950 | 0.000 |
| 1 | 0.909 | 0.061 |
| 2 | 0.868 | 0.122 |
| 3 | 0.738 | 0.240 |
| 4 | 0.534 | 0.403 |
| 5 | 0.343 | 0.557 |
| 6 | 0.271 | 0.629 |
| 7 | 0.200 | 0.700 |

Current generation blocker:

- `models/Z-Image-Turbo` is missing.
- `models/Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors` is missing.
- Optional `models/Z-Image-i2L/model.safetensors` is missing.

Config also contains a disabled experimental entry:

- `models.zimage_turbo_staged_control`

It is intentionally not enabled as a default until actual generation passes structure, line-leakage, and appearance consistency gates.

## Local Probe: Current Flux2 Klein

Script:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python3 scripts/probe_dual_image_staged_weights.py \
  --manifest outputs/flux2_klein_high_quality_car_reference/white_renders/manifest.json \
  --appearance-reference outputs/flux2_klein_high_quality_car_reference/racecar_reference_image2.png \
  --output outputs/flux2_klein_high_quality_car_reference/dual_image_staged_weight_probe \
  --view view_locked --width 768 --height 768 \
  --structure-steps 8 --appearance-steps 8 --total-steps 16 --seed 20260613
```

Artifacts:

- Report: `outputs/flux2_klein_high_quality_car_reference/dual_image_staged_weight_probe/dual_image_staged_weight_probe_report.json`
- Contact sheet: `outputs/flux2_klein_high_quality_car_reference/dual_image_staged_weight_probe/dual_image_staged_weight_probe_contact.png`

Capability finding:

```json
{
  "pipeline": "Flux2KleinPipeline",
  "callback_tensor_inputs": ["latents", "prompt_embeds"],
  "has_per_reference_weight": false,
  "has_controlnet_scale": false,
  "has_ip_adapter_scale": false,
  "true_dual_image_step_weight_schedule_supported": false
}
```

Interpretation:

- Current Flux2 Klein can accept a list of reference images, but it does not expose independent weights for each reference image.
- Its step callback can access `latents` and `prompt_embeds`, not image embeddings or per-reference attention weights.
- Therefore, true "early structure image high, late appearance image high" scheduling is not supported by this pipeline.

Probe results:

| Experiment | Mode | Structure total | Edge chamfer | Edge leak | Visual result |
|---|---|---:|---:|---:|---|
| A_structure_only | structure buffers only, 8 steps | 0.898 | 0.709 | 0.223 | Structure is stable, but output remains clay/white-model-like. |
| B_appearance_only | appearance reference only, 8 steps | 0.250 | 0.000 | 0.144 | Material is strong, but geometry ignores the mesh view. |
| C_simultaneous_structure_appearance | all refs together, 16 steps | 0.738 | 0.584 | 0.331 | Better looking, but structure drops and line/condition leakage rises. |
| D_two_stage_approx | two separate calls, structure then appearance | 0.761 | 0.260 | 0.127 | Not true step scheduling; produces ghosting/partial overlay and poor edge alignment. |

Conclusion:

The current Flux2 Klein path is useful as a fallback renderer, but it is the wrong substrate for real dual-image staged weighting. The next implementation should use a model/interface with separate internal branches:

- structure branch with ControlNet-like scale or `control_context_scale`;
- appearance branch with LoRA/IP/adapter scale;
- a scheduler that changes those branch weights across denoising steps.

This reinforces the primary recommendation: test Z-Image-Turbo + Fun ControlNet Union + i2L/LoRA before investing more in Flux2 Klein prompt/reference ordering.
