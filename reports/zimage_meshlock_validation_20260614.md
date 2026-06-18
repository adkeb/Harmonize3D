# Z-Image MeshLock Validation

Date: 2026-06-14

## Context

This continues session `019ebc45-0d81-7241-b4bc-564d6fb21aab`.

The prior interruption happened while three Z-Image constant-high control probes
(`gray`, `depth`, `canny`) were launched in parallel. The GPU was later idle and
the output folders contained only control previews, so the probes were rerun
serially.

The local Z-Image path now has complete weights:

- Base: `models/Z-Image`
- ControlNet: `models/Z-Image-Fun-Controlnet-Union-2.1/Z-Image-Fun-Controlnet-Union-2.1-lite.safetensors`
- Local lite config: `models/Z-Image-ControlNet-config-lite/config.json`

The lite config uses `control_refiner_layers_places: [0]`, which was the
minimum setting that passed a real ControlNet forward test.

## Probe Update

`scripts/probe_zimage_staged_control.py` now records both direct generation and
optional MeshLock output.

New CLI fields:

- `--mesh-lock-mode none|position|detail|adaptive`
- `--detail-reference PATH`

For generated runs, the report now includes:

- `direct_file`
- `direct_structure_scores`
- final `file`
- final `structure_scores`
- `mesh_lock_mode`

This keeps the direct model behavior auditable instead of hiding it behind
post-processing.

## Constant-High Control Results

Command shape:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python3 scripts/probe_zimage_staged_control.py \
  --steps 12 --width 512 --height 512 --max-views 1 \
  --structure-schedule 0:0.95,1:0.95 \
  --appearance-schedule 0:0,1:0 \
  --generate
```

Single-view direct outputs did not pass the structure gate.

| Channel | Total | Silhouette IoU | Edge Chamfer | Added-Part Penalty | Result |
|---|---:|---:|---:|---:|---|
| gray | 0.520606 | 0.410782 | 0.747923 | 0.559782 | Best direct channel, but foreground/background hallucination remains too high. |
| depth | 0.322047 | 0.211920 | 0.139810 | 0.762272 | Too weak; large red scene hallucination and poor edge fit. |
| canny | 0.338274 | 0.292432 | 0.178402 | 0.687282 | Better material signal, but direct structure still fails. |

Contact sheet:

`outputs/flux2_klein_high_quality_car_reference/zimage_control_constant_512_contact_comparison.png`

Interpretation:

Z-Image ControlNet scheduling is active, but the model's direct output is not
yet a reliable final renderer for this project. It needs the existing
mesh/camera/render-buffer constraint layer.

## Adaptive MeshLock Results

Single-view adaptive MeshLock:

| Channel | Direct Total | Final Total | Final Silhouette IoU | Final Edge Chamfer | Final Added-Part Penalty |
|---|---:|---:|---:|---:|---:|
| gray | 0.520606 | 0.902579 | 0.989548 | 0.755640 | 0.0 |
| depth | 0.322047 | 0.897877 | 0.989927 | 0.703755 | 0.0 |
| canny | 0.338274 | 0.933229 | 0.989336 | 0.877183 | 0.0 |

Contact sheet:

`outputs/flux2_klein_high_quality_car_reference/zimage_control_adaptive_512_contact_comparison.png`

Three-view canny adaptive MeshLock:

| View | Direct Total | Final Total | Final Silhouette IoU | Final Edge Chamfer | Final Added-Part Penalty | Final Background Cleanliness |
|---|---:|---:|---:|---:|---:|---:|
| view_locked | 0.338274 | 0.933229 | 0.989336 | 0.877183 | 0.0 | 0.999079 |
| view_left_30 | 0.297799 | 0.940175 | 0.995809 | 0.889330 | 0.0 | 0.999042 |
| view_right_30 | 0.235248 | 0.924417 | 0.991748 | 0.864963 | 0.0 | 0.998689 |

Mean direct total: `0.290440`

Mean final total: `0.932607`

Report:

`outputs/flux2_klein_high_quality_car_reference/zimage_control_canny_adaptive_512_multiview/zimage_staged_control_probe_report.json`

Contact sheet:

`outputs/flux2_klein_high_quality_car_reference/zimage_control_canny_adaptive_512_multiview/zimage_generation_contact.png`

## Decision

Z-Image non-turbo + lite ControlNet is useful as a candidate generator, but not
as a standalone final renderer. The strongest validated direction is:

```text
Z-Image internal ControlNet candidate
  -> direct output retained for audit
  -> adaptive MeshLock with Blender rgb/mask/edge authority
  -> Structure v2 scoring and contact-sheet review
```

This matches the root improvement direction: the innovation should sit in the
mesh-conditioned multi-view consistency and structure-adherence agent, not in
the raw choice of image model.

## Remaining Work

- Add a real appearance branch. Current tests use no appearance LoRA, so
  material quality is still closer to "mesh recoloring" than a fully transferred
  product-render style.
- Keep direct-vs-final reporting mandatory. Direct Z-Image failures are useful
  diagnostics and should not be hidden by MeshLock.
- Test more than one object class before generalizing beyond the car case.
- If an appearance model is downloaded next, prefer a small LoRA/adapter path
  and keep the same direct non-proxy checks used for previous downloads.
