# Harmonize3D

Harmonize3D is a local 3D-to-image rendering workbench. It starts from a real mesh, renders deterministic Blender structure channels, and lets an AI backend generate product-style images that are checked against the original 3D shape.

The current direction is MeshLock-MV: a lightweight mesh-guided Agent for structure-adherent, multi-view AI rendering. The project does not treat the image model as the main invention; the main control layer is the explicit 3D mesh, locked camera, render channels, structure scoring, and multi-view selection loop around the model.

## Paper And Latest Results

Latest paper artifact and source of truth: [`reports/论文.docx`](reports/%E8%AE%BA%E6%96%87.docx).

Paper title: **Harmonize3D：面向 3D 结构约束与多视图一致性的本地 AI 渲染 Agent**  
Subtitle: **从单对象闭环到模块化全场景生成流程的阶段性研究**  
Author metadata in the paper: 徐洋 / 20300290037 / 计算机科学与技术专业 / 指导教师：徐志平 / 2026 年 6 月

Markdown mirrors generated from the latest DOCX:

- [`docs/harmonize3d_paper.md`](docs/harmonize3d_paper.md)
- [`reports/Harmonize3D_Paper_Deliverable.md`](reports/Harmonize3D_Paper_Deliverable.md)

The paper has been updated with the June 18, 2026 Auto Scene run. This run validates the full modular chain:

```text
concept planning -> module references -> module 3D -> 3D scene assembly -> Blender white-model channels -> final AI render
```

Current status is intentionally recorded as `needs_review`, not `pass`. The latest run generated 5 Hunyuan3D 2.1 high-profile module GLBs with no procedural fallback and reached a module presence score of `0.855333`, a multiview score of `0.789947`, and a minimum structure review score of `0.551932`. The concept/final comparison still fails `white_hero_presence`; the final image's central white subject ratio is about `0.000431`, far below the concept target. The next engineering targets are flatter-screen mesh sanity checks, concept-aligned camera search, and stricter final-render adherence to the Blender white-model position.

![Latest module references](docs/paper_assets/module_references_contact.png)

![Latest concept vs white model vs final](docs/paper_assets/concept_vs_final.png)

![Latest final contact sheet](docs/paper_assets/final_contact_sheet.png)

![Latest white-model hero view](docs/paper_assets/white_model_view_hero.png)

![Latest final hero view](docs/paper_assets/final_view_hero.png)

## What It Does

1. Render a mesh through Blender into `rgb`, `depth`, `edge`, `normal`, `mask`, and `skeleton` channels.
2. Generate AI candidates from model-derived channels only.
3. Score candidates with `structure_v2`, which combines silhouette IoU, edge chamfer alignment, added-part penalties, roughness, and background cleanliness.
4. Run the Agent across a locked anchor view and optional left/right views, then choose the best structure and multi-view consistency combination.
5. Produce inspectable artifacts such as `agent_report.json`, `final.png`, `white_vs_final.png`, and multi-view contact sheets.

## Quick Check

```bash
cd /root/sakura/work/Harmonize3D
.venv/bin/python -m pytest -q
```

CPU-safe smoke test:

```bash
PYTHONPATH=src .venv/bin/python -m local3dai.cli sample-renders \
  --output outputs/sample_renders \
  --views 3 \
  --resolution 128

PYTHONPATH=src .venv/bin/python -m local3dai.cli agent-render \
  --input-renders outputs/sample_renders/manifest.json \
  --output outputs/sample_agent \
  --prompt "premium red concept sports car, clean studio product render" \
  --backend mock \
  --max-generations 3
```

## Auto Agent Mode

Auto Agent Mode turns one natural-language request into the full controlled pipeline: task expansion, camera planning, mesh generation or loading, Blender channel rendering, AI candidate search, structure scoring, visual judgement, and final packaging.

CPU-safe smoke test:

```bash
PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-run \
  --request "生成一辆未来感白色电动跑车，做三张产品图" \
  --source-mode procedural \
  --views 3 \
  --quality fast \
  --geometry strict \
  --style product \
  --backend mock \
  --candidates 1 \
  --max-retries 0 \
  --dry-run \
  --output outputs/auto_smoke
```

The run writes `auto_task.json`, `prompt_plan.json`, `camera_plan.json`, `mesh/metadata.json`, `mesh/sanity.json`, `reports/tool_calls.json`, `reports/visual_judgement.json`, `reports/agent_report.json`, `reports/scores.json`, `final/final.png`, `final/white_vs_final.png`, and `final/contact_sheet.png`.

The planner uses an OpenAI-compatible Qwen service. The default configuration points to DashScope `qwen3.7-plus`; credentials and endpoint overrides are loaded from project `.env`:

```bash
DASHSCOPE_API_KEY=...
H3D_AGENT_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
H3D_AGENT_LLM_MODEL=qwen3.7-plus
```

Check planner connectivity:

```bash
PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-doctor
```

Optional local vLLM helpers for the previous NVFP4 planner model are still available:

```bash
scripts/download_qwen36_agent_model_direct.sh
scripts/start_qwen36_agent_service.sh
```

The local download helper defaults to `/root/sakura/models/Qwen3.6-35B-A3B-NVFP4`. The start helper mounts `/root/sakura/models` into the vLLM container and automatically uses that local directory when `config.json` exists; otherwise vLLM resolves the model ID through `HF_ENDPOINT=https://hf-mirror.com` with proxy variables removed.

For local-model wiring diagnostics:

```bash
PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-doctor --allow-not-ready
PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-doctor --check-hf-mirror --allow-not-ready
```

The agent exposes controlled tool execution through a fixed local tool list and records every call in `reports/tool_calls.json`. Visual judgement is a required packaging gate; it checks final pixels, comparison/contact-sheet images, structure scores, and multi-view consistency before declaring `complete`, `needs_review`, or `failed`.

## Auto Scene Mode

Auto Scene Mode extends the one-object flow into modular scene generation. It expands one request into `scene_plan.json`, `module_plan.json`, a global concept image, per-module reference images, module GLBs, `scene_assembly.json`, an assembled `final_scene.glb`, scene render channels, module scoring, and final geometry-locked AI outputs.

Mock end-to-end run:

```bash
PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene \
  --request "生成一个未来汽车发布会展台，中央是一辆白色低趴电动超跑，周围有蓝色灯带、黑色展示台、两块发光屏幕、机械臂和灰色反光地面，输出三张产品级渲染图。" \
  --output outputs/auto_scene/demo_showroom \
  --views 3 \
  --quality fast \
  --geometry strict \
  --style exhibition \
  --backend mock \
  --candidates 1 \
  --max-retries 0 \
  --dry-run \
  --no-llm
```

Blender render smoke run with mock AI output:

```bash
PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene \
  --request "生成一个未来汽车发布会展台，中央是一辆白色低趴电动超跑，周围有蓝色灯带、黑色展示台、两块发光屏幕、机械臂和灰色反光地面，输出三张产品级渲染图。" \
  --output outputs/auto_scene/demo_showroom \
  --views 3 \
  --quality fast \
  --geometry strict \
  --style exhibition \
  --backend mock \
  --candidates 1 \
  --max-retries 0 \
  --no-llm \
  --render-backend blender
```

Web/API entry:

```text
POST /api/auto-scene
GET  /api/auto-scene/{task_id}
```

The planner model is loaded from `.env` through the OpenAI-compatible Aliyun Bailian settings (`H3D_AGENT_LLM_BASE_URL`, `H3D_AGENT_LLM_MODEL`, `DASHSCOPE_API_KEY`). Mock mode can skip the LLM with `--no-llm`; real planner mode uses the configured model but still keeps concept and module reference images out of the final AI render inputs. If `concept/global_concept.png` or `modules/<module_id>/reference.png` already exists in the workdir, Auto Scene preserves that image2 asset and records it as `image2_provided` instead of replacing it with the fallback preview generator.

Typical output layout:

```text
outputs/auto_scene/<task>/
  auto_task.json
  scene_plan.json
  concept/concept_image_plan.json
  concept/global_concept.png
  modules/module_plan.json
  modules/module_asset_manifest.json
  modules/<module_id>/reference.png
  modules/<module_id>/model.glb
  scene/scene_assembly.json
  scene/final_scene.glb
  scene/scene_preview.png
  cameras/camera_plan.json
  renders/render_manifest.json
  final/final_view_hero.png
  final/final_view_left_30.png
  final/final_view_right_30.png
  final/contact_sheet.png
  reports/module_scores.json
  reports/agent_report.json
```

## Default Local Setup

The checked-in local config is tuned for this workstation:

- Blender: `tools/blender-5.1.2-linux-x64/blender`
- Default image backend: `flux2-klein`
- Default Agent model key: `flux2_klein_4b`
- Default score version: `structure_v2`
- Default Agent views: `view_locked`, `view_left_30`, `view_right_30`
- Default Agent reference channels: `rgb`, `edge`, `depth`, `normal`, `mask`, `skeleton`

Use `mock` for fast regression tests and UI checks. Use the real configured backends only when the local model weights and GPU environment are ready.

## Main CLI Commands

Render an existing mesh:

```bash
PYTHONPATH=src .venv/bin/python -m local3dai.cli render \
  --model examples/sample_model.obj \
  --output outputs/sample_model/renders \
  --views 3 \
  --resolution 512
```

Run Agent rendering from a render manifest:

```bash
PYTHONPATH=src .venv/bin/python -m local3dai.cli agent-render \
  --input-renders outputs/sample_model/renders/manifest.json \
  --output outputs/sample_model/agent \
  --prompt "premium graphite and white hypercar material, clean studio product render" \
  --backend mock \
  --max-generations 4
```

Run the full scripted workflow:

```bash
PYTHONPATH=src .venv/bin/python -m local3dai.cli run \
  --model examples/sample_model.obj \
  --prompt "photorealistic product render, crisp studio light" \
  --backend mock \
  --workdir outputs/workflow_smoke
```

## Web Workbench

Start the local web app:

```bash
PYTHONPATH=src .venv/bin/python -m local3dai.webapp --host 127.0.0.1 --port 7866
```

Open:

```text
http://127.0.0.1:7866
```

The workbench supports staged operation:

1. Load or generate a 3D white model.
2. Inspect the model in Three.js and lock the current camera.
3. Render Blender white-model channels for that locked view.
4. Generate the final AI render and comparison artifacts.

For an automated UI smoke test, run the server and then:

```bash
H3D_VALIDATE_AI_BACKEND=mock node scripts/validate_web_workbench.mjs
node scripts/validate_auto_agent_web.mjs
node scripts/validate_auto_agent_api.mjs
```

## Z-Image Research Path

The repo contains experimental Z-Image staged ControlNet scripts and reports. The local non-turbo Z-Image plus lite ControlNet path has been validated as loadable, but direct single-view outputs are not yet good enough to become the default Agent backend. Keep it as a candidate generator and keep direct-vs-final reporting enabled.

Useful entry points:

- `scripts/probe_zimage_staged_control.py`
- `scripts/probe_zimage_i2l_lora.py`
- `scripts/run_zimage_i2l_img2img_triview.py`
- `reports/zimage_meshlock_validation_20260614.md`
- `reports/dual_image_staged_weight_research_20260613.md`

## Project Layout

- `src/local3dai/agent.py`: MeshLock Agent search, scoring, multi-view selection, and report output.
- `src/local3dai/scoring_v2.py`: structure-adherence scoring.
- `src/local3dai/ai/backends.py`: mock, Diffusers, SDXL ControlNet, Flux2 Klein, and HiDream backend adapters.
- `src/local3dai/ai/geometry.py`: mesh position/detail/adaptive/quality lock compositing helpers.
- `src/local3dai/workflow.py`: reusable end-to-end workflow.
- `src/local3dai/webapp.py`: staged local API and web workbench server.
- `blender_scripts/batch_render.py`: Blender channel rendering.
- `web/`: static Three.js workbench UI.
- `reports/`: technical reports and validation notes.

## Reports

The current technical report artifacts live under `reports/`, including:

- `reports/Harmonize3D_Technical_Report.pdf`
- `reports/harmonize3d_technical_report_content.json`
- `reports/flux2_staged_weight_matrix_20260613.md`
- `reports/zimage_meshlock_validation_20260614.md`
