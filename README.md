# 本地 3D 到 AI 图像渲染流水线

这个工程按 `计划.md` 搭好了一个本地脚本化系统：

1. Blender 批量渲染 3D 模型的 `RGB / Depth / Edge / Normal / Mask` 通道。
2. AI 后端读取通道图生成候选图；当前默认使用 SDXL Base + Canny/Depth ControlNet，并用白模 `Mask/Edge/RGB` 做几何锁定。
3. 一致性评分器按结构边缘和遮罩相似度排序，并复制最佳结果。
4. `doctor` 自检会读取本机 GPU、Blender、Python 包和本地模型状态。

当前机器已配置 RTX 5090 D 32GB、WSL2 Ubuntu 22.04、`uv`、官方 Blender 5.1.2，并在当前工程目录下保留了 Hunyuan3D 2.1、HiDream-O1-Image、SDXL/ControlNet 等本地权重路径。`mock` 后端仍可用于无 GPU 的快速回归验证。

## 快速开始

```bash
cd /root/sakura/work/Harmonize3D
bash scripts/bootstrap_core.sh
source .venv/bin/activate
local3dai doctor --scan-models
local3dai run --prompt "cyberpunk fox figurine, glossy enamel, studio product lighting" --backend mock
```

不创建虚拟环境也可以临时这样跑：

```bash
PYTHONPATH=src python3 -m local3dai.cli run \
  --prompt "cyberpunk fox figurine, glossy enamel, studio product lighting" \
  --backend mock
```

## 使用真实 3D 模型

官方 Blender 已安装在 `tools/blender-5.1.2-linux-x64/blender`，配置文件也已指向它。以后如果要重装或升级，可运行：

```bash
bash scripts/install_blender_official.sh
```

可以先生成或接入一个 3D 模型：

```bash
local3dai generate-3d \
  --prompt "cyberpunk fox figurine" \
  --backend sample \
  --output outputs/generated_models/fox.obj
```

要接 Hunyuan3D、TRELLIS 等本地工具，把 `configs/local.json` 的 `model_generation.external_command` 设成你本机命令模板，模板里可用 `{prompt}` 和 `{output}`，然后运行 `--backend external`。

把 `.glb/.gltf/.obj` 放到 `models/` 或任意本地路径：

```bash
local3dai render --model models/my_model.glb --output outputs/my_model/renders
local3dai ai-render \
  --input-renders outputs/my_model/renders/manifest.json \
  --output outputs/my_model/candidates \
  --prompt "photorealistic product render, crisp studio light" \
  --backend mock
local3dai score \
  --input-renders outputs/my_model/renders/manifest.json \
  --input-candidates outputs/my_model/candidates/manifest.json \
  --output outputs/my_model/score
```

## Agent 自动调优渲染

Agent v1 会在已有白模通道上渐进式调优提示词、参考通道、seed 和 guidance。默认只使用 `rgb + edge` 作为 HiDream-O1-Image 参考，禁用容易放大噪点的 `depth/normal`，先跑单视图，达标后再扩展到 3 视图一致性检查。

```bash
local3dai agent-render \
  --input-renders outputs/my_model/renders/manifest.json \
  --output outputs/my_model/agent \
  --prompt "premium red concept sports car, clean studio product render" \
  --model-key hidream_o1_image_full \
  --target-view view_05 \
  --max-generations 10
```

CPU/快速测试可使用 mock：

```bash
local3dai sample-renders --output outputs/sample_renders --views 4 --resolution 128
local3dai agent-render \
  --input-renders outputs/sample_renders/manifest.json \
  --output outputs/sample_agent \
  --prompt "premium red concept sports car, clean studio product render" \
  --backend mock \
  --max-generations 4
```

Agent 输出包括 `agent_report.json`、`agent_trials/`、`final.png`、`white_vs_final.png`，通过 3 视图检查时还会生成 `three_view_contact.png`。

或一条命令：

```bash
local3dai run \
  --model models/my_model.glb \
  --prompt "photorealistic product render, crisp studio light" \
  --backend mock
```

工程内带了一个验证用模型：

```bash
local3dai run \
  --generate-model \
  --prompt "premium faceted figurine, clean studio product render" \
  --backend mock \
  --workdir outputs/full_model_check
```

## 接入 FLUX / SD 本地权重

先安装核心环境，再安装 CUDA 13.0 PyTorch wheel 和 Diffusers：

```bash
bash scripts/bootstrap_core.sh
bash scripts/install_ai_cuda130.sh
```

然后编辑 `configs/local.json`：

```json
{
  "ai": {
    "default_backend": "diffusers-flux"
  },
  "models": {
    "flux_schnell": {
      "local_path": "/absolute/path/to/FLUX.1-schnell"
    }
  }
}
```

运行：

```bash
local3dai run \
  --model models/my_model.glb \
  --prompt "premium studio product image, sharp focus, consistent geometry" \
  --backend sdxl-controlnet-geometry \
  --model-key sdxl_controlnet_geometry \
  --geometry-lock
```

## Web 控制台

本工程带了一个本地分阶段工作台，不再只是启动一条黑盒流水线。用户可以先用提示词、参考图或已有模型载入 3D 白模，在浏览器中直接拖转、缩放、平移模型，固定当前视角，然后后端按这个视角渲染 `RGB / Depth / Edge / Normal / Mask` 白模通道，最后根据用户输入的最终提示词生成白模/成图对照。

```bash
cd /root/sakura/work/Harmonize3D
source .venv/bin/activate
local3dai-web --host 127.0.0.1 --port 7866
```

然后打开：

```text
http://127.0.0.1:7866
```

工作台主路径：

1. `生成或载入 3D 白模`：支持提示词生成 3D、图片生成 3D、已有模型路径和程序化测试模型。
2. `固定 3D 视角`：Three.js 本地 viewer 支持拖转、缩放、平移、前/侧/后/45 度视角和固定当前相机。
3. `渲染白模通道`：调用 Blender 的 `view_locked` 单视角通道渲染。
4. `AI 渲染成图`：使用固定视角白模 manifest 生成最终图、对照图和 Agent report。

对应阶段 API：

- `POST /api/stage/3d`
- `POST /api/stage/white-render`
- `POST /api/stage/ai-render`

旧的 `/api/run` 和 CLI 一键流程仍保留，用于脚本化或回归验证。

## 目录

- `src/local3dai/cli.py`：统一 CLI。
- `src/local3dai/webapp.py`：本地 Web 控制台和任务 API。
- `src/local3dai/workflow.py`：Hunyuan3D/Blender/AI 渲染/评分的可复用工作流。
- `src/local3dai/agent.py`：Agent v1 自动调优策略、图像质量指标和报告输出。
- `blender_scripts/batch_render.py`：Blender 后台多视角多通道渲染。
- `src/local3dai/ai/backends.py`：`mock`、普通 Diffusers、SDXL ControlNet 几何约束图像后端。
- `src/local3dai/ai/geometry.py`：白模几何锁定和对照图工具。
- `src/local3dai/scoring.py`：结构一致性评分和最佳图复制。
- `configs/local.json`：按本机 RTX 5090 D/WSL2 生成的默认配置。

## 产物

每次 `run` 会在 `outputs/run-YYYYMMDD-HHMMSS/` 下生成：

- `renders/manifest.json`：每个视角的通道图索引。
- `candidates/manifest.json`：AI 候选图索引。
- `score/report.json`：候选排序分数。
- `score/ranked/`：复制出的最佳图。
