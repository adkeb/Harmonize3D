# 本地 3D 到 AI 图像渲染流水线

这个工程按 `计划.md` 搭好了一个本地脚本化系统：

1. Blender 批量渲染 3D 模型的 `RGB / Depth / Edge / Normal / Mask` 通道。
2. AI 后端读取通道图生成候选图；当前默认使用 SDXL Base + Canny/Depth ControlNet，并用白模 `Mask/Edge/RGB` 做几何锁定。
3. 一致性评分器按结构边缘和遮罩相似度排序，并复制最佳结果。
4. `doctor` 自检会读取本机 GPU、Blender、Python 包和本地模型状态。

当前机器已配置 RTX 5090 D 32GB、WSL2 Ubuntu 22.04、`uv`、官方 Blender 5.1.2。图像生成模型权重尚未在工程目录中发现，因此默认后端是 `mock`，能离线跑通全流程；等你放好 FLUX/SD 权重后，把 `configs/local.json` 里的 `local_path` 和 `ai.default_backend` 改成真实后端即可。

## 快速开始

```bash
cd /root/sakura/work/build
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

本工程带了一个本地前端，能从浏览器启动和监控全流程：选择来源模型或上传参考图，运行 Hunyuan3D 2.1 白模生成、Blender 通道渲染、SDXL ControlNet 约束渲染、结构评分，并展示白模/最终图/对照图。

```bash
cd /root/sakura/work/build
source .venv/bin/activate
local3dai-web --host 127.0.0.1 --port 7866
```

然后打开：

```text
http://127.0.0.1:7866
```

## 目录

- `src/local3dai/cli.py`：统一 CLI。
- `src/local3dai/webapp.py`：本地 Web 控制台和任务 API。
- `src/local3dai/workflow.py`：Hunyuan3D/Blender/AI 渲染/评分的可复用工作流。
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
