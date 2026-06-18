# Harmonize3D 论文框架草案

拟题：
Harmonize3D: 面向结构一致性的模块化 3D 场景生成与多视角 AI 渲染 Agent

本文定位：
系统型论文 / 工程型方法论文。重点不是提出一个新的图像生成大模型，而是提出一个围绕 3D 几何、显式渲染通道、工具执行、自动评分和多视角选择构建的可控场景生成 Agent。

图像说明：
本文档先给出论文大体结构和文字框架。所有图像、截图、最终渲染图、流程图和对照图后续补充。

## 摘要草稿

现有文本到图像或图像到图像模型能够生成高质量视觉结果，但在产品级 3D 工作流中常出现结构漂移、对象缺失、多视角不一致和过程不可审计等问题。Harmonize3D 面向这一问题，构建了一个本地运行的模块化 3D 场景生成与渲染 Agent。系统从一句自然语言请求出发，使用 OpenAI-compatible Qwen planner 进行任务理解和模块拆分，再生成场景计划、模块计划、参考图、模块 GLB、场景装配、相机计划和模型派生渲染通道。最终图像生成阶段不直接使用未声明参考图，而仅使用 `render_manifest` 中的 RGB、Edge、Depth、Normal、Mask 等结构通道约束本地图像后端。系统进一步引入 MeshLock-MV 流程，通过结构评分、多视角一致性评分和视觉判断报告，自动筛选并打包最终结果。

在一个未来汽车发布会展台任务中，系统生成了包含白色电动超跑、黑色展台、蓝色灯带、发光屏幕、机械臂和反光地面的三视角产品图。正式运行使用 `qwen3.7-plus` 作为规划模型、`flux2-klein` 作为本地图像生成后端，最终视觉判断状态为 `pass`，多视角一致性为 0.870，模块存在评分为 0.833。结果表明，该系统能够将自然语言请求转换为可审计的模块化 3D 场景生产流程，并在生成质量和结构可控性之间建立可操作的工程闭环。

## 1. 引言

### 1.1 背景

生成式 AI 正在改变 3D 内容生产流程。文本到图像模型可以快速给出视觉概念，图像到 3D 和文本到 3D 模型也逐渐能够产出可用的几何资产。但是在产品渲染、工业展示、虚拟发布会、展台设计等场景中，单纯依赖生成模型仍存在几个核心问题：

- 生成图像可能视觉上合理，但不严格遵守已有 3D 几何。
- 多视角输出容易出现车身比例、模块位置、背景元素不一致。
- 模块化场景需求无法稳定拆解为可复用资产。
- 输出过程缺少中间 manifest、评分报告和可追踪工具调用。
- 用户难以判断失败发生在规划、几何、渲染还是后处理阶段。

### 1.2 研究问题

本文关注的问题是：

如何将一句自然语言场景需求转化为一个可审计、可评分、可复现的模块化 3D 场景生成与多视角 AI 渲染流程，并尽量减少最终图像相对 3D 结构的漂移？

### 1.3 项目目标

Harmonize3D 的目标不是替代底层大模型，而是在底层模型之外增加一个结构控制层：

- 使用 Agent 完成需求理解、任务拆分和工具编排。
- 将复杂场景显式拆分成模块。
- 使用 3D 场景和渲染通道作为最终生成的结构依据。
- 对结果执行自动结构评分、多视角一致性判断和可视化报告。
- 生成完整的工程产物，包括 GLB、渲染图、contact sheet、manifest 和日志。

### 1.4 主要贡献

本文可以总结为四点贡献：

1. 提出一个模块化场景生成 Agent 工作流，将自然语言请求拆解为场景计划、模块计划、资产生成、场景装配、通道渲染、AI 渲染和报告打包。
2. 提出 `render_manifest` 约束策略，最终 AI 渲染只使用由场景几何派生的结构通道，避免直接绕过 3D 结构使用未声明参考图。
3. 实现 MeshLock-MV 多视角结构控制与选择流程，结合轮廓 IoU、边缘 Chamfer、额外部件惩罚、粗糙度、背景洁净度和多视角一致性评分。
4. 给出一个可运行的本地产品化原型，支持 CLI、Web/API、工具调用记录、视觉判断报告和完整输出目录。

## 2. 相关工作

本章后续需要补充正式引用。当前先保留结构。

### 2.1 文本到图像与图像到图像生成

讨论扩散模型、图像先验、ControlNet、参考图约束和高质量产品渲染能力。重点指出纯图像模型在结构一致性方面的不足。

待补引用：
- Diffusion models
- Stable Diffusion / SDXL
- FLUX 系列或类似高质量图像生成模型
- ControlNet / T2I-Adapter 等结构控制方法

### 2.2 文本到 3D 与图像到 3D

讨论从文本或图片生成 3D 资产的工作，包括 NeRF、3D Gaussian、mesh generation、多视角重建等。指出单个对象生成和复杂场景装配之间仍有工程断层。

待补引用：
- DreamFusion 类文本到 3D 方法
- Point-E / Shap-E / Hunyuan3D 等资产生成路线
- 多视角重建和 image-to-3D 相关工作

### 2.3 多视角一致性与结构保持

讨论多视角图像生成中常见的一致性问题。Harmonize3D 的思路是将一致性从纯图像域转移到 3D 渲染通道和评分选择上。

### 2.4 Agentic Workflow 与工具执行

讨论 LLM Agent 在复杂内容生产中的任务规划、工具编排和日志记录。Harmonize3D 将 Agent 限定在固定工具集合中运行，降低不可控性，并通过 manifest 保留每个阶段的证据。

## 3. 系统概览

### 3.1 输入与输出

输入：
- 一句自然语言场景请求。
- 输出视角数量、质量模式、几何严格程度、风格 preset 等参数。
- 可选后端配置，例如 planner 模型、图像生成后端和模型 key。

输出：
- `auto_task.json`
- `scene_plan.json`
- `module_plan.json`
- 模块参考图和模块 GLB
- `scene_assembly.json`
- `final_scene.glb`
- `render_manifest.json`
- 三视角最终图和 contact sheet
- `agent_report.json`
- `visual_judgement.json`
- `tool_calls.json`

### 3.2 总体流程

建议图 1：
Harmonize3D Auto Scene Pipeline 总览图。

图中包含以下阶段：

1. 用户输入自然语言请求。
2. Qwen planner 进行任务理解和结构化规划。
3. Rule planner / expander 生成保底计划。
4. 模块拆分，得到 hero object、supporting object、background object 等。
5. 生成概念图和模块参考图。
6. 生成模块 3D 资产或程序化 GLB。
7. 场景布局与装配，生成 `final_scene.glb`。
8. 从场景生成 RGB、Edge、Mask、Depth、Normal 通道。
9. 使用本地图像后端生成候选图。
10. MeshLock-MV 进行结构锁定、多视角选择和评分。
11. 视觉判断和最终打包。

### 3.3 设计原则

- 显式结构优先：最终图像生成必须以 3D 派生通道作为结构依据。
- 过程可审计：每个阶段写入 manifest、日志、报告和 artifact 路径。
- 模型可替换：planner 和 image backend 都通过配置切换。
- 测试可运行：mock backend 必须可完整跑通，不依赖真实大模型。
- 产品化输出：结果不仅是单张图，还包括 GLB、三视角、contact sheet、报告和 Web/API 状态。

## 4. 方法

### 4.1 自然语言规划

系统首先接收一句场景请求，例如：

生成一个未来汽车发布会展台，中央是一辆白色低趴电动超跑，周围有蓝色灯带、黑色展示台、两块发光屏幕、机械臂和灰色反光地面，输出三张产品级渲染图。

planner 输出包括：
- `auto_task`
- `scene_plan`
- `concept_image_plan`
- `module_plan`
- `prompt_plan`
- `camera_plan`

正式运行中，planner 使用：
- Provider: OpenAI-compatible
- Model: `qwen3.7-plus`
- Endpoint: DashScope compatible mode

### 4.2 模块化场景拆分

场景被拆分为若干模块，每个模块包含：
- `module_id`
- `name`
- `category`
- `role`
- `priority`
- `expected_real_world_size`
- `placement`
- `reference_prompt`

正式运行中的模块包括：

| 模块 | 类型 | 角色 | 优先级 |
| --- | --- | --- | --- |
| `main_vehicle` | vehicle | hero_object | 1 |
| `display_platform` | stage_prop | supporting_object | 2 |
| `left_led_screen` | background_prop | background | 3 |
| `right_led_screen` | background_prop | background | 3 |
| `robotic_arm_left` | stage_prop | supporting_object | 4 |
| `robotic_arm_right` | stage_prop | supporting_object | 4 |
| `blue_light_strips` | decorative_object | background | 5 |
| `reflective_floor` | environment | background | 6 |

这一拆分使系统可以对复杂场景进行模块级资产生成、布局、失败处理和存在性评分。

### 4.3 概念图与模块参考图

系统会生成：
- 全局概念图：用于构图和整体风格预览。
- 模块参考图：用于模块级 3D 资产生成或程序化代理。

重要约束：
概念图和模块参考图只作为规划或资产阶段输入，不直接作为最终 AI 渲染参考。最终渲染参考来自 `render_manifest` 中的场景通道。

### 4.4 模块资产生成与失败策略

每个模块会生成或绑定一个 GLB 资产，并写入：
- `metadata.json`
- `sanity.json`
- `reference_manifest.json`

失败策略：
- hero object 失败时任务应失败。
- 平台、屏幕、灯带等非核心模块可使用程序化 fallback。
- 背景或装饰模块可按策略跳过或降级。

当前原型中，模块 3D 资产可以通过 mock/procedural GLB 跑通完整链路。后续可替换为真实 image-to-3D 后端。

### 4.5 场景布局与装配

layout agent 根据模块 role、priority、anchor 和 expected size 输出：
- position
- rotation
- scale
- layout reason
- collision report

最终生成：
- `scene_assembly.json`
- `scene/final_scene.glb`
- `scene/scene_preview.png`

### 4.6 场景渲染通道

系统从装配后的场景生成多视角结构通道：
- RGB white/render reference
- Edge
- Mask
- Depth
- Normal

这些通道写入 `render_manifest.json`。正式运行中，渲染源标记为：

```text
auto_scene_procedural_scene_channels
```

后续真实产品化版本可将该阶段替换为 Blender 对完整场景的真实通道渲染。

### 4.7 最终 AI 渲染

最终 AI 渲染阶段使用本地图像后端：

```text
backend: flux2-klein
model_key: flux2_klein_4b
```

输入策略：

```text
final_ai_inputs: render_manifest.rgb, render_manifest.edge
planning_only_images: concept/global_concept.png, modules/*/reference.png
```

这保证最终图像不是直接从概念图或模块参考图生成，而是受场景结构通道约束。

### 4.8 MeshLock-MV 结构控制

MeshLock-MV 包含三部分：

1. 单视角结构评分。
2. 多视角一致性选择。
3. 视觉判断 gate。

单视角评分指标：
- silhouette IoU
- edge chamfer score
- added-part penalty
- roughness
- background cleanliness

多视角指标：
- body color consistency
- mask/edge feature consistency
- mean structure
- selection score

严格几何模式下，系统启用 mesh-position lock，将 AI 生成结果裁回场景白模结构，提升结构保持与自动评分稳定性。

## 5. 实现

### 5.1 代码模块

主要实现文件：

| 文件 | 作用 |
| --- | --- |
| `src/local3dai/auto_scene.py` | Auto Scene 主流程、模块计划、场景装配、通道生成、打包 |
| `src/local3dai/agent.py` | MeshLock Agent、候选搜索、评分、多视角选择 |
| `src/local3dai/ai/backends.py` | 图像生成后端，包括 mock 和 `flux2-klein` |
| `src/local3dai/ai/geometry.py` | 几何锁定、位置锁定、质量锁定后处理 |
| `src/local3dai/scoring_v2.py` | 结构评分 |
| `src/local3dai/webapp.py` | Web/API 入口 |
| `src/local3dai/cli.py` | CLI 入口 |

### 5.2 CLI 与 API

CLI：

```bash
PYTHONPATH=src .venv/bin/python -m local3dai.cli auto-scene \
  --request "..." \
  --output outputs/auto_scene/demo_showroom \
  --views 3 \
  --quality fast \
  --geometry strict \
  --style exhibition \
  --candidates 1 \
  --max-retries 0
```

API：

```text
POST /api/auto-scene
GET  /api/auto-scene/{task_id}
```

### 5.3 Web 工作台

Web 工作台展示每个阶段：
- status
- artifact
- message
- retry_count
- warnings
- error

建议图 2：
Web Auto Scene Mode 截图，展示阶段列表、最终图和报告链接。

### 5.4 可复现性与配置

系统通过 `.env` 读取 planner 配置，通过 `configs/local.json` 读取本地模型、GPU、Blender 和后端配置。真实密钥不应写入论文正文，只描述环境变量名称和配置方式。

## 6. 实验设置

### 6.1 实验任务

正式测试任务：

生成一个未来汽车发布会展台，中央是一辆白色低趴电动超跑，周围有蓝色灯带、黑色展示台、两块发光屏幕、机械臂和灰色反光地面，输出三张产品级渲染图。

### 6.2 运行环境

当前实验证据来自正式运行目录：

```text
/root/sakura/work/Harmonize3D/outputs/auto_scene/formal_showroom_meshlock_20260616_003228
```

关键环境：
- GPU: NVIDIA GeForce RTX 5090 D
- Planner: `qwen3.7-plus`
- Image backend: `flux2-klein`
- Model key: `flux2_klein_4b`
- Output views: 3
- Geometry mode: strict

### 6.3 评价指标

本文使用以下评价维度：

| 维度 | 指标 | 含义 |
| --- | --- | --- |
| 结构保持 | structure score | 输出图与渲染通道的轮廓和边缘匹配 |
| 多视角一致性 | multiview total | 三视角之间的颜色和结构特征一致性 |
| 模块存在性 | module presence score | 计划模块是否在场景中保留 |
| 图像有效性 | nonblank / dynamic range | 输出图是否为空、是否有足够动态范围 |
| 工程可审计性 | tool calls / manifests | 是否记录每阶段工具调用和 artifact |

## 7. 实验结果

### 7.1 正式运行结果

| 项目 | 结果 |
| --- | --- |
| 总状态 | `complete` |
| Planner backend | `qwen_openai_compatible` |
| Planner model | `qwen3.7-plus` |
| Image backend | `flux2-klein` |
| Model key | `flux2_klein_4b` |
| 生成模块数 | 8 |
| 输出视角数 | 3 |
| Visual judgement | `pass` |
| Structure minimum total | 0.699084 |
| Selected score | 0.718201 |
| Multiview total | 0.870261 |
| Module score total | 0.833333 |
| Tool calls | 11 |
| Total elapsed seconds | 130.725 |

### 7.2 定性结果

建议图 3：
最终三视角 contact sheet。

待补图像：

```text
outputs/auto_scene/formal_showroom_meshlock_20260616_003228/final/contact_sheet.png
```

建议图 4：
White reference vs final render 对照图。

待补图像：

```text
outputs/auto_scene/formal_showroom_meshlock_20260616_003228/final/white_vs_final.png
```

建议图 5：
场景通道示例，包括 RGB、Edge、Mask、Depth、Normal。

待补图像：

```text
outputs/auto_scene/formal_showroom_meshlock_20260616_003228/renders/view_hero/
```

### 7.3 工程验证

当前验证结果：

```text
49 passed
py_compile passed
node --check passed
git diff --check passed
```

这说明 mock 路径、Auto Scene 流程、Agent 渲染和前端脚本均通过当前测试集合。

### 7.4 结果分析

结果显示系统能够：

- 从一句话生成完整场景规划。
- 自动拆分出车、展台、屏幕、机械臂、灯带和地面模块。
- 生成可检查的 `final_scene.glb` 和 `scene_assembly.json`。
- 生成三视角最终图和 contact sheet。
- 通过视觉判断 gate。
- 保留工具调用和评分报告。

同时也观察到一个重要 trade-off：
非锁定 FLUX 输出更写实，但容易偏离场景结构；strict MeshLock 输出更受控，结构评分和多视角一致性更稳定。论文中可以将这一点作为方法动机和消融实验方向。

## 8. 消融实验设计

当前尚未系统完成消融，但论文可以设计如下实验：

### 8.1 无 MeshLock vs MeshLock

对比：
- 直接 `flux2-klein` 输出。
- strict MeshLock 输出。

指标：
- structure minimum total
- edge chamfer score
- added-part penalty
- multiview total
- 人工视觉偏好

### 8.2 单视角选择 vs 多视角选择

对比：
- 只生成 hero view。
- 同时生成 hero、left、right 三视角，并进行组合选择。

### 8.3 模块化场景 vs 单对象流程

对比：
- 只生成一辆车。
- 生成完整发布会展台。

评估模块完整性、场景可解释性和最终输出丰富度。

### 8.4 Planner 关闭 vs Planner 打开

对比：
- 只使用规则 planner。
- 使用 `qwen3.7-plus` planner + 规则 planner merge。

评估 module plan 的语义完整性和稳定性。

## 9. 讨论

### 9.1 优势

- 可控：最终图像由 3D 通道约束，不是纯文本自由生成。
- 可审计：每阶段都有 manifest 和报告。
- 可替换：planner、image backend、3D backend 可替换。
- 可产品化：已具备 CLI、API、Web 工作台和完整输出目录。
- 可测试：mock backend 支持完整回归测试。

### 9.2 局限

当前系统仍有明显限制：

- 模块 3D 资产目前可使用程序化代理，真实 image-to-3D 后端仍需增强。
- 场景通道阶段当前可以是程序化渲染，后续需要替换为完整 Blender scene render。
- 视觉评分主要基于 CV 结构指标，无法完全表达语义质量。
- strict MeshLock 会提升结构稳定性，但可能降低自由生成模型的写实细节。
- planner 偶尔可能返回非 JSON，需要更强 schema 约束或重试策略。

### 9.3 产品化价值

Harmonize3D 更像一个生成式 3D 内容生产控制台，而不是单一模型 demo。它可以服务于：

- 电商产品渲染。
- 汽车和工业产品展示。
- 展会展台概念设计。
- 3D 资产快速预览。
- 多视角一致内容生成。

## 10. 未来工作

后续可以从以下方向扩展：

1. 使用 Blender 真实装配完整模块场景，并输出更准确的 RGB、Edge、Mask、Depth、Normal。
2. 接入真实 image-to-3D 后端，替换程序化模块 GLB。
3. 引入模块级材质记忆，保持同一模块跨视角和跨任务外观一致。
4. 增强 occlusion、visibility buffer、face id 和实例 mask。
5. 加入用户确认 concept image 后再继续的 human-in-the-loop 流程。
6. 增加 VLM 语义评分，判断屏幕、机械臂、灯带等元素是否真实出现。
7. 构建标准 benchmark，覆盖单对象、室内、展台、产品组合等任务。
8. 支持更高质量最终渲染后端和局部修复。

## 11. 结论

本文介绍了 Harmonize3D，一个面向结构一致性和工程可审计性的模块化 3D 场景生成 Agent。系统通过自然语言规划、模块拆分、场景装配、几何通道渲染、AI 候选生成、MeshLock-MV 多视角选择和视觉判断报告，将一句话请求转换为完整的产品级输出包。正式运行结果表明，该系统能够在本地 GPU 环境中生成三视角展台产品图，并通过自动视觉判断 gate。后续工作将进一步提高真实 3D 资产质量、语义评分能力和 Blender 场景渲染精度。

## 附录 A：建议图表清单

| 编号 | 图表 | 状态 |
| --- | --- | --- |
| Figure 1 | Auto Scene 总体流程图 | 待补 |
| Figure 2 | Web 工作台阶段展示 | 待补 |
| Figure 3 | 正式运行三视角 contact sheet | 待补 |
| Figure 4 | White reference vs final render | 待补 |
| Figure 5 | RGB / Edge / Mask / Depth / Normal 通道示例 | 待补 |
| Figure 6 | 模块分解和 scene assembly 可视化 | 待补 |
| Table 1 | 模块列表与角色 | 已有草表 |
| Table 2 | 正式运行指标 | 已有草表 |
| Table 3 | 消融实验结果 | 待补 |

## 附录 B：正式运行产物路径

```text
/root/sakura/work/Harmonize3D/outputs/auto_scene/formal_showroom_meshlock_20260616_003228
```

关键文件：

```text
auto_scene_summary.json
auto_task.json
scene_plan.json
modules/module_plan.json
scene/final_scene.glb
scene/scene_assembly.json
renders/render_manifest.json
final/final_view_hero.png
final/final_view_left_30.png
final/final_view_right_30.png
final/contact_sheet.png
final/white_vs_final.png
reports/agent_report.json
reports/module_scores.json
reports/multiview_score.json
reports/visual_judgement.json
reports/tool_calls.json
```

## 附录 C：当前可引用的验证结果

```text
Auto Scene formal status: complete
Visual judgement: pass
Planner model: qwen3.7-plus
Image backend: flux2-klein
Module count: 8
Output views: 3
Structure minimum total: 0.699084
Selected score: 0.718201
Multiview total: 0.870261
Module score total: 0.833333
Tool calls: 11
Tests: 49 passed
```
