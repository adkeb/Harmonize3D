# Flux2 Klein 分阶段权重生图测试报告

日期：2026-06-13  
对象：车模型白模 `outputs/flux2_klein_high_quality_car_reference/white_renders/manifest.json`  
模型：`flux2_klein_4b / Flux2KleinPipeline`，本轮强制 `HF_HUB_OFFLINE=1`

## 结论

Flux2 Klein 当前不支持真正的“双图分阶段权重”。

它可以接收多张参考图，但 `Flux2KleinPipeline.__call__` 和 callback 暴露的接口没有 per-reference image embedding、per-reference weight、ControlNet scale、IP-Adapter scale，也没有可在 denoise step 中分别调节“结构图权重”和“外观图权重”的入口。因此本轮只能测试两类近似方案：

- 单次调用：把结构通道和外观参考图按不同顺序一起传入。
- 两次调用：第一阶段先用结构通道出结构图，第二阶段把第一阶段结果作为 `rgb` 参考，再加入外观参考图。

测试结果表明：两次调用 + position lock 能把轮廓分数拉高，但视觉上出现明显裁切/贴片感，不应判定为可用主方案。直接同时输入结构图和外观图能获得红色材质，但结构变形、新增部件和视角漂移仍然明显。

## 实验输出

- Smoke：`outputs/flux2_klein_high_quality_car_reference/flux2_staged_weight_matrix_smoke/`
- 三视角低步数矩阵：`outputs/flux2_klein_high_quality_car_reference/flux2_staged_weight_matrix_3views_512/`
- 三视角高步数关键组合：`outputs/flux2_klein_high_quality_car_reference/flux2_staged_weight_key_3views_768/`
- 探针脚本：`scripts/probe_flux2_staged_weight_matrix.py`

## 关键指标

### 512px / 3+3 / 6 steps / 三视角

| Recipe | 平均结构分 | 最低轮廓 IoU | 新增部件惩罚 | Edge leak | 红色指标 | 失败数 |
|---|---:|---:|---:|---:|---:|---:|
| A_structure_full | 0.8936 | 0.9827 | 0.0000 | 0.2739 | 0.0002 | 0/3 |
| C_simultaneous_full_after | 0.4778 | 0.3054 | 0.5156 | 0.2234 | 0.2041 | 3/3 |
| E_simultaneous_light_after | 0.4080 | 0.3210 | 0.5574 | 0.2137 | 0.1735 | 3/3 |
| G_two_stage_full_position_lock | 0.8304 | 0.9390 | 0.0000 | 0.1779 | 0.2627 | 3/3 |

解释：

- `A_structure_full` 是结构最佳，但几乎全是白模/灰模，外观失败。
- `C/E` 获得红色材质，但轮廓和新增部件失败，说明外观参考压过结构约束。
- `G` 数值接近可用，但 contact sheet 显示明显局部裁切和贴片，属于后处理把 mask 拉回来的伪高分。

### 768px / 8+8 / 16 steps / 三视角关键组合

| Recipe | 平均结构分 | 最低轮廓 IoU | 新增部件惩罚 | Edge leak | 红色覆盖 | Clay-like 比例 | 失败数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_structure_full | 0.8747 | 0.9890 | 0.0000 | 0.2102 | 0.0000 | 0.9598 | 1/3 |
| C_simultaneous_full_after | 0.3972 | 0.2718 | 0.5093 | 0.1856 | 0.6058 | 0.0571 | 3/3 |
| G_two_stage_full_position_lock | 0.7824 | 0.8751 | 0.0000 | 0.1535 | 0.5711 | 0.0704 | 3/3 |

高步数没有自然修复结构问题：

- 同时双图的红色覆盖提升明显，但结构分更低，右视角尤其崩。
- 两段 + position lock 的 clay-like 比例低，说明外观进来了，但右视角仍有明显裁切，edge chamfer 全部失败。
- 因此问题不是步数太少，而是 Flux2 Klein 缺少真正的结构控制/分阶段权重控制。

## 对换模型的判断

可以考虑换模型，但理由不是“Flux2 Klein 老”，而是它的接口不满足这个任务：

- 需要模型或后端能把结构作为内部 control conditioning，而不是仅作为普通参考图。
- 需要可在 denoise 过程中调度结构权重，例如 early high structure / late low structure。
- 需要可独立调度外观参考权重，例如 early low appearance / late high appearance。
- 如果没有 per-step 权重，至少也要有 ControlNet scale 或 adapter scale 可以在 step callback 中改变。

下一步更合理的方向是：用小体积、近半年内、支持 ControlNet/adapter 的模型做同样的 staged schedule 测试；在验证前不应该把它接入默认 Agent。

### 候选模型核对

本轮只做候选判断，不继续下载。

| 候选 | 判断 |
|---|---|
| [Z-Image-Fun-Controlnet-Union-2.1](https://huggingface.co/alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1) | 优先。Apache-2.0；支持 Canny/Depth/Pose/MLSD/Scribble/HED/Gray；lite 版给低规格机器用，能调 `control_context_scale`。更符合“结构进入内部 control conditioning”的方向。 |
| [Z-Image-Turbo-Fun-Controlnet-Union-2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | 可做 8-step 快速实验。2026-01/02 更新包含 lite、Scribble/Gray、多分辨率控制图，并明确提到改善 mask 信息泄漏；但 Turbo 基座体积偏大，需谨慎。 |
| [FLUX.2-dev-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union) | 次选。支持多种 control 和 `controlnet_conditioning_scale`，但许可证是 Flux dev 非商业许可，并且面向 FLUX.2-dev，不适合直接套到本地 Flux2 Klein。 |

因此，如果继续换模型测试，我建议先从 Z-Image lite ControlNet 路线做“小图、单视角、单 control channel、step callback scale schedule”烟测，而不是继续尝试 Flux2 Klein 多参考图顺序调参。

## 验收状态

本轮 Flux2 Klein 分阶段权重测试状态：`needs_model_or_backend_change`。

当前可保留 Flux2 Klein 作为普通多参考图实验后端，但不应把它作为 MeshLock-MV 的长期结构锁定后端。
