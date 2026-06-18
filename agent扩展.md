你是 Harmonize3D 项目的资深 AI Agent 工程师、3D 场景生成架构师和产品化开发专家。

请基于当前 Harmonize3D 仓库继续开发，把系统从“单个对象的一句话 3D 渲染 Agent”升级为“模块化场景生成与渲染 Agent”。

当前项目已经具备：
1. Hunyuan3D 2.1 生成或导入 GLB 白模；
2. Three.js 前端真实加载 3D 模型并支持用户交互定帧；
3. 后端将 Three.js Y-up 相机转换为 Blender Z-up；
4. Blender 固定视角输出 rgb / depth / edge / normal / mask；
5. AI 渲染阶段默认只使用模型渲染通道，尤其是 rgb + edge；
6. Agent 能记录 prompt、seed、reference_channels、评分和 agent_report.json；
7. Web 工作台已经是分阶段流程，而不是单一静态页面。

当前项目原则必须继续保留：
最终 AI 渲染阶段不能直接绕过 3D 模型和 Blender 通道。最终图像模型必须主要依赖 render_manifest 中由真实 3D 模型渲染出的 rgb / edge / mask / depth / normal 等通道。概念图和模块参考图可以用于前期 3D 生成和风格规划，但必须在 manifest 中清晰记录，不能偷偷作为最终 AI 渲染的未声明参考。

==================================================
一、新目标：模块化场景生成 Agent
==================================================

用户只输入一句自然语言需求，例如：

“生成一个未来汽车发布会展台，中央是一辆白色低趴电动超跑，周围有蓝色灯带、黑色展示台、两块发光屏幕、机械臂和灰色反光地面，输出三张产品级渲染图。”

系统需要自动完成：

1. 需求理解；
2. 需求扩写；
3. 场景元素识别；
4. 总体概念图生成；
5. 场景模块拆分；
6. 每个模块生成独立参考图；
7. 每个模块 image-to-3D 建模；
8. 每个模块 mesh 质量检查；
9. 每个模块缩放、旋转、位置摆放；
10. 合成完整 3D 场景；
11. 自动相机规划；
12. Blender 输出多视角白模通道；
13. AI 图像渲染；
14. 多候选评分；
15. 自动重试；
16. 多视角一致性检查；
17. 输出最终图、对比图、场景报告和完整参数记录。

核心产品体验：
用户只说一句话，系统自动把复杂场景拆成多个可控模块，再组合成一个可渲染的 3D 场景。

==================================================
二、新增核心流程
==================================================

请实现如下完整流程：

Natural Language Request
  ↓
Multimodal Requirement Expander
  ↓
Scene Plan
  ↓
Global Concept Image Generation
  ↓
Module Decomposition
  ↓
Per-Module Reference Image Generation
  ↓
Per-Module Image-to-3D
  ↓
Mesh Sanity Check
  ↓
Auto Layout / Scale / Placement
  ↓
Scene Assembly
  ↓
Camera Planning
  ↓
Blender White Channel Rendering
  ↓
AI Final Rendering
  ↓
Scoring / Retry / Report

其中：

1. Global Concept Image
   用于确定整体构图、风格、色彩、主次关系和场景氛围。

2. Module Reference Images
   用于每个独立模块的 image-to-3D 建模，例如：
   - main_vehicle.png
   - display_platform.png
   - led_light_strip.png
   - background_screen_left.png
   - background_screen_right.png
   - robotic_arm.png
   - floor_panel.png

3. Module 3D Models
   每个模块通过 image-to-3D 生成独立 GLB，然后进入统一 scene assembly。

4. Scene Assembly
   AI 根据 scene_plan.json 自动决定每个模块的：
   - scale
   - position
   - rotation
   - parent / child relation
   - anchor point
   - collision avoidance
   - foreground / midground / background role
   - camera visibility priority

==================================================
三、新增数据结构
==================================================

请新增以下 JSON schema。

--------------------------------------------------
1. auto_task.json
--------------------------------------------------

{
  "task_id": "...",
  "created_at": "...",
  "user_request": "...",
  "expanded_request": "...",
  "mode": "modular_scene_agent",
  "source_mode": "text_to_scene",
  "style_preset": "product | cinematic | ecommerce | concept | exhibition | architecture",
  "quality_mode": "fast | balanced | high",
  "geometry_mode": "loose | balanced | strict",
  "output_views": 3,
  "num_candidates_per_view": 3,
  "max_retries": 2
}

--------------------------------------------------
2. scene_plan.json
--------------------------------------------------

{
  "scene_type": "automotive_showroom | product_stage | ecommerce_scene | architecture | interior | game_asset_preview | custom",
  "main_subject": {
    "name": "future white electric hypercar",
    "role": "hero_object",
    "priority": 1
  },
  "environment": {
    "description": "neutral gray futuristic studio with reflective floor and blue accent lighting",
    "floor": "glossy gray reflective floor",
    "background": "clean gradient studio backdrop",
    "lighting": "softbox lighting with subtle blue LED accents"
  },
  "composition": {
    "camera_style": "three-quarter front hero view",
    "subject_coverage": "70-85%",
    "foreground": [],
    "midground": ["main_vehicle", "display_platform"],
    "background": ["led_screens", "light_strips"]
  },
  "global_style": {
    "material_language": "clean premium CGI product render",
    "color_palette": ["pearl white", "charcoal black", "cool blue", "neutral gray"],
    "avoid": ["text artifacts", "random logos", "dirty textures", "extra unplanned objects"]
  }
}

--------------------------------------------------
3. concept_image_plan.json
--------------------------------------------------

{
  "concept_prompt": "...",
  "negative_prompt": "...",
  "width": 1024,
  "height": 1024,
  "seed": 0,
  "backend": "image2 | configured_image_backend",
  "output": "concept/global_concept.png",
  "purpose": "global composition and style reference only"
}

--------------------------------------------------
4. module_plan.json
--------------------------------------------------

{
  "modules": [
    {
      "module_id": "main_vehicle",
      "name": "white futuristic electric hypercar",
      "category": "vehicle",
      "role": "hero_object",
      "priority": 1,
      "generate_reference_image": true,
      "generate_3d": true,
      "reference_prompt": "...",
      "negative_prompt": "...",
      "expected_real_world_size": {
        "width": 2.1,
        "depth": 4.8,
        "height": 1.2,
        "unit": "meters"
      },
      "placement": {
        "anchor": "scene_center",
        "position": [0, 0, 0],
        "rotation_deg": [0, 0, 0],
        "scale_policy": "real_world_vehicle_scale"
      },
      "constraints": [
        "must be the largest and most visible object",
        "must face hero camera",
        "must not intersect with platform"
      ]
    },
    {
      "module_id": "display_platform",
      "name": "low black circular display platform",
      "category": "stage_prop",
      "role": "supporting_object",
      "priority": 2,
      "generate_reference_image": true,
      "generate_3d": true,
      "expected_real_world_size": {
        "width": 5.5,
        "depth": 5.5,
        "height": 0.25,
        "unit": "meters"
      },
      "placement": {
        "anchor": "under_main_vehicle",
        "position": [0, 0, -0.15],
        "rotation_deg": [0, 0, 0],
        "scale_policy": "fit_under_hero_object"
      }
    },
    {
      "module_id": "left_led_screen",
      "name": "vertical blue glowing display screen",
      "category": "background_prop",
      "role": "background",
      "priority": 3,
      "generate_reference_image": true,
      "generate_3d": true,
      "placement": {
        "anchor": "back_left",
        "position": [-3.0, 2.5, 1.2],
        "rotation_deg": [0, 20, 0],
        "scale_policy": "background_visible_not_dominant"
      }
    }
  ]
}

--------------------------------------------------
5. module_asset_manifest.json
--------------------------------------------------

{
  "modules": [
    {
      "module_id": "main_vehicle",
      "reference_image": "modules/main_vehicle/reference.png",
      "preprocessed_image": "modules/main_vehicle/preprocessed.png",
      "model_path": "modules/main_vehicle/model.glb",
      "metadata": "modules/main_vehicle/metadata.json",
      "sanity": {
        "vertices": 312693,
        "faces": 625706,
        "component_count": 6,
        "largest_component_ratio": 0.999,
        "status": "pass | needs_review | failed"
      },
      "bbox": {
        "width": 2.1,
        "depth": 4.8,
        "height": 1.2
      }
    }
  ]
}

--------------------------------------------------
6. scene_assembly.json
--------------------------------------------------

{
  "coordinate_system": "blender_z_up",
  "units": "meters",
  "modules": [
    {
      "module_id": "main_vehicle",
      "model_path": "modules/main_vehicle/model.glb",
      "transform": {
        "position": [0, 0, 0.35],
        "rotation_deg": [0, 0, 0],
        "scale": [1.0, 1.0, 1.0]
      },
      "visibility_priority": 1,
      "material_hint": "pearl white ceramic paint"
    },
    {
      "module_id": "display_platform",
      "model_path": "modules/display_platform/model.glb",
      "transform": {
        "position": [0, 0, 0.0],
        "rotation_deg": [0, 0, 0],
        "scale": [1.0, 1.0, 1.0]
      },
      "visibility_priority": 2,
      "material_hint": "matte black platform"
    }
  ],
  "collision_report": {
    "has_major_collision": false,
    "warnings": []
  }
}

--------------------------------------------------
7. final_scene_manifest.json
--------------------------------------------------

{
  "task_id": "...",
  "scene_model_path": "scene/final_scene.glb",
  "scene_assembly": "scene_assembly.json",
  "camera_plan": "camera_plan.json",
  "render_manifest": "renders/render_manifest.json",
  "agent_report": "reports/agent_report.json"
}

==================================================
四、多模态需求扩写与模块识别
==================================================

请实现 Multimodal Requirement Expander。

第一阶段可以是规则版 + 可插拔 LLM backend，不要求一定接外部 LLM。

输入：
{
  "request": "用户一句话",
  "optional_reference_image": null
}

输出：
1. expanded_request；
2. scene_type；
3. main_subject；
4. environment；
5. style_preset；
6. required_modules；
7. optional_modules；
8. forbidden_modules；
9. concept_image_prompt；
10. per_module_prompts；
11. camera_intent；
12. material_intent；
13. quality_constraints。

示例：

用户输入：
“做一个未来汽车发布会展台，中央一辆白色跑车，旁边有机械臂和发光屏幕。”

输出模块：
- main_vehicle
- display_platform
- robotic_arm_left
- robotic_arm_right
- led_screen_left
- led_screen_right
- reflective_floor
- studio_background
- blue_light_strips

模块识别规则：
1. 名词短语通常是候选模块；
2. 带有空间关系的短语必须进入 placement constraints；
3. “中央、旁边、背后、上方、下方、环绕、左右两侧”等词必须转为布局约束；
4. “白色、金属、玻璃、发光、透明、粗糙、柔软”等词必须转为材质提示；
5. “产品图、发布会、展台、电商图、建筑概念、游戏资产”必须影响 style_preset 和 camera_plan；
6. 用户没有明确说的模块不要随意添加，除非是基础场景元素，如 floor、background、light。

==================================================
五、总体概念图生成
==================================================

请新增 Concept Image Generation 阶段。

目标：
在 3D 建模前先生成一张总体概念图，用于确定：
- 总体构图；
- 主体与配角关系；
- 色彩；
- 灯光；
- 场景风格；
- 模块列表是否合理。

实现要求：
1. 通过当前已有 image generation backend 调用，暂时可以由 Codex 使用 image2 生成；
2. 输出 concept/global_concept.png；
3. 写入 concept_image_plan.json；
4. 不要直接把 concept image 传给最终 AI 渲染模型，除非进入显式 concept-guided final render 模式；
5. concept image 可以用于：
   - 生成每个模块的 reference prompt；
   - 辅助布局；
   - 辅助风格一致性描述；
   - 用户预览和确认。

Concept prompt 示例：

“Wide premium CGI concept image of a futuristic automotive launch stage. A pearl white low electric hypercar sits in the center on a low black circular platform. Clean gray reflective studio floor, two vertical blue glowing screens in the background, subtle robotic arms on both sides, softbox lighting, minimal unbranded design, high-end product presentation, clean composition, no text, no logos.”

==================================================
六、每个模块独立生图
==================================================

请新增 Module Reference Image Generation 阶段。

目标：
根据 module_plan.json，为每个需要 3D 建模的模块生成独立参考图。

要求：
1. 每个模块一张干净、主体居中、背景简单的参考图；
2. 尽量使用白底或透明背景；
3. 不要在模块参考图中加入其他模块；
4. 每个模块 prompt 必须继承 global_style；
5. 输出路径：
   modules/{module_id}/reference.png
6. 每个模块写入：
   modules/{module_id}/reference_manifest.json

模块参考图 prompt 示例：

main_vehicle:
“Single object reference image of a futuristic pearl white electric hypercar, low wide silhouette, clean unbranded body, dark glass canopy, black wheels, centered on plain white background, orthographic three-quarter view, no text, no logos, no environment.”

display_platform:
“Single object reference image of a low black circular automotive display platform, minimal futuristic design, subtle blue LED edge lighting, centered on plain white background, no vehicle, no people, no text.”

robotic_arm:
“Single object reference image of a sleek black industrial robotic arm for automotive showroom display, compact base, clean mechanical joints, centered on plain white background, no vehicle, no text.”

led_screen:
“Single object reference image of a vertical rectangular futuristic LED display panel, thin black frame, glowing blue abstract light surface, centered on plain white background, no text, no logo.”

==================================================
七、每个模块 image-to-3D
==================================================

请新增 Module Image-to-3D 阶段。

目标：
对每个模块 reference.png 调用 Hunyuan3D image-to-3D 或现有 image2-to-3D 管线，生成独立 GLB。

要求：
1. 每个模块独立输出：
   modules/{module_id}/model.glb
   modules/{module_id}/metadata.json
   modules/{module_id}/sanity.json
2. 使用已有背景去除和预处理逻辑；
3. 对每个 mesh 做 sanity check；
4. 如果模块失败，根据重要程度决定：
   - hero_object 失败：整个任务 failed；
   - supporting_object 失败：重试；
   - background_prop 失败：可降级为简单几何代理；
   - decorative_object 失败：可跳过并记录 warning。

模块失败降级策略：
- display_platform 失败：用 Blender 程序化圆柱体替代；
- floor 失败：用 Blender 平面替代；
- background screen 失败：用 Blender 矩形面板替代；
- light strip 失败：用曲线 / 发光材质替代；
- simple wall 失败：用平面替代；
- main vehicle / character / core product 失败：必须重试或失败。

==================================================
八、AI 自动缩放与位置摆放
==================================================

请新增 Scene Layout Agent。

输入：
- scene_plan.json
- module_plan.json
- module_asset_manifest.json
- 每个模块 bbox
- 每个模块 role
- 每个模块 expected_real_world_size
- 用户空间描述

输出：
- scene_assembly.json
- final_scene.glb 或 final_scene.blend

布局原则：
1. hero_object 放在世界中心或视觉中心；
2. supporting_object 根据 anchor 依附 hero；
3. background_prop 放在主体后方；
4. decorative_object 不得遮挡 hero；
5. 所有模块默认落地；
6. 需要支持 z-up Blender 坐标；
7. 避免明显穿模；
8. 保证相机可见；
9. 保证主体占最终画面 65% 到 85%；
10. 每个 transform 都必须可解释，并写入 layout_reason。

scene_assembly.json 示例：

{
  "modules": [
    {
      "module_id": "main_vehicle",
      "transform": {
        "position": [0, 0, 0.35],
        "rotation_deg": [0, 0, 0],
        "scale": [1.0, 1.0, 1.0]
      },
      "layout_reason": "hero object centered on platform and facing hero camera"
    },
    {
      "module_id": "display_platform",
      "transform": {
        "position": [0, 0, 0.0],
        "rotation_deg": [0, 0, 0],
        "scale": [1.2, 1.2, 0.15]
      },
      "layout_reason": "supporting platform scaled to fit under vehicle"
    },
    {
      "module_id": "left_led_screen",
      "transform": {
        "position": [-3.0, 2.8, 1.4],
        "rotation_deg": [0, 0, 15],
        "scale": [1.0, 0.08, 2.0]
      },
      "layout_reason": "background screen placed behind and to the left of hero object"
    }
  ]
}

必须实现基本自动缩放：
- 如果模块有 expected_real_world_size，则按真实尺寸缩放；
- 如果没有，则根据 role 估算：
  hero_object: 标准化到主尺度；
  supporting_object: 0.5 到 1.2 倍 hero；
  background_prop: 0.5 到 2.0 倍 hero，但不得抢主体；
  decorative_object: 0.1 到 0.5 倍 hero。

必须实现基本碰撞检查：
- bbox 是否严重重叠；
- 是否低于地面；
- 是否遮挡 hero；
- 是否离主体过远；
- 是否完全不在相机视野。

==================================================
九、场景合成
==================================================

请新增 Scene Assembler。

目标：
把多个模块 GLB 导入 Blender，按 scene_assembly.json 放置，并输出完整场景。

输出：
- scene/final_scene.blend
- scene/final_scene.glb
- scene/scene_preview.png
- scene/assembly_report.json

Scene Assembler 要支持：
1. 导入多个 GLB；
2. 应用 scale / rotation / position；
3. 添加程序化地面；
4. 添加背景面或摄影棚；
5. 添加基础灯光；
6. 给失败模块使用 procedural fallback；
7. 输出一个完整 scene model；
8. 可被后续 batch_render.py 直接渲染。

建议新增：
blender_scripts/assemble_scene.py

调用方式示例：
blender --background --python blender_scripts/assemble_scene.py -- \
  --assembly scene_assembly.json \
  --output scene/final_scene.blend \
  --export-glb scene/final_scene.glb \
  --preview scene/scene_preview.png

==================================================
十、自动相机与白模通道
==================================================

Scene Assembly 完成后，进入已有相机与白模通道流程。

但要注意：
以前是单模型白模；
现在是多模块 scene 白模。

请修改或扩展 batch_render.py，使其支持：
1. 单模型模式；
2. 多模块场景模式；
3. scene_assembly.json 输入；
4. final_scene.blend 输入；
5. 多视角输出。

默认三视角：
- hero
- left_30
- right_30

每个视角输出：
- rgb
- edge
- mask
- depth
- normal

render_manifest.json 必须记录：
- scene_model_path
- scene_assembly_path
- view_id
- camera
- channels
- module_ids_visible，可选
- resolution
- samples

==================================================
十一、最终 AI 渲染
==================================================

最终 AI 渲染阶段继续沿用 geometry-locked 原则。

默认输入：
- view_x/rgb.png
- view_x/edge.png

评分辅助：
- mask
- depth
- normal

不要默认把 global_concept.png 或 module reference.png 传给最终渲染模型。

如果确实需要风格一致，可以把 global concept 的文字描述、色彩 palette、scene_plan 中的 style 字段转成 prompt，而不是直接喂图。

默认 render prompt 应该由 scene_plan 自动生成：

“Render the same assembled 3D scene as a premium CGI product image. Preserve the exact spatial layout, object placement, scale relationships, camera angle, silhouette, and visible geometry from the white render and edge guide. Keep the central vehicle as the hero object. Maintain the platform, background screens, light strips and robotic arms in their original positions. Use clean pearl white automotive paint, dark glass, black tires, cool blue accent lighting, neutral gray reflective studio floor, softbox lighting, smooth spotless surfaces, no text, no logos, no added objects, no changed layout.”

==================================================
十二、模块化评分
==================================================

请扩展评分逻辑，不只评整张图，还要评模块是否存在和位置是否正确。

新增 module_presence_score：

输入：
- scene_assembly.json
- render_manifest mask / edge
- final image

指标：
1. hero_object 是否存在；
2. supporting_object 是否没有消失；
3. background_prop 是否大致存在；
4. 模块是否被错误移动；
5. 模块是否被图像模型融合成奇怪形状；
6. 是否新增了未规划物体。

输出：
{
  "module_scores": [
    {
      "module_id": "main_vehicle",
      "presence": 0.95,
      "position_adherence": 0.88,
      "scale_adherence": 0.83,
      "status": "pass"
    }
  ],
  "total": 0.0
}

第一阶段可以用 bbox / mask / edge 的轻量方法，不要求复杂 VLM。

==================================================
十三、前端产品形态
==================================================

Web 工作台新增 Auto Scene Mode。

用户输入区：
- 一句话需求
- 输出视角数量
- 质量模式
- 几何严格程度
- 候选数量
- 最大重试次数
- 是否允许程序化 fallback
- 是否需要用户确认 concept image

阶段展示：

1. 需求理解
2. 总体概念图
3. 模块拆分
4. 模块参考图
5. 模块 3D 建模
6. 模块质量检查
7. 场景摆放
8. 场景预览
9. 相机规划
10. 白模通道
11. AI 渲染
12. 候选评分
13. 多视图一致性
14. 最终输出

每个阶段都要显示：
- status
- artifact
- message
- retry_count
- warnings
- error

用户应该可以看到：
- global_concept.png
- module reference images
- 每个模块的 GLB 链接
- scene_preview.png
- white render
- final render
- contact sheet
- agent report

==================================================
十四、CLI
==================================================

新增 CLI：

local3dai auto-scene \
  --request "生成一个未来汽车发布会展台，中央是一辆白色跑车，周围有蓝色灯带、黑色展示台、发光屏幕和机械臂" \
  --output outputs/auto_scene/demo_showroom \
  --views 3 \
  --quality balanced \
  --geometry strict \
  --style exhibition \
  --candidates 3 \
  --max-retries 2 \
  --allow-procedural-fallback

CLI 必须和 Web 使用同一套 workflow。

==================================================
十五、建议输出目录
==================================================

outputs/auto_scene/{task_id}/
  auto_task.json
  scene_plan.json
  concept/
    concept_image_plan.json
    global_concept.png
  modules/
    module_plan.json
    main_vehicle/
      reference.png
      reference_manifest.json
      preprocessed.png
      model.glb
      metadata.json
      sanity.json
    display_platform/
      reference.png
      model.glb
      sanity.json
    left_led_screen/
    right_led_screen/
    robotic_arm_left/
    robotic_arm_right/
  scene/
    scene_assembly.json
    final_scene.blend
    final_scene.glb
    scene_preview.png
    assembly_report.json
  cameras/
    camera_plan.json
  renders/
    render_manifest.json
    view_hero/
      rgb.png
      edge.png
      mask.png
      depth.png
      normal.png
    view_left_30/
    view_right_30/
  ai/
    view_hero/
      candidate_00.png
      candidate_01.png
      selected.png
      scores.json
    view_left_30/
    view_right_30/
  final/
    final_view_hero.png
    final_view_left_30.png
    final_view_right_30.png
    contact_sheet.png
    white_vs_final.png
  reports/
    module_scores.json
    structure_scores.json
    multiview_score.json
    agent_report.json
    run.log

==================================================
十六、测试要求
==================================================

必须保证 mock backend 下可跑通，不依赖真实大模型。

新增测试：

1. test_scene_expander.py
   测试一句话能拆出 scene_plan 和 module_plan。

2. test_concept_plan.py
   测试 concept_image_plan 生成。

3. test_module_plan.py
   测试模块 prompt、模块 role、expected size、placement constraints。

4. test_scene_layout.py
   测试模块 bbox 自动缩放和摆放。

5. test_scene_assembly_manifest.py
   测试 scene_assembly.json schema。

6. test_auto_scene_workflow_mock.py
   使用 mock image backend、mock 3D backend 跑通完整 auto-scene 流程。

7. test_module_failure_policy.py
   测试 hero 失败时任务 failed，平台 / 屏幕失败时 procedural fallback。

8. Playwright 测试：
   - 输入一句话；
   - 启动 Auto Scene Mode；
   - 查看 concept image；
   - 查看 module images；
   - 查看 scene preview；
   - 查看 final contact sheet。

运行：
python3 -m pytest -q
git diff --check
node scripts/validate_web_workbench.mjs

==================================================
十七、优先级
==================================================

P0 必须完成：
- scene_plan.json
- module_plan.json
- concept_image_plan.json
- requirement expander 支持模块拆分
- concept image generation 阶段
- module reference image generation 阶段
- module image-to-3D workflow 封装
- scene layout agent v1
- scene_assembly.json
- auto-scene CLI
- /api/auto-scene
- mock backend 完整闭环

P1 强烈建议完成：
- blender_scripts/assemble_scene.py
- procedural fallback
- scene_preview.png
- 三视角 camera_plan
- module presence score
- contact_sheet
- Web Auto Scene Mode 阶段展示

P2 后续扩展：
- 多视图一致性评分增强
- face_id / visibility buffer
- appearance memory
- 局部修复
- 用户确认 concept image 后再继续
- LLM backend planner
- 模块级材质继承
- 复杂碰撞和遮挡优化

==================================================
十八、验收标准
==================================================

完成后必须满足：

1. 用户可以输入一句话启动 Auto Scene Mode。
2. 系统能生成总体概念图。
3. 系统能自动拆分至少 3 个模块。
4. 系统能为每个模块生成 reference image。
5. 系统能为每个模块调用 image-to-3D 或 mock 3D backend。
6. 系统能为模块生成 module_asset_manifest.json。
7. 系统能自动计算每个模块 scale / position / rotation。
8. 系统能生成 scene_assembly.json。
9. 系统能导出 final_scene.glb 或 final_scene.blend。
10. 系统能对完整场景渲染 rgb / edge / mask / depth / normal。
11. 系统能进入最终 AI 渲染阶段。
12. 系统能输出至少 hero view 的 final image。
13. 三视角模式下能输出 contact_sheet。
14. 所有阶段都有 manifest、log、status 和 error 信息。
15. mock backend 下测试必须通过。
16. 真实 backend 可作为高质量路径，但不能阻塞测试。
17. 最终 AI 渲染不得绕过 render_manifest 直接使用原始用户输入图或未声明参考图。

==================================================
十九、最终交付说明
==================================================

完成后请输出：

1. 实施计划；
2. 改动文件列表；
3. 新增 API；
4. 新增 CLI；
5. 输出目录示例；
6. mock backend 测试结果；
7. 真实 backend 使用说明；
8. 当前限制；
9. 下一步建议。

请先阅读项目并给出实施计划，然后分 P0 / P1 小步实现，不要一次性大规模重构。