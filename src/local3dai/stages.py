from __future__ import annotations

from typing import Any


PIPELINE_STAGE_DEFINITIONS: list[dict[str, str]] = [
    {
        "id": "prepare",
        "label": "准备",
        "description": "加载配置、创建目录并初始化本次任务。",
    },
    {
        "id": "source",
        "label": "3D 来源",
        "description": "解析已有模型、程序化模型或 Hunyuan3D 白模来源。",
    },
    {
        "id": "render",
        "label": "白模通道",
        "description": "用 Blender 渲染 RGB、Depth、Edge、Normal、Mask 通道。",
    },
    {
        "id": "agent",
        "label": "Agent 调优",
        "description": "自动选择提示词、参考通道、seed 和视图扩展策略。",
    },
    {
        "id": "ai",
        "label": "AI 渲染",
        "description": "按固定参数生成 AI 渲染候选图。",
    },
    {
        "id": "score",
        "label": "结构评分",
        "description": "对候选图进行边缘、遮罩和结构一致性排序。",
    },
    {
        "id": "package",
        "label": "打包产物",
        "description": "复制最终图、生成对照图并写出 summary/report。",
    },
    {
        "id": "complete",
        "label": "完成",
        "description": "流程结束，所有产物可查看。",
    },
]

AUTO_STAGE_DEFINITIONS: list[dict[str, str]] = [
    {"id": "understand", "label": "理解需求", "description": "解析一句话需求并调用配置的 Qwen 规划器。"},
    {"id": "expand", "label": "扩写提示", "description": "生成 auto_task 和 prompt_plan。"},
    {"id": "plan", "label": "规划流程", "description": "决定来源模式、质量目标、候选数量和重试策略。"},
    {"id": "source", "label": "3D 来源", "description": "生成、导入或选择 3D 白模来源。"},
    {"id": "mesh_check", "label": "模型检查", "description": "检查模型文件存在性、类型和基本可用性。"},
    {"id": "camera", "label": "相机规划", "description": "生成可复现的 Blender 相机状态和多视角计划。"},
    {"id": "render", "label": "白模通道", "description": "渲染 RGB、Edge、Mask、Depth、Normal 等模型通道。"},
    {"id": "agent", "label": "候选生成", "description": "从 render_manifest 通道生成 AI 候选图。"},
    {"id": "score", "label": "候选评分", "description": "执行结构评分和多视图一致性评分。"},
    {"id": "retry", "label": "重试决策", "description": "根据评分决定是否需要收紧提示词或重试。"},
    {"id": "package", "label": "打包产物", "description": "写出最终图、报告、日志和参数记录。"},
    {"id": "complete", "label": "完成", "description": "流程结束，输出 Done / Needs Review / Failed。"},
]

AUTO_SCENE_STAGE_DEFINITIONS: list[dict[str, str]] = [
    {"id": "concept", "label": "总体概念图", "description": "生成全局概念图和 concept_image_plan。"},
    {"id": "decompose", "label": "模块拆分", "description": "生成 scene_plan 和 module_plan。"},
    {"id": "module_reference", "label": "模块参考图", "description": "为每个模块生成 reference image。"},
    {"id": "module_3d", "label": "模块 3D 建模", "description": "为每个模块生成或模拟独立 3D 资产。"},
    {"id": "module_check", "label": "模块质量检查", "description": "检查模块 mesh 并应用失败降级策略。"},
    {"id": "layout", "label": "场景摆放", "description": "计算每个模块的位置、旋转、缩放和布局原因。"},
    {"id": "scene_preview", "label": "场景预览", "description": "合成 final_scene.glb 并生成 scene_preview。"},
    {"id": "consistency", "label": "多视图一致性", "description": "记录多视图和模块存在性检查结果。"},
]

STAGE_DEFINITIONS = [*PIPELINE_STAGE_DEFINITIONS, *AUTO_STAGE_DEFINITIONS, *AUTO_SCENE_STAGE_DEFINITIONS]
PIPELINE_STAGE_LABELS = {stage["id"]: stage["label"] for stage in STAGE_DEFINITIONS}
PIPELINE_STAGE_DESCRIPTIONS = {stage["id"]: stage["description"] for stage in STAGE_DEFINITIONS}
PIPELINE_STAGE_ORDER = [stage["id"] for stage in PIPELINE_STAGE_DEFINITIONS]


def planned_stage_ids(*, agent_render: bool) -> list[str]:
    stages = ["prepare", "source", "render"]
    stages.extend(["agent"] if agent_render else ["ai", "score"])
    stages.extend(["package", "complete"])
    return stages


def stage_definition(stage_id: str) -> dict[str, str]:
    return {
        "id": stage_id,
        "label": PIPELINE_STAGE_LABELS.get(stage_id, stage_id),
        "description": PIPELINE_STAGE_DESCRIPTIONS.get(stage_id, ""),
    }


def stage_sort_key(stage: dict[str, Any]) -> int:
    try:
        return PIPELINE_STAGE_ORDER.index(str(stage.get("id", "")))
    except ValueError:
        return len(PIPELINE_STAGE_ORDER)
