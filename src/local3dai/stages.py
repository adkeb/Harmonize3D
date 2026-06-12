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

PIPELINE_STAGE_LABELS = {stage["id"]: stage["label"] for stage in PIPELINE_STAGE_DEFINITIONS}
PIPELINE_STAGE_DESCRIPTIONS = {stage["id"]: stage["description"] for stage in PIPELINE_STAGE_DEFINITIONS}
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
