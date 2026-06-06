"""结构化 target 构建阶段契约。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StructuredTargetBuildSpec:
    """描述一次结构化 target 构建任务。"""

    task_name: str
    source_dataset_version: str
    target_dataset_version: str
    confidence: float = 0.9
    splits: tuple[str, ...] = ("train", "validation")