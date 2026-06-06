"""通用运行契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunPlan:
    """描述命令执行前的计划信息。"""

    stage: str
    summary: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StageResult:
    """描述单个环节的执行结果。"""

    stage: str
    success: bool
    message: str
    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)