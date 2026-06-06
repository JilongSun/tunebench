"""评测产物存储抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tunebench.artifacts.path import ModelArtifactLayout


@dataclass(slots=True)
class EvalReportExportResult:
    """描述一次评测汇总导出的结果。"""

    output_path: Path | None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TrainMetricsPlotsExportResult:
    """描述训练指标图导出结果。"""

    loss_plot_path: Path | None = None
    eval_metrics_plot_path: Path | None = None
    warnings: list[str] = field(default_factory=list)


class EvalArtifactStore(ABC):
    """定义评测产物存储能力。"""

    @abstractmethod
    def append_artifact_rows(
        self,
        model_layout: ModelArtifactLayout,
        artifact_name: str,
        fieldnames: list[str],
        rows: list[dict[str, Any]],
    ) -> Path:
        """追加写入一个表格型评测产物。"""

    @abstractmethod
    def export_eval_report(
        self,
        model_layout: ModelArtifactLayout,
        artifact_names: list[str] | None = None,
    ) -> EvalReportExportResult:
        """将指定评测产物汇总导出为报告。"""

    @abstractmethod
    def export_train_metrics_plots(self, model_layout: ModelArtifactLayout) -> TrainMetricsPlotsExportResult:
        """基于训练指标表导出训练图。"""