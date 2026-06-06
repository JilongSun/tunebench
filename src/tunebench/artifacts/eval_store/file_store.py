"""基于本地文件系统的评测产物存储实现。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tunebench.artifacts.names import (
    DEFAULT_EVAL_REPORT_ARTIFACT_NAMES,
    TEST_LABEL_METRICS_ARTIFACT_NAME,
    TEST_METRICS_ARTIFACT_NAME,
    TEST_PREDICTIONS_ARTIFACT_NAME,
    TRAIN_METRICS_ARTIFACT_NAME,
    VALIDATION_LABEL_METRICS_ARTIFACT_NAME,
)
from tunebench.artifacts.path import ModelArtifactLayout

from .base import EvalArtifactStore, EvalReportExportResult, TrainMetricsPlotsExportResult
from .plot_exporter import MatplotlibTrainMetricsPlotExporter
from .xlsx_exporter import XlsxEvalReportExporter


class CsvArtifactWriter:
    """统一管理 CSV 产物追加写入。"""

    def append_rows(self, output_path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = output_path.exists()
        with output_path.open("a", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
        return output_path


class FileSystemEvalArtifactStore(EvalArtifactStore):
    """将评测产物存储到本地文件系统。"""

    def __init__(
        self,
        csv_writer: CsvArtifactWriter | None = None,
        xlsx_exporter: XlsxEvalReportExporter | None = None,
        train_metrics_plot_exporter: MatplotlibTrainMetricsPlotExporter | None = None,
    ) -> None:
        self.csv_writer = csv_writer or CsvArtifactWriter()
        self.xlsx_exporter = xlsx_exporter or XlsxEvalReportExporter()
        self.train_metrics_plot_exporter = train_metrics_plot_exporter or MatplotlibTrainMetricsPlotExporter()
        self.csv_artifact_path_getters = {
            TRAIN_METRICS_ARTIFACT_NAME: lambda model_layout: model_layout.train_metrics_csv,
            VALIDATION_LABEL_METRICS_ARTIFACT_NAME: lambda model_layout: model_layout.validation_label_metrics_csv,
            TEST_METRICS_ARTIFACT_NAME: lambda model_layout: model_layout.test_metrics_csv,
            TEST_LABEL_METRICS_ARTIFACT_NAME: lambda model_layout: model_layout.test_label_metrics_csv,
            TEST_PREDICTIONS_ARTIFACT_NAME: lambda model_layout: model_layout.test_predictions_csv,
        }

    def _resolve_csv_path(self, model_layout: ModelArtifactLayout, artifact_name: str) -> Path:
        try:
            path_getter = self.csv_artifact_path_getters[artifact_name]
        except KeyError as exc:
            supported_artifact_names = ", ".join(self.csv_artifact_path_getters)
            raise ValueError(f"不支持的评测产物名称: {artifact_name}；当前支持: {supported_artifact_names}") from exc
        return path_getter(model_layout)

    def append_artifact_rows(
        self,
        model_layout: ModelArtifactLayout,
        artifact_name: str,
        fieldnames: list[str],
        rows: list[dict[str, Any]],
    ) -> Path:
        output_path = self._resolve_csv_path(model_layout, artifact_name)
        return self.csv_writer.append_rows(output_path, fieldnames, rows)

    def export_eval_report(
        self,
        model_layout: ModelArtifactLayout,
        artifact_names: list[str] | None = None,
    ) -> EvalReportExportResult:
        artifact_names = artifact_names or list(DEFAULT_EVAL_REPORT_ARTIFACT_NAMES)
        csv_sources = [self._resolve_csv_path(model_layout, artifact_name) for artifact_name in artifact_names]
        optional_sources = {model_layout.train_metrics_csv, model_layout.validation_label_metrics_csv}
        return self.xlsx_exporter.export_from_csv_sources(
            csv_sources=csv_sources,
            output_path=model_layout.eval_report_xlsx,
            optional_sources=optional_sources,
        )

    def export_train_metrics_plots(self, model_layout: ModelArtifactLayout) -> TrainMetricsPlotsExportResult:
        return self.train_metrics_plot_exporter.export(
            input_path=model_layout.train_metrics_csv,
            loss_output_path=model_layout.train_loss_plot_png,
            eval_metrics_output_path=model_layout.train_eval_metrics_plot_png,
        )