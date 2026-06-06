"""训练指标折线图导出器。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tunebench.artifacts.names import TRAIN_METRICS_FILENAME

from .base import TrainMetricsPlotsExportResult


_LOSS_COLUMNS = {"loss"}
_DEFAULT_X_FIELD = "step"
_EVAL_METRICS_X_FIELD = "epoch"
_NON_METRIC_COLUMNS = {"epoch", "step"}
_TRAIN_LOSS_COLUMN = "train_loss"
_VALIDATION_LOSS_COLUMN = "validation_loss"


class MatplotlibTrainMetricsPlotExporter:
    """从训练指标 CSV 生成折线图。"""

    def _filter_rows(
        self,
        rows: list[dict[str, str]],
        *,
        split_name: str | None = None,
        stage_name: str | None = None,
    ) -> list[dict[str, str]]:
        filtered_rows = rows
        if split_name is not None:
            filtered_rows = [row for row in filtered_rows if (row.get("split") or "").strip() == split_name]
        if stage_name is not None:
            filtered_rows = [row for row in filtered_rows if (row.get("stage") or "").strip() == stage_name]
        return filtered_rows

    def _read_rows(self, csv_path: Path) -> list[dict[str, str]]:
        with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            return [dict(row) for row in reader]

    def _parse_float(self, value: str | None) -> float | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        try:
            return float(normalized)
        except ValueError:
            return None

    def _collect_numeric_columns(self, rows: list[dict[str, str]]) -> list[str]:
        if not rows:
            return []

        numeric_columns: list[str] = []
        for fieldname in rows[0]:
            if any(self._parse_float(row.get(fieldname)) is not None for row in rows):
                numeric_columns.append(fieldname)
        return numeric_columns

    def _resolve_x_values(
        self,
        rows: list[dict[str, str]],
        numeric_columns: list[str],
        preferred_x_field: str = _DEFAULT_X_FIELD,
    ) -> tuple[str | None, list[float]]:
        x_field = preferred_x_field if preferred_x_field in numeric_columns else None
        x_values: list[float] = [float(index) for index in range(1, len(rows) + 1)]
        if x_field is None:
            return None, x_values

        parsed_x_values = [self._parse_float(row.get(x_field)) for row in rows]
        if any(value is not None for value in parsed_x_values):
            x_values = [value if value is not None else float(index + 1) for index, value in enumerate(parsed_x_values)]
            return x_field, x_values
        return None, [float(index) for index in range(1, len(rows) + 1)]

    def _build_series_values(self, rows: list[dict[str, str]], column_name: str) -> list[float]:
        y_values: list[float] = []
        for row in rows:
            parsed_value = self._parse_float(row.get(column_name))
            y_values.append(parsed_value if parsed_value is not None else float("nan"))
        return y_values

    def _build_loss_rows(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        loss_rows: list[dict[str, str]] = []
        for row in rows:
            stage_name = (row.get("stage") or "").strip()
            if stage_name not in {"train", "epoch_evaluate"}:
                continue

            normalized_row = {
                "step": row.get("step", ""),
                _TRAIN_LOSS_COLUMN: "",
                _VALIDATION_LOSS_COLUMN: "",
            }
            loss_value = row.get("loss", "")
            if stage_name == "train":
                normalized_row[_TRAIN_LOSS_COLUMN] = loss_value
            else:
                normalized_row[_VALIDATION_LOSS_COLUMN] = loss_value
            loss_rows.append(normalized_row)
        return loss_rows

    def _plot_columns(
        self,
        rows: list[dict[str, str]],
        x_field: str | None,
        x_values: list[float],
        y_columns: list[str],
        output_path: Path,
        title: str,
        y_label: str,
    ) -> Path | None:
        if not y_columns:
            return None

        selected_indices = [
            index
            for index, row in enumerate(rows)
            if any(self._parse_float(row.get(column_name)) is not None for column_name in y_columns)
        ]
        if not selected_indices:
            return None

        selected_x_values = [x_values[index] for index in selected_indices]

        figure, axis = plt.subplots(figsize=(12, 6))
        for column_name in y_columns:
            y_values = [self._build_series_values(rows, column_name)[index] for index in selected_indices]
            if all(np.isnan(value) for value in y_values):
                continue
            axis.plot(selected_x_values, y_values, marker="o", linewidth=1.5, markersize=3, label=column_name)

        if not axis.lines:
            plt.close(figure)
            return None

        axis.set_title(title)
        axis.set_xlabel(x_field or "row_index")
        axis.set_ylabel(y_label)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(loc="best")
        figure.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        return output_path

    def export(
        self,
        input_path: Path,
        loss_output_path: Path,
        eval_metrics_output_path: Path,
    ) -> TrainMetricsPlotsExportResult:
        """读取训练指标 CSV 并分别导出 loss 与验证指标图。"""
        if not input_path.exists():
            return TrainMetricsPlotsExportResult(warnings=[f"缺少训练指标文件: {input_path}"])

        rows = self._read_rows(input_path)
        numeric_columns = self._collect_numeric_columns(rows)
        if not rows or not numeric_columns:
            return TrainMetricsPlotsExportResult(warnings=["训练指标 CSV 中没有可绘制的数值列。"])

        loss_rows = self._build_loss_rows(rows)
        loss_numeric_columns = self._collect_numeric_columns(loss_rows)
        x_field, x_values = self._resolve_x_values(loss_rows, loss_numeric_columns, preferred_x_field=_DEFAULT_X_FIELD)
        y_columns = [column for column in loss_numeric_columns if column != x_field]
        loss_columns = [column for column in y_columns if column in {_TRAIN_LOSS_COLUMN, _VALIDATION_LOSS_COLUMN}]

        validation_rows = self._filter_rows(rows, split_name="validation", stage_name="epoch_evaluate")
        validation_numeric_columns = self._collect_numeric_columns(validation_rows)
        validation_x_field, validation_x_values = self._resolve_x_values(
            validation_rows,
            validation_numeric_columns,
            preferred_x_field=_EVAL_METRICS_X_FIELD,
        )
        eval_metric_columns = [
            column
            for column in validation_numeric_columns
            if column not in _LOSS_COLUMNS and column not in _NON_METRIC_COLUMNS and column != validation_x_field
        ]

        warnings: list[str] = []
        loss_plot_path = self._plot_columns(
            rows=loss_rows,
            x_field=x_field,
            x_values=x_values,
            y_columns=loss_columns,
            output_path=loss_output_path,
            title="Train Loss Trend",
            y_label="loss",
        )
        eval_metrics_plot_path = self._plot_columns(
            rows=validation_rows,
            x_field=validation_x_field,
            x_values=validation_x_values,
            y_columns=eval_metric_columns,
            output_path=eval_metrics_output_path,
            title="Validation Metrics Trend",
            y_label="metric",
        )

        if loss_plot_path is None:
            warnings.append(f"未生成 loss 图，{TRAIN_METRICS_FILENAME} 中缺少可绘制的 loss 列。")
        if eval_metrics_plot_path is None:
            warnings.append("未生成验证指标图，validation 行中缺少可按 epoch 绘制的验证指标列。")

        return TrainMetricsPlotsExportResult(
            loss_plot_path=loss_plot_path,
            eval_metrics_plot_path=eval_metrics_plot_path,
            warnings=warnings,
        )