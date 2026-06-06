"""分类任务数据准备器。"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tunebench.artifacts import DatasetPathManager, get_dataset_path_manager
from tunebench.contracts.classification import DatasetSpec
from tunebench.contracts.common import RunPlan, StageResult
from tunebench.util import XlsxJsonConverter, get_logger


logger = get_logger("classification.data_preparer")

_TRAIN_SPLIT_NAME = "train"
_VALIDATION_SPLIT_NAME = "validation"
_TEST_SPLIT_NAME = "test"


class ClassificationDataPreparer:
    """负责分类任务的数据清洗与格式转换。"""

    def __init__(
        self,
        dataset_path_manager: DatasetPathManager | None = None,
        converter: XlsxJsonConverter | None = None,
    ) -> None:
        self.dataset_path_manager = dataset_path_manager or get_dataset_path_manager()
        self.converter = converter or XlsxJsonConverter()

    def _resolve_primary_split_name(self, spec: DatasetSpec) -> str:
        """根据模式解析主 split 名称。"""
        return _TEST_SPLIT_NAME if spec.is_test else _TRAIN_SPLIT_NAME

    def _resolve_final_asset_path(self, final_dir: Path, split_name: str, output_format: str) -> Path:
        """解析 assets/final 下的标准输出路径。"""
        extension = "json" if output_format == "json" else "jsonl"
        return final_dir / f"{split_name}.{extension}"

    def _normalize_text(self, value: Any) -> str | None:
        """规范化文本字段。"""
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def _normalize_label(self, value: Any) -> str | None:
        """规范化标签字段。"""
        if value is None:
            return None
        if isinstance(value, float) and value.is_integer():
            normalized = str(int(value))
        else:
            normalized = str(value).strip()
        return normalized or None

    def _write_json(self, output_path: Path, payload: Any) -> Path:
        """写入 JSON 文件。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    def _write_jsonl(self, output_path: Path, records: list[dict[str, Any]]) -> Path:
        """写入 JSONL 文件。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return output_path

    def _write_records(
        self,
        records: list[dict[str, Any]],
        output_path: Path,
        output_format: str,
    ) -> Path:
        """按指定格式写出记录。"""
        if output_format == "json":
            return self._write_json(output_path, records)
        return self._write_jsonl(output_path, records)

    def _validate_split_config(self, spec: DatasetSpec) -> None:
        """校验自动切分参数。"""
        if not 0.0 <= spec.validation_ratio < 1.0:
            raise ValueError("validation_ratio 必须在 [0, 1) 区间内。")
        if spec.is_test and spec.validation_ratio > 0.0:
            raise ValueError("--is-test 模式下不允许同时传入 validation_ratio。")

    def _normalize_allowed_labels(self, allowed_labels: tuple[str, ...]) -> tuple[str, ...]:
        """规范化 CLI 传入的保留标签集合。"""
        normalized_labels: list[str] = []
        seen_labels: set[str] = set()

        for raw_label in allowed_labels:
            normalized_label = self._normalize_label(raw_label)
            if normalized_label is None or normalized_label in seen_labels:
                continue
            normalized_labels.append(normalized_label)
            seen_labels.add(normalized_label)

        return tuple(normalized_labels)

    def _stratified_split_pairs(
        self,
        stage_records: list[dict[str, Any]],
        final_records: list[dict[str, str]],
        validation_ratio: float,
        split_seed: int,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, str]],
        list[dict[str, Any]],
        list[dict[str, str]],
    ]:
        """按标签分层切分 train/validation。"""
        if validation_ratio <= 0.0:
            return stage_records, final_records, [], []

        grouped_pairs: dict[str, list[tuple[dict[str, Any], dict[str, str]]]] = defaultdict(list)
        for stage_record, final_record in zip(stage_records, final_records):
            grouped_pairs[final_record["label"]].append((stage_record, final_record))

        random_generator = random.Random(split_seed)
        train_pairs: list[tuple[dict[str, Any], dict[str, str]]] = []
        validation_pairs: list[tuple[dict[str, Any], dict[str, str]]] = []

        for label_name in sorted(grouped_pairs):
            pairs = list(grouped_pairs[label_name])
            random_generator.shuffle(pairs)
            validation_count = int(round(len(pairs) * validation_ratio))

            if len(pairs) >= 2 and validation_count == 0:
                validation_count = 1
            if validation_count >= len(pairs):
                validation_count = len(pairs) - 1
            if len(pairs) == 1:
                validation_count = 0

            validation_pairs.extend(pairs[:validation_count])
            train_pairs.extend(pairs[validation_count:])

        random_generator.shuffle(train_pairs)
        random_generator.shuffle(validation_pairs)

        train_stage_records = [stage_record for stage_record, _ in train_pairs]
        train_final_records = [final_record for _, final_record in train_pairs]
        validation_stage_records = [stage_record for stage_record, _ in validation_pairs]
        validation_final_records = [final_record for _, final_record in validation_pairs]
        return train_stage_records, train_final_records, validation_stage_records, validation_final_records

    def _build_stage_and_final_records(
        self,
        records: list[dict[str, Any]],
        spec: DatasetSpec,
        allowed_labels: tuple[str, ...],
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]], int, int, list[str]]:
        """将原始记录转换为 stage/final 两层。"""
        stage_records: list[dict[str, Any]] = []
        final_records: list[dict[str, str]] = []
        dropped_count = 0
        filtered_label_count = 0
        observed_labels: set[str] = set()
        allowed_label_set = set(allowed_labels)

        for row_number, record in enumerate(records, start=1):
            text_value = self._normalize_text(record.get(spec.text_key))
            label_value = self._normalize_label(record.get(spec.label_key))

            if text_value is None or label_value is None:
                dropped_count += 1
                continue

            observed_labels.add(label_value)
            if allowed_label_set and label_value not in allowed_label_set:
                filtered_label_count += 1
                continue

            stage_record = {
                "source_row": row_number,
                "text": text_value,
                "label": label_value,
            }
            final_record = {
                "text": text_value,
                "label": label_value,
            }
            stage_records.append(stage_record)
            final_records.append(final_record)

        return stage_records, final_records, dropped_count, filtered_label_count, sorted(observed_labels)

    def _load_dataset_metadata(self, metadata_path: Path, task_name: str, dataset_version: str) -> dict[str, Any]:
        """读取已有数据版本元数据，不存在时返回默认结构。"""
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("task_name", task_name)
                payload.setdefault("dataset_version", dataset_version)
                payload.setdefault(
                    "final_schema",
                    {
                        "text": "string",
                        "label": "string",
                    },
                )
                payload.setdefault("splits", {})
                return payload
        return {
            "task_name": task_name,
            "dataset_version": dataset_version,
            "final_schema": {
                "text": "string",
                "label": "string",
            },
            "splits": {},
        }

    def _cleanup_validation_artifacts(self, layout: Any) -> None:
        """当不再需要 validation 时，清理旧的 validation 文件。"""
        candidates = [
            layout.stage_dir / f"{_VALIDATION_SPLIT_NAME}.jsonl",
            layout.final_dir / f"{_VALIDATION_SPLIT_NAME}.jsonl",
            layout.final_dir / f"{_VALIDATION_SPLIT_NAME}.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                candidate.unlink()

    def build_plan(self, spec: DatasetSpec) -> RunPlan:
        """生成数据处理计划，供 CLI 或后续自动化入口复用。"""
        layout = self.dataset_path_manager.build_layout(spec.task_name, spec.dataset_version)
        primary_split_name = self._resolve_primary_split_name(spec)
        final_output_path = self._resolve_final_asset_path(layout.final_dir, primary_split_name, spec.output_format)
        export_output_path = spec.output_path or final_output_path
        outputs = {
            "dataset_version_dir": str(layout.version_dir),
            "raw_dataset": str(layout.raw_dir / f"{primary_split_name}.json"),
            "stage_dataset": str(layout.stage_dir / f"{primary_split_name}.jsonl"),
            "final_dataset": str(final_output_path),
            "export_dataset": str(export_output_path),
            "metadata": str(layout.metadata_path),
        }
        if not spec.is_test and spec.validation_ratio > 0.0:
            validation_extension = "json" if spec.output_format == "json" else "jsonl"
            outputs["validation_stage_dataset"] = str(layout.stage_dir / f"{_VALIDATION_SPLIT_NAME}.jsonl")
            outputs["validation_final_dataset"] = str(layout.final_dir / f"{_VALIDATION_SPLIT_NAME}.{validation_extension}")
        return RunPlan(
            stage="prepare-data",
            summary="执行分类数据清洗与标准化输出。",
            inputs=asdict(spec),
            outputs=outputs,
            notes=[
                "raw 保存原始表格快照，stage 保存筛选后的中间记录，final 输出标准 {text, label} 模板。",
                "默认模式固定写入 train；validation_ratio > 0 时，会按 label 做分层切分并生成 validation。",
                "--is-test 模式固定写入 test，不做切分。",
                "如传入 --keep-label，则会先做标签保留筛选，再进入标准化输出与分层切分。",
            ],
        )

    def run(self, spec: DatasetSpec) -> StageResult:
        """执行数据处理逻辑。"""
        try:
            logger.info(
                "开始数据清洗: task=%s, dataset_version=%s, is_test=%s, input=%s",
                spec.task_name,
                spec.dataset_version,
                spec.is_test,
                spec.input_path,
            )
            if spec.input_path.suffix.lower() not in {".xlsx", ".xls"}:
                logger.warning("输入文件格式不受支持: %s", spec.input_path)
                return StageResult(
                    stage="prepare-data",
                    success=False,
                    message="当前仅支持 xlsx/xls 输入，请提供 Excel 文件。",
                )

            if spec.output_format not in {"json", "jsonl"}:
                logger.warning("输出格式不受支持: %s", spec.output_format)
                return StageResult(
                    stage="prepare-data",
                    success=False,
                    message="output_format 仅支持 json 或 jsonl。",
                )

            self._validate_split_config(spec)
            allowed_labels = self._normalize_allowed_labels(spec.allowed_labels)
            if spec.allowed_labels and not allowed_labels:
                raise ValueError("--keep-label 解析后为空，请检查是否传入了非空标签名称。")

            layout = self.dataset_path_manager.ensure_layout(spec.task_name, spec.dataset_version)
            primary_split_name = self._resolve_primary_split_name(spec)
            raw_output_path = layout.raw_dir / f"{primary_split_name}.json"
            primary_stage_output_path = layout.stage_dir / f"{primary_split_name}.jsonl"
            primary_final_output_path = self._resolve_final_asset_path(layout.final_dir, primary_split_name, spec.output_format)
            export_output_path = spec.output_path

            convert_result = self.converter.read_records(
                spec.input_path,
                sheet_name=spec.sheet_name,
            )
            logger.info(
                "Excel 读取完成: rows=%s, columns=%s, sheet=%s",
                convert_result.row_count,
                len(convert_result.column_names),
                spec.sheet_name,
            )

            missing_columns = [
                column_name
                for column_name in (spec.text_key, spec.label_key)
                if column_name not in convert_result.column_names
            ]
            if missing_columns:
                missing_display = ", ".join(missing_columns)
                available_display = ", ".join(map(str, convert_result.column_names))
                logger.warning("Excel 缺少必要字段: %s", missing_display)
                return StageResult(
                    stage="prepare-data",
                    success=False,
                    message=f"Excel 缺少必要字段: {missing_display}。当前可用字段: {available_display}",
                )

            stage_records, final_records, dropped_count, filtered_label_count, observed_labels = self._build_stage_and_final_records(
                convert_result.records,
                spec,
                allowed_labels=allowed_labels,
            )
            if allowed_labels:
                missing_keep_labels = [label for label in allowed_labels if label not in observed_labels]
                if missing_keep_labels:
                    logger.warning(
                        "部分 keep-label 未在当前数据中出现: missing=%s, observed=%s",
                        missing_keep_labels,
                        observed_labels,
                    )
            if allowed_labels and not final_records:
                observed_display = ", ".join(observed_labels) if observed_labels else "无"
                logger.warning(
                    "标签筛选后无可用记录: keep_labels=%s, observed_labels=%s",
                    list(allowed_labels),
                    observed_labels,
                )
                return StageResult(
                    stage="prepare-data",
                    success=False,
                    message=(
                        "标签筛选后无可用记录。"
                        f"keep_labels={list(allowed_labels)}；"
                        f"当前数据中的有效标签={observed_display}"
                    ),
                )

            if spec.is_test:
                primary_stage_records = stage_records
                primary_final_records = final_records
                validation_stage_records = []
                validation_final_records = []
            else:
                (
                    primary_stage_records,
                    primary_final_records,
                    validation_stage_records,
                    validation_final_records,
                ) = self._stratified_split_pairs(
                    stage_records,
                    final_records,
                    validation_ratio=spec.validation_ratio,
                    split_seed=spec.split_seed,
                )
            logger.info(
                "记录清洗完成: retained=%s, dropped=%s, filtered_labels=%s, primary=%s, validation=%s",
                len(final_records),
                dropped_count,
                filtered_label_count,
                len(primary_final_records),
                len(validation_final_records),
            )

            self._write_json(raw_output_path, convert_result.records)
            self._write_jsonl(primary_stage_output_path, primary_stage_records)
            self._write_records(primary_final_records, primary_final_output_path, spec.output_format)

            artifacts = {
                "dataset_version_dir": layout.version_dir,
                "raw_dataset": raw_output_path,
                "stage_dataset": primary_stage_output_path,
                "final_dataset": primary_final_output_path,
                "metadata": layout.metadata_path,
            }

            validation_stage_output_path: Path | None = None
            validation_final_output_path: Path | None = None
            if not spec.is_test and validation_final_records:
                validation_stage_output_path = layout.stage_dir / f"{_VALIDATION_SPLIT_NAME}.jsonl"
                validation_final_output_path = self._resolve_final_asset_path(
                    layout.final_dir,
                    _VALIDATION_SPLIT_NAME,
                    spec.output_format,
                )
                self._write_jsonl(validation_stage_output_path, validation_stage_records)
                self._write_records(validation_final_records, validation_final_output_path, spec.output_format)
                artifacts["validation_stage_dataset"] = validation_stage_output_path
                artifacts["validation_final_dataset"] = validation_final_output_path
            elif not spec.is_test:
                self._cleanup_validation_artifacts(layout)

            if export_output_path is not None and export_output_path != primary_final_output_path:
                self._write_records(primary_final_records, export_output_path, spec.output_format)
                artifacts["export_dataset"] = export_output_path

            metadata = self._load_dataset_metadata(layout.metadata_path, spec.task_name, spec.dataset_version)
            metadata["splits"][primary_split_name] = {
                "input_path": str(spec.input_path),
                "text_key": spec.text_key,
                "label_key": spec.label_key,
                "output_format": spec.output_format,
                "sheet_name": spec.sheet_name,
                "source_row_count": convert_result.row_count,
                "retained_row_count": len(primary_final_records),
                "dropped_row_count": dropped_count,
                "filtered_label_row_count": filtered_label_count,
                "allowed_labels": list(allowed_labels),
                "observed_label_values": observed_labels,
                "retained_label_values": sorted({record["label"] for record in primary_final_records}),
                "column_names": convert_result.column_names,
                "raw_output_path": str(raw_output_path),
                "stage_output_path": str(primary_stage_output_path),
                "final_output_path": str(primary_final_output_path),
                "export_output_path": str(export_output_path) if export_output_path else None,
                "is_test": spec.is_test,
            }
            if not spec.is_test:
                metadata["train_validation_split"] = {
                    "validation_ratio": spec.validation_ratio,
                    "split_seed": spec.split_seed,
                }
                if validation_final_output_path is not None and validation_stage_output_path is not None:
                    metadata["splits"][_VALIDATION_SPLIT_NAME] = {
                        "source_input_path": str(spec.input_path),
                        "source_sheet_name": spec.sheet_name,
                        "source_split": _TRAIN_SPLIT_NAME,
                        "output_format": spec.output_format,
                        "row_count": len(validation_final_records),
                        "allowed_labels": list(allowed_labels),
                        "retained_label_values": sorted({record["label"] for record in validation_final_records}),
                        "column_names": convert_result.column_names,
                        "stage_output_path": str(validation_stage_output_path),
                        "final_output_path": str(validation_final_output_path),
                        "generated_by_validation_ratio": spec.validation_ratio,
                        "split_seed": spec.split_seed,
                    }
                else:
                    metadata["splits"].pop(_VALIDATION_SPLIT_NAME, None)
            layout.metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "数据清洗完成: version_dir=%s, final_output=%s",
                layout.version_dir,
                primary_final_output_path,
            )

            return StageResult(
                stage="prepare-data",
                success=True,
                message="数据已完成 raw/stage/final 分层，并按固定 split 更新训练集或测试集。",
                artifacts=artifacts,
            )
        except Exception as exc:  # pragma: no cover - 作为 CLI 防护分支
            logger.exception("数据处理失败")
            return StageResult(
                stage="prepare-data",
                success=False,
                message=f"数据处理失败: {exc}",
            )