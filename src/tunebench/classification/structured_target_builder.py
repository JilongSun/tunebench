"""分类结构化 target 构建执行器。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tunebench.artifacts import DatasetPathManager, get_dataset_path_manager
from tunebench.contracts import RunPlan, StageResult, StructuredTargetBuildSpec
from tunebench.util import get_logger

from .dataset_loader import (
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
    VALIDATION_SPLIT_NAME,
    build_label_mapping,
    load_classification_records,
    resolve_optional_split_file,
    resolve_split_file,
    validate_num_labels,
    validate_validation_labels,
)


logger = get_logger("classification.structured_target_builder")

_STAGE_NAME = "build-structured-target"
_STAGE_FILE_SUFFIX = ".jsonl"
_SUPPORTED_CONFIDENCES = {0.3, 0.6, 0.9}


@dataclass(slots=True)
class StructuredTargetDatasetBundle:
    """描述带结构化 target 的训练数据包。"""

    train_file: Path
    validation_file: Path | None
    train_records: list[dict[str, Any]]
    validation_records: list[dict[str, Any]] | None
    label_to_id: dict[str, int]


class ClassificationStructuredTargetBuilder:
    """将 reasoning 增强数据转换为固定结构化 target。"""

    def __init__(self, dataset_path_manager: DatasetPathManager | None = None) -> None:
        self.dataset_path_manager = dataset_path_manager or get_dataset_path_manager()

    def build_plan(self, spec: StructuredTargetBuildSpec) -> RunPlan:
        """生成 structured target 构建计划。"""
        source_layout = self.dataset_path_manager.build_layout(spec.task_name, spec.source_dataset_version)
        target_layout = self.dataset_path_manager.build_layout(spec.task_name, spec.target_dataset_version)
        outputs: dict[str, str] = {
            "source_dataset_version_dir": str(source_layout.version_dir),
            "target_dataset_version_dir": str(target_layout.version_dir),
            "metadata": str(target_layout.metadata_path),
        }
        for split_name in spec.splits:
            outputs[f"stage_{split_name}"] = str(target_layout.stage_dir / f"{split_name}{_STAGE_FILE_SUFFIX}")
            outputs[f"final_{split_name}"] = str(target_layout.final_dir / f"{split_name}.jsonl")
        return RunPlan(
            stage=_STAGE_NAME,
            summary="将 reasoning 数据转换为固定结构化 target。",
            inputs=asdict(spec),
            outputs=outputs,
            notes=[
                "source_dataset_version 与 target_dataset_version 必须不同。",
                "每条 accepted 记录会生成带 target 字段的 final 输出。",
                "confidence 目前只支持 0.3、0.6、0.9。",
            ],
        )

    def run(self, spec: StructuredTargetBuildSpec) -> StageResult:
        """执行结构化 target 构建逻辑。"""
        try:
            self._validate_spec(spec)
            return self._run(spec)
        except Exception as exc:  # pragma: no cover - CLI 防护分支
            logger.exception("结构化 target 构建失败")
            return StageResult(
                stage=_STAGE_NAME,
                success=False,
                message=f"结构化 target 构建失败: {exc}",
            )

    def _run(self, spec: StructuredTargetBuildSpec) -> StageResult:
        source_layout = self.dataset_path_manager.build_layout(spec.task_name, spec.source_dataset_version)
        target_layout = self.dataset_path_manager.ensure_layout(spec.task_name, spec.target_dataset_version)

        artifacts: dict[str, Path] = {
            "dataset_version_dir": target_layout.version_dir,
            "metadata": target_layout.metadata_path,
        }
        metrics: dict[str, float] = {}
        metadata = {
            "task_name": spec.task_name,
            "dataset_version": spec.target_dataset_version,
            "source_dataset_version": spec.source_dataset_version,
            "stage": _STAGE_NAME,
            "confidence": spec.confidence,
            "final_schema": {
                "text": "string",
                "label": "string",
                "reasoning": "string",
                "target": "object",
            },
            "splits": {},
        }

        for split_name in spec.splits:
            source_file = _resolve_source_split_file(source_layout.final_dir, split_name)
            if source_file is None:
                logger.info("跳过缺失 split: %s", split_name)
                continue

            split_result = self._process_split(
                spec=spec,
                split_name=split_name,
                source_file=source_file,
                target_layout=target_layout,
            )
            metadata["splits"][split_name] = split_result["metadata"]
            metrics.update(split_result["metrics"])
            artifacts[f"stage_{split_name}"] = split_result["stage_path"]
            artifacts[f"final_{split_name}"] = split_result["final_path"]

        target_layout.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return StageResult(
            stage=_STAGE_NAME,
            success=True,
            message="结构化 target 已构建完成，新的 stage/final 数据版本已生成。",
            artifacts=artifacts,
            metrics=metrics,
        )

    def _process_split(
        self,
        *,
        spec: StructuredTargetBuildSpec,
        split_name: str,
        source_file: Path,
        target_layout: Any,
    ) -> dict[str, Any]:
        source_records = validate_reasoning_records(load_classification_records(source_file), split_name)
        final_records = [_build_structured_record(record, spec.confidence) for record in source_records]
        stage_records = [
            {
                "source_index": index,
                "split": split_name,
                "text": record["text"],
                "label": record["label"],
                "reasoning": record["reasoning"],
                "target": record["target"],
                "target_output": json.dumps(record["target"], ensure_ascii=False),
                "status": "accepted",
            }
            for index, record in enumerate(final_records, start=1)
        ]

        output_format = "jsonl" if source_file.suffix == ".jsonl" else "json"
        stage_path = target_layout.stage_dir / f"{split_name}{_STAGE_FILE_SUFFIX}"
        final_path = target_layout.final_dir / source_file.name

        _write_jsonl(stage_path, stage_records)
        _write_records(final_path, final_records, output_format)

        return {
            "stage_path": stage_path,
            "final_path": final_path,
            "metadata": {
                "source_dataset_version": spec.source_dataset_version,
                "source_input_path": str(source_file),
                "output_format": output_format,
                "row_count": len(final_records),
                "accepted_row_count": len(final_records),
                "stage_output_path": str(stage_path),
                "final_output_path": str(final_path),
            },
            "metrics": {
                f"{split_name}_row_count": float(len(final_records)),
                f"{split_name}_accepted_count": float(len(final_records)),
            },
        }

    def _validate_spec(self, spec: StructuredTargetBuildSpec) -> None:
        if spec.source_dataset_version == spec.target_dataset_version:
            raise ValueError("source_dataset_version 与 target_dataset_version 不能相同。")
        if spec.confidence not in _SUPPORTED_CONFIDENCES:
            raise ValueError(f"confidence 仅支持 {_SUPPORTED_CONFIDENCES}。")
        unsupported_splits = [split for split in spec.splits if split not in {TRAIN_SPLIT_NAME, VALIDATION_SPLIT_NAME, TEST_SPLIT_NAME}]
        if unsupported_splits:
            raise ValueError(f"存在不支持的 split: {unsupported_splits}")
        if not spec.splits:
            raise ValueError("至少需要指定一个 split。")


def validate_reasoning_records(records: list[dict[str, Any]], split_name: str) -> list[dict[str, str]]:
    """校验 reasoning 增强后的 final 层记录结构。"""
    if not records:
        raise ValueError(f"split={split_name} 的数据为空，无法继续。")

    normalized_records: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"split={split_name} 第 {index} 条记录不是对象。")

        text_value = str(record.get("text", "")).strip()
        label_value = str(record.get("label", "")).strip()
        status_value = str(record.get("status", "accepted")).strip()
        reasoning_raw_value = record.get("reasoning")
        reasoning_value = str(reasoning_raw_value).strip() if isinstance(reasoning_raw_value, str) else ""
        if not text_value or not label_value:
            raise ValueError(f"split={split_name} 第 {index} 条记录缺少非空 text/label。")
        if status_value not in {"accepted", "rejected"}:
            raise ValueError(f"split={split_name} 第 {index} 条记录存在非法 status: {status_value}")
        if status_value == "rejected":
            continue
        if not reasoning_value:
            raise ValueError(f"split={split_name} 第 {index} 条 accepted 记录缺少非空 reasoning。")

        normalized_records.append(
            {
                "text": text_value,
                "label": label_value,
                "reasoning": reasoning_value,
            }
        )

    if not normalized_records:
        raise ValueError(f"split={split_name} 没有可用于后续阶段的 accepted reasoning 记录。")
    return normalized_records


def load_structured_target_dataset_bundle(
    dataset_path_manager: DatasetPathManager,
    task_name: str,
    dataset_version: str,
    num_labels: int | None = None,
) -> StructuredTargetDatasetBundle:
    """加载带结构化 target 的训练数据包。"""
    dataset_layout = dataset_path_manager.build_layout(task_name, dataset_version)
    train_file = resolve_split_file(dataset_layout.final_dir, TRAIN_SPLIT_NAME)
    train_records = validate_structured_target_records(load_classification_records(train_file), TRAIN_SPLIT_NAME)

    validation_file = resolve_optional_split_file(dataset_layout.final_dir, VALIDATION_SPLIT_NAME)
    validation_records: list[dict[str, Any]] | None = None
    if validation_file is not None:
        validation_records = validate_structured_target_records(
            load_classification_records(validation_file),
            VALIDATION_SPLIT_NAME,
        )

    label_to_id = build_label_mapping(train_records)
    validate_num_labels(num_labels, label_to_id)
    validate_validation_labels(validation_records, label_to_id)
    return StructuredTargetDatasetBundle(
        train_file=train_file,
        validation_file=validation_file,
        train_records=train_records,
        validation_records=validation_records,
        label_to_id=label_to_id,
    )


def validate_structured_target_records(records: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    """校验结构化 target final 层记录结构。"""
    reasoning_records = validate_reasoning_records(records, split_name)
    normalized_records: list[dict[str, Any]] = []
    for index, (source_record, normalized_reasoning_record) in enumerate(zip(records, reasoning_records, strict=False), start=1):
        target = source_record.get("target")
        if not isinstance(target, dict):
            raise ValueError(f"split={split_name} 第 {index} 条记录缺少 target 对象。")
        normalized_records.append({**normalized_reasoning_record, "target": target})
    return normalized_records


def _resolve_source_split_file(final_dir: Path, split_name: str) -> Path | None:
    if split_name == TRAIN_SPLIT_NAME:
        return resolve_split_file(final_dir, split_name)
    return resolve_optional_split_file(final_dir, split_name)


def _build_structured_record(record: dict[str, str], confidence: float) -> dict[str, Any]:
    target = {
        "reasoning": record["reasoning"],
        "intents": [
            {
                "intent": [record["label"]],
                "confidence": confidence,
                "follow_up_question": "",
            }
        ],
        "intent_relations": None,
    }
    return {
        "text": record["text"],
        "label": record["label"],
        "reasoning": record["reasoning"],
        "target": target,
    }


def _write_jsonl(output_path: Path, records: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_records(output_path: Path, records: list[dict[str, Any]], output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


__all__ = [
    "ClassificationStructuredTargetBuilder",
    "StructuredTargetDatasetBundle",
    "load_structured_target_dataset_bundle",
    "validate_reasoning_records",
    "validate_structured_target_records",
]