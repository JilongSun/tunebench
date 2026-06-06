"""分类任务数据加载与校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tunebench.artifacts import DatasetPathManager


TRAIN_SPLIT_NAME = "train"
VALIDATION_SPLIT_NAME = "validation"
TEST_SPLIT_NAME = "test"


@dataclass(slots=True)
class ClassificationDatasetBundle:
    """描述一次训练所需的分类数据包。"""

    train_file: Path
    validation_file: Path | None
    train_records: list[dict[str, Any]]
    validation_records: list[dict[str, Any]] | None
    label_to_id: dict[str, int]

    @property
    def id_to_label(self) -> dict[int, str]:
        """返回反向标签映射。"""
        return {index: label for label, index in self.label_to_id.items()}


def resolve_split_file(final_dir: Path, split_name: str) -> Path:
    """解析必需的 split 文件路径。"""
    candidates = [final_dir / f"{split_name}.jsonl", final_dir / f"{split_name}.json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"未找到 split={split_name} 的标准化数据文件，已检查: {searched}")


def resolve_optional_split_file(final_dir: Path, split_name: str) -> Path | None:
    """解析可选的 split 文件路径。"""
    candidates = [final_dir / f"{split_name}.jsonl", final_dir / f"{split_name}.json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_classification_records(input_path: Path) -> list[dict[str, Any]]:
    """读取标准化 JSON 或 JSONL 数据。"""
    content = input_path.read_text(encoding="utf-8")
    if input_path.suffix == ".jsonl":
        lines = [line for line in content.splitlines() if line.strip()]
        return [json.loads(line) for line in lines]

    payload = json.loads(content)
    if not isinstance(payload, list):
        raise ValueError(f"标准化数据文件必须是记录数组: {input_path}")
    return payload


def validate_classification_records(
    records: list[dict[str, Any]],
    split_name: str,
) -> list[dict[str, Any]]:
    """校验分类任务的 final 层记录结构。"""
    if not records:
        raise ValueError(f"split={split_name} 的数据为空，无法继续。")

    normalized_records: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"split={split_name} 第 {index} 条记录不是对象。")

        text_value = str(record.get("text", "")).strip()
        label_value = str(record.get("label", "")).strip()
        if not text_value or not label_value:
            raise ValueError(f"split={split_name} 第 {index} 条记录缺少非空 text/label。")

        normalized_record: dict[str, Any] = {"text": text_value, "label": label_value}
        reasoning_value = record.get("reasoning")
        if reasoning_value is not None:
            normalized_reasoning_value = str(reasoning_value).strip()
            if not normalized_reasoning_value:
                raise ValueError(f"split={split_name} 第 {index} 条记录的 reasoning 不能为空字符串。")
            normalized_record["reasoning"] = normalized_reasoning_value

        target_value = record.get("target")
        if target_value is not None:
            if not isinstance(target_value, dict):
                raise ValueError(f"split={split_name} 第 {index} 条记录的 target 必须是对象。")
            normalized_record["target"] = target_value

        normalized_records.append(normalized_record)

    return normalized_records


def build_label_mapping(train_records: list[dict[str, Any]]) -> dict[str, int]:
    """基于训练集构建标签映射。"""
    label_names = sorted({record["label"] for record in train_records})
    return {label_name: index for index, label_name in enumerate(label_names)}


def validate_num_labels(num_labels: int | None, label_to_id: dict[str, int]) -> None:
    """校验显式传入的标签数量。"""
    if num_labels is None:
        return
    if num_labels != len(label_to_id):
        label_names = list(label_to_id.keys())
        raise ValueError(
            "num_labels 与训练数据标签空间不一致。"
            f"期望 num_labels={num_labels}；"
            f"训练数据实际标签数={len(label_to_id)}；"
            f"训练数据标签列表={label_names}"
        )


def validate_validation_labels(
    validation_records: list[dict[str, Any]] | None,
    label_to_id: dict[str, int],
) -> None:
    """校验验证集标签是否都出现在训练集中。"""
    if validation_records is None:
        return

    unknown_labels = sorted({record["label"] for record in validation_records if record["label"] not in label_to_id})
    if unknown_labels:
        unknown_display = ", ".join(unknown_labels)
        train_label_names = list(label_to_id.keys())
        raise ValueError(
            "验证集存在训练集中未出现的标签。"
            f"未知标签={unknown_display}；"
            f"训练集标签数={len(label_to_id)}；"
            f"训练集标签列表={train_label_names}"
        )


def load_training_dataset_bundle(
    dataset_path_manager: DatasetPathManager,
    task_name: str,
    dataset_version: str,
    num_labels: int | None = None,
) -> ClassificationDatasetBundle:
    """加载训练与验证所需的分类数据包。"""
    dataset_layout = dataset_path_manager.build_layout(task_name, dataset_version)
    train_file = resolve_split_file(dataset_layout.final_dir, TRAIN_SPLIT_NAME)
    train_records = validate_classification_records(load_classification_records(train_file), TRAIN_SPLIT_NAME)

    validation_file = resolve_optional_split_file(dataset_layout.final_dir, VALIDATION_SPLIT_NAME)
    validation_records: list[dict[str, str]] | None = None
    if validation_file is not None:
        validation_records = validate_classification_records(
            load_classification_records(validation_file),
            VALIDATION_SPLIT_NAME,
        )

    label_to_id = build_label_mapping(train_records)
    validate_num_labels(num_labels, label_to_id)
    validate_validation_labels(validation_records, label_to_id)
    return ClassificationDatasetBundle(
        train_file=train_file,
        validation_file=validation_file,
        train_records=train_records,
        validation_records=validation_records,
        label_to_id=label_to_id,
    )