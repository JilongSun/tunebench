"""训练 run metadata schema。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """描述训练数据集规模。"""

    train_examples: int
    validation_examples: int


@dataclass(frozen=True, slots=True)
class TrainingHyperparameters:
    """描述一次训练的核心超参数。"""

    learning_rate: float
    batch_size: int
    num_train_epochs: int
    max_sequence_length: int
    warmup_ratio: float
    seed: int


@dataclass(frozen=True, slots=True)
class ClassificationTrainRunMetadata:
    """描述分类训练 run 的统一 metadata 结构。"""

    backend: str
    task_name: str
    dataset_version: str
    model_name: str | None
    model_key: str | None
    reasoning_mode: str | None
    run_id: str
    output_dir: str
    train_file: str
    validation_file: str | None
    num_labels: int
    label_names: list[str]
    label_to_id: dict[str, int]
    dataset_stats: DatasetStats
    device: str
    hyperparameters: TrainingHyperparameters
    backend_config: dict[str, Any]
    train_metrics: dict[str, float] | None = None
    eval_metrics: dict[str, float] | None = None
    instruction: str | None = None
    status: str = "prepared"


def serialize_classification_train_run_metadata(metadata: ClassificationTrainRunMetadata) -> dict[str, Any]:
    """序列化训练 metadata。"""
    return asdict(metadata)


__all__ = [
    "ClassificationTrainRunMetadata",
    "DatasetStats",
    "TrainingHyperparameters",
    "serialize_classification_train_run_metadata",
]