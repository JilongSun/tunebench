"""分类任务运行清单构建工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tunebench.artifacts.run_metadata import (
    ClassificationTrainRunMetadata,
    DatasetStats,
    TrainingHyperparameters,
    serialize_classification_train_run_metadata,
)
from tunebench.contracts import TrainSpec


def build_classification_train_manifest(
    *,
    spec: TrainSpec,
    run_id: str,
    train_file: Path,
    validation_file: Path | None,
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]] | None,
    label_to_id: dict[str, int],
    output_dir: Path,
    device: str,
    backend_config: dict[str, Any],
    instruction: str | None = None,
    train_result: dict[str, float] | None = None,
    eval_result: dict[str, float] | None = None,
) -> dict[str, Any]:
    """构建统一的分类训练元数据。"""
    metadata = ClassificationTrainRunMetadata(
        backend=spec.backend,
        task_name=spec.task_name,
        dataset_version=spec.dataset_version,
        model_name=spec.model_name,
        model_key=spec.model_key,
        reasoning_mode=spec.reasoning_mode,
        run_id=run_id,
        output_dir=str(output_dir),
        train_file=str(train_file),
        validation_file=str(validation_file) if validation_file else None,
        num_labels=len(label_to_id),
        label_names=list(label_to_id.keys()),
        label_to_id=label_to_id,
        dataset_stats=DatasetStats(
            train_examples=len(train_records),
            validation_examples=len(validation_records) if validation_records is not None else 0,
        ),
        device=device,
        hyperparameters=TrainingHyperparameters(
            learning_rate=spec.learning_rate,
            batch_size=spec.batch_size,
            num_train_epochs=spec.num_train_epochs,
            max_sequence_length=spec.max_sequence_length,
            warmup_ratio=spec.warmup_ratio,
            seed=spec.seed,
        ),
        backend_config=backend_config,
        train_metrics=train_result,
        eval_metrics=eval_result,
        instruction=instruction,
        status="completed" if train_result is not None else "prepared",
    )
    return serialize_classification_train_run_metadata(metadata)