"""BERT 分类评测后端。"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from tunebench.artifacts import (
    DEFAULT_EVAL_REPORT_ARTIFACT_NAMES,
    DatasetPathManager,
    ModelPathManager,
    TEST_LABEL_METRICS_ARTIFACT_NAME,
    TEST_METRICS_ARTIFACT_NAME,
    TEST_PREDICTIONS_ARTIFACT_NAME,
    get_dataset_path_manager,
    get_model_path_manager,
)
from tunebench.artifacts.eval_store import (
    EvalArtifactStore,
    FileSystemEvalArtifactStore,
)
from tunebench.classification import (
    TEST_SPLIT_NAME,
    compute_classification_metrics_bundle,
    load_classification_records,
    resolve_split_file,
    validate_classification_records,
)
from tunebench.contracts import EvalSpec, RunPlan, StageResult
from tunebench.util import get_logger


logger = get_logger("backends.bert.eval_runner")

_BERT_BACKEND = "bert"
_TEST_METRICS_FIELDNAMES = [
    "run_id",
    "split",
    "sample_count",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "avg_confidence",
    "total_latency_ms",
    "avg_latency_ms",
]
_LABEL_METRICS_FIELDNAMES = [
    "run_id",
    "stage",
    "split",
    "epoch",
    "step",
    "label",
    "support",
    "precision",
    "recall",
    "f1",
]
_TEST_PREDICTIONS_FIELDNAMES = [
    "run_id",
    "split",
    "index",
    "text",
    "true_label",
    "pred_label",
    "is_correct",
    "confidence",
]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def _build_label_metrics_rows(
    *,
    run_id: str,
    stage: str,
    split: str,
    epoch: float | None,
    step: int | None,
    id_to_label: dict[int, str],
    label_metrics: dict[int, dict[str, float | int]],
) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    for label_id in sorted(label_metrics):
        metrics = label_metrics[label_id]
        rows.append(
            {
                "run_id": run_id,
                "stage": stage,
                "split": split,
                "epoch": epoch,
                "step": step,
                "label": id_to_label.get(label_id, str(label_id)),
                "support": int(metrics.get("support", 0)),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
            }
        )
    return rows


class BertClassificationEvalRunner:
    """负责当前 BERT 分类独立评测与结果导出。"""

    def __init__(
        self,
        dataset_path_manager: DatasetPathManager | None = None,
        model_path_manager: ModelPathManager | None = None,
        artifact_store: EvalArtifactStore | None = None,
    ) -> None:
        self.dataset_path_manager = dataset_path_manager or get_dataset_path_manager()
        self.model_path_manager = model_path_manager or get_model_path_manager()
        self.artifact_store = artifact_store or FileSystemEvalArtifactStore()

    def _load_metadata(self, metadata_path: Path) -> dict[str, Any]:
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _build_dataset(
        self, records: list[dict[str, str]], label_to_id: dict[str, int]
    ) -> Dataset:
        normalized_records = []
        for index, record in enumerate(records):
            label_value = record["label"]
            normalized_records.append(
                {
                    "index": index,
                    "text": record["text"],
                    "label": label_to_id[label_value],
                    "label_name": label_value,
                }
            )
        return Dataset.from_list(normalized_records)

    def _tokenize_dataset(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_sequence_length: int | None = None,
    ) -> Dataset:
        return dataset.map(
            lambda batch: tokenizer(
                batch["text"], truncation=True, max_length=max_sequence_length
            ),
            batched=True,
            desc="tokenizing-eval",
        )

    def _resolve_tokenizer_source(self, model_dir: Path, model_name: str) -> str:
        if (model_dir / "tokenizer_config.json").exists():
            return str(model_dir)
        return model_name

    def _load_model_and_tokenizer(
        self,
        spec: EvalSpec,
        metadata: dict[str, Any],
        label_to_id: dict[str, int],
    ) -> tuple[Any, PreTrainedTokenizerBase]:
        model_name = str(metadata["model_name"])
        model_layout = self.model_path_manager.build_layout(
            _BERT_BACKEND, spec.task_name, spec.run_id
        )

        if spec.artifact_type == "merged":
            tokenizer = cast(
                PreTrainedTokenizerBase,
                AutoTokenizer.from_pretrained(
                    self._resolve_tokenizer_source(
                        model_layout.merged_model_dir, model_name
                    ),
                    use_fast=True,
                ),
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_layout.merged_model_dir)
            )
            return model, tokenizer

        if spec.artifact_type == "lora":
            tokenizer = cast(
                PreTrainedTokenizerBase,
                AutoTokenizer.from_pretrained(
                    self._resolve_tokenizer_source(model_layout.lora_dir, model_name),
                    use_fast=True,
                ),
            )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(label_to_id),
                id2label={index: label for label, index in label_to_id.items()},
                label2id=label_to_id,
            )
            model = PeftModel.from_pretrained(base_model, str(model_layout.lora_dir))
            return model, tokenizer

        raise ValueError(
            f"artifact_type={spec.artifact_type} 非法，仅支持 merged 或 lora。"
        )

    def build_plan(self, spec: EvalSpec) -> RunPlan:
        model_layout = self.model_path_manager.build_layout(
            _BERT_BACKEND, spec.task_name, spec.run_id
        )
        return RunPlan(
            stage="evaluate",
            summary="执行 BERT 模型评测并生成结果摘要。",
            inputs=asdict(spec),
            outputs={
                "eval_dir": str(model_layout.eval_dir),
                "test_metrics_csv": str(model_layout.test_metrics_csv),
                "test_label_metrics_csv": str(model_layout.test_label_metrics_csv),
                "test_predictions_csv": str(model_layout.test_predictions_csv),
                "eval_report_xlsx": str(model_layout.eval_report_xlsx)
                if spec.export_xlsx
                else None,
            },
            notes=[
                "评测结果默认写入当前 run 下的 eval 目录。",
                f"默认会将 {'/'.join(DEFAULT_EVAL_REPORT_ARTIFACT_NAMES)} 汇总为 {model_layout.eval_report_xlsx.name}。",
                "训练期指标和独立评测指标都会以 CSV 形式持久化。",
            ],
        )

    def run(self, spec: EvalSpec) -> StageResult:
        try:
            logger.info(
                "开始评测: task=%s, run_id=%s, dataset_version=%s, split=%s, artifact_type=%s",
                spec.task_name,
                spec.run_id,
                spec.dataset_version,
                TEST_SPLIT_NAME,
                spec.artifact_type,
            )
            model_layout = self.model_path_manager.ensure_layout(
                _BERT_BACKEND, spec.task_name, spec.run_id
            )
            dataset_layout = self.dataset_path_manager.build_layout(
                spec.task_name, spec.dataset_version
            )
            metadata = self._load_metadata(model_layout.metadata_path)
            label_to_id = {
                str(key): int(value) for key, value in metadata["label_to_id"].items()
            }
            id_to_label = {value: key for key, value in label_to_id.items()}

            split_file = resolve_split_file(dataset_layout.final_dir, TEST_SPLIT_NAME)
            records = validate_classification_records(
                load_classification_records(split_file), TEST_SPLIT_NAME
            )
            model, tokenizer = self._load_model_and_tokenizer(
                spec, metadata, label_to_id
            )
            dataset = self._build_dataset(records, label_to_id)
            tokenized_dataset = self._tokenize_dataset(
                dataset, tokenizer, spec.max_sequence_length
            )

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=str(model_layout.eval_dir / "tmp_eval"),
                    per_device_eval_batch_size=spec.batch_size,
                    report_to=[],
                    remove_unused_columns=True,
                ),
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            )

            started_at = time.perf_counter()
            prediction_output = trainer.predict(cast(Any, tokenized_dataset))
            total_latency_ms = (time.perf_counter() - started_at) * 1000.0

            logits = np.asarray(prediction_output.predictions)
            if isinstance(prediction_output.predictions, tuple):
                logits = np.asarray(prediction_output.predictions[0])
            labels = np.asarray(prediction_output.label_ids)
            metrics, label_metrics = compute_classification_metrics_bundle(
                logits, labels
            )
            probabilities = _softmax(logits)
            predictions = np.argmax(probabilities, axis=-1)

            metrics_row = {
                "run_id": spec.run_id,
                "split": TEST_SPLIT_NAME,
                "sample_count": len(records),
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "avg_confidence": metrics["avg_confidence"],
                "total_latency_ms": total_latency_ms,
                "avg_latency_ms": total_latency_ms / len(records) if records else 0.0,
            }
            prediction_rows = [
                {
                    "run_id": spec.run_id,
                    "split": TEST_SPLIT_NAME,
                    "index": index,
                    "text": record["text"],
                    "true_label": record["label"],
                    "pred_label": id_to_label[int(predictions[index])],
                    "is_correct": record["label"]
                    == id_to_label[int(predictions[index])],
                    "confidence": float(np.max(probabilities[index])),
                }
                for index, record in enumerate(records)
            ]

            self.artifact_store.append_artifact_rows(
                model_layout=model_layout,
                artifact_name=TEST_METRICS_ARTIFACT_NAME,
                fieldnames=_TEST_METRICS_FIELDNAMES,
                rows=[metrics_row],
            )
            self.artifact_store.append_artifact_rows(
                model_layout=model_layout,
                artifact_name=TEST_LABEL_METRICS_ARTIFACT_NAME,
                fieldnames=_LABEL_METRICS_FIELDNAMES,
                rows=_build_label_metrics_rows(
                    run_id=spec.run_id,
                    stage="evaluate",
                    split=TEST_SPLIT_NAME,
                    epoch=None,
                    step=None,
                    id_to_label=id_to_label,
                    label_metrics=label_metrics,
                ),
            )
            self.artifact_store.append_artifact_rows(
                model_layout=model_layout,
                artifact_name=TEST_PREDICTIONS_ARTIFACT_NAME,
                fieldnames=_TEST_PREDICTIONS_FIELDNAMES,
                rows=prediction_rows,
            )

            artifacts = {
                "eval_dir": model_layout.eval_dir,
                "test_metrics_csv": model_layout.test_metrics_csv,
                "test_label_metrics_csv": model_layout.test_label_metrics_csv,
                "test_predictions_csv": model_layout.test_predictions_csv,
            }
            message = "评测完成，结果已写入 CSV 文件。"
            if spec.export_xlsx:
                try:
                    export_result = self.artifact_store.export_eval_report(model_layout)
                    if export_result.output_path is not None:
                        artifacts["eval_report_xlsx"] = export_result.output_path
                        message = "评测完成，结果已写入 CSV 文件并汇总为 XLSX。"
                    for warning_message in export_result.warnings:
                        logger.warning("XLSX 汇总导出提示: %s", warning_message)
                except Exception as export_exc:
                    logger.warning("XLSX 汇总导出失败: %s", export_exc)
                    message = (
                        f"评测完成，CSV 已生成，但 XLSX 汇总导出失败: {export_exc}"
                    )

            return StageResult(
                stage="evaluate",
                success=True,
                message=message,
                artifacts=artifacts,
                metrics={
                    key: float(value)
                    for key, value in metrics_row.items()
                    if isinstance(value, (int, float))
                },
            )
        except Exception as exc:
            logger.exception("BERT 评测失败")
            return StageResult(
                stage="evaluate",
                success=False,
                message=f"评测失败: {exc}",
            )
