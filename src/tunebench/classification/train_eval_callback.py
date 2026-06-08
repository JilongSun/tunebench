"""分类训练过程中的评测回调。"""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback

from tunebench.artifacts import (
    EvalArtifactStore,
    FileSystemEvalArtifactStore,
    ModelArtifactLayout,
    TRAIN_METRICS_ARTIFACT_NAME,
    # VALIDATION_LABEL_METRICS_ARTIFACT_NAME,  # 暂时禁用
)
from tunebench.classification.metrics import extract_label_metrics_from_flattened


_TRAIN_METRICS_FIELDNAMES = [
    "stage",
    "split",
    "epoch",
    "step",
    "loss",
    "precision_macro",
    "recall_macro",
    "f1_macro",
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


def _build_train_metrics_row(
    *,
    stage: str,
    split: str,
    epoch: float | None,
    step: int | None,
    loss: float | None,
    precision_macro: float | None,
    recall_macro: float | None,
    f1_macro: float | None,
) -> dict[str, float | int | str | None]:
    return {
        "stage": stage,
        "split": split,
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


class ClassificationTrainEvalCallback(TrainerCallback):
    """训练阶段将 loss 与 validation 指标落到 CSV。"""

    def __init__(
        self,
        model_layout: ModelArtifactLayout,
        run_id: str,
        id_to_label: dict[int, str],
        artifact_store: EvalArtifactStore | None = None,
    ) -> None:
        self.model_layout = model_layout
        self.run_id = run_id
        self.id_to_label = id_to_label
        self.artifact_store = artifact_store or FileSystemEvalArtifactStore()
        self._suppress_next_evaluate = False

    def _append_label_metric_rows(
        self,
        *,
        stage: str,
        split: str,
        epoch: float | None,
        step: int | None,
        metrics: dict[str, Any],
    ) -> None:
        label_metrics = extract_label_metrics_from_flattened(metrics)
        if not label_metrics:
            return
        # 暂时禁用 validation_label_metrics 落盘（validation 指标已包含在 train_metrics 中）
        # self.artifact_store.append_artifact_rows(
        #     model_layout=self.model_layout,
        #     artifact_name=VALIDATION_LABEL_METRICS_ARTIFACT_NAME,
        #     fieldnames=_LABEL_METRICS_FIELDNAMES,
        #     rows=_build_label_metrics_rows(
        #         run_id=self.run_id,
        #         stage=stage,
        #         split=split,
        #         epoch=epoch,
        #         step=step,
        #         id_to_label=self.id_to_label,
        #         label_metrics=label_metrics,
        #     ),
        # )

    def suppress_next_evaluate_artifact(self) -> None:
        """跳过下一次 on_evaluate 落盘，用于显式最终评估。"""
        self._suppress_next_evaluate = True

    def append_final_evaluate_summary(self, metrics: dict[str, Any]) -> None:
        """将训练结束后的最终评估写为独立 summary 行。"""
        self._append_label_metric_rows(
            stage="final_evaluate",
            split="summary",
            epoch=None,
            step=None,
            metrics=metrics,
        )
        self.artifact_store.append_artifact_rows(
            model_layout=self.model_layout,
            artifact_name=TRAIN_METRICS_ARTIFACT_NAME,
            fieldnames=_TRAIN_METRICS_FIELDNAMES,
            rows=[
                _build_train_metrics_row(
                    stage="final_evaluate",
                    split="summary",
                    epoch=None,
                    step=None,
                    loss=metrics.get("eval_loss"),
                    precision_macro=metrics.get("eval_precision_macro"),
                    recall_macro=metrics.get("eval_recall_macro"),
                    f1_macro=metrics.get("eval_f1_macro"),
                )
            ],
        )

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
        if not logs or "loss" not in logs or "eval_loss" in logs:
            return
        self.artifact_store.append_artifact_rows(
            model_layout=self.model_layout,
            artifact_name=TRAIN_METRICS_ARTIFACT_NAME,
            fieldnames=_TRAIN_METRICS_FIELDNAMES,
            rows=[
                _build_train_metrics_row(
                    stage="train",
                    split="train",
                    epoch=float(state.epoch) if state.epoch is not None else None,
                    step=state.global_step,
                    loss=logs.get("loss"),
                    precision_macro=None,
                    recall_macro=None,
                    f1_macro=None,
                )
            ],
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # type: ignore[no-untyped-def]
        if not metrics:
            return
        if self._suppress_next_evaluate:
            self._suppress_next_evaluate = False
            return

        epoch = float(state.epoch) if state.epoch is not None else None
        step = state.global_step

        self._append_label_metric_rows(
            stage="epoch_evaluate",
            split="validation",
            epoch=epoch,
            step=step,
            metrics=metrics,
        )

        self.artifact_store.append_artifact_rows(
            model_layout=self.model_layout,
            artifact_name=TRAIN_METRICS_ARTIFACT_NAME,
            fieldnames=_TRAIN_METRICS_FIELDNAMES,
            rows=[
                _build_train_metrics_row(
                    stage="epoch_evaluate",
                    split="validation",
                    epoch=epoch,
                    step=step,
                    loss=metrics.get("eval_loss"),
                    precision_macro=metrics.get("eval_precision_macro"),
                    recall_macro=metrics.get("eval_recall_macro"),
                    f1_macro=metrics.get("eval_f1_macro"),
                )
            ],
        )