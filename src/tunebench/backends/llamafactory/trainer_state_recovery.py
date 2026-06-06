"""LlamaFactory trainer_state 回收工具。"""

from __future__ import annotations

import json

from tunebench.artifacts import EvalArtifactStore, ModelArtifactLayout, TRAIN_METRICS_ARTIFACT_NAME
from tunebench.classification import VALIDATION_SPLIT_NAME
from tunebench.util import get_logger


logger = get_logger("backends.llamafactory.trainer_state_recovery")

_TRAINER_STATE_FILENAME = "trainer_state.json"
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


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


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


def load_trainer_state_payload(model_layout: ModelArtifactLayout) -> dict[str, object] | None:
    """读取 checkpoints 目录中的 trainer_state。"""
    trainer_state_path = model_layout.checkpoints_dir / _TRAINER_STATE_FILENAME
    if not trainer_state_path.exists():
        logger.warning("跳过 trainer_state 回收：未找到 %s", trainer_state_path)
        return None

    payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        logger.warning("跳过 trainer_state 回收：%s 内容类型无效。", trainer_state_path)
        return None
    return payload


def build_train_result(
    *,
    model_layout: ModelArtifactLayout,
    total_runtime_seconds: float,
    train_command_runtime_seconds: float,
    export_command_runtime_seconds: float,
    train_history_recovered: bool,
) -> dict[str, float]:
    """基于运行时信息与 trainer_state 构建训练结果摘要。"""
    train_result: dict[str, float] = {
        "total_runtime_seconds": round(total_runtime_seconds, 4),
        "train_command_runtime_seconds": round(train_command_runtime_seconds, 4),
        "export_command_runtime_seconds": round(export_command_runtime_seconds, 4),
        "train_history_recovered": 1.0 if train_history_recovered else 0.0,
    }

    payload = load_trainer_state_payload(model_layout)
    if payload is None:
        return train_result

    top_level_numeric_fields = {
        "max_steps": payload.get("max_steps"),
        "num_train_epochs": payload.get("num_train_epochs"),
        "train_batch_size": payload.get("train_batch_size"),
        "total_flos": payload.get("total_flos"),
        "num_input_tokens_seen": payload.get("num_input_tokens_seen"),
    }
    for field_name, raw_value in top_level_numeric_fields.items():
        numeric_value = _coerce_float(raw_value)
        if numeric_value is not None:
            train_result[field_name] = numeric_value

    raw_history = payload.get("log_history")
    if not isinstance(raw_history, list):
        return train_result

    final_train_summary = next(
        (
            item
            for item in reversed(raw_history)
            if isinstance(item, dict) and _coerce_float(item.get("train_runtime")) is not None
        ),
        None,
    )
    if final_train_summary is None:
        return train_result

    history_numeric_fields = {
        "epoch": final_train_summary.get("epoch"),
        "step": final_train_summary.get("step"),
        "train_runtime": final_train_summary.get("train_runtime"),
        "train_samples_per_second": final_train_summary.get("train_samples_per_second"),
        "train_steps_per_second": final_train_summary.get("train_steps_per_second"),
        "train_loss": final_train_summary.get("train_loss"),
        "total_flos": final_train_summary.get("total_flos"),
    }
    for field_name, raw_value in history_numeric_fields.items():
        numeric_value = _coerce_float(raw_value)
        if numeric_value is not None:
            train_result[field_name] = numeric_value
    return train_result


def recover_trainer_history_metrics(
    *,
    artifact_store: EvalArtifactStore,
    model_layout: ModelArtifactLayout,
) -> bool:
    """从 trainer_state.log_history 回收训练与逐 epoch 验证指标。"""
    payload = load_trainer_state_payload(model_layout)
    if payload is None:
        return False

    trainer_state_path = model_layout.checkpoints_dir / _TRAINER_STATE_FILENAME
    raw_history = payload.get("log_history")
    if not isinstance(raw_history, list):
        logger.warning("跳过训练日志回收：trainer_state.log_history 缺失或类型无效。")
        return False

    rows: list[dict[str, float | int | str | None]] = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue

        epoch = _coerce_float(item.get("epoch"))
        step = _coerce_int(item.get("step"))
        train_loss = _coerce_float(item.get("loss"))
        if train_loss is not None and "eval_loss" not in item:
            rows.append(
                _build_train_metrics_row(
                    stage="train",
                    split="train",
                    epoch=epoch,
                    step=step,
                    loss=train_loss,
                    precision_macro=None,
                    recall_macro=None,
                    f1_macro=None,
                )
            )

        eval_loss = _coerce_float(item.get("eval_loss"))
        if eval_loss is not None:
            rows.append(
                _build_train_metrics_row(
                    stage="epoch_evaluate",
                    split=VALIDATION_SPLIT_NAME,
                    epoch=epoch,
                    step=step,
                    loss=eval_loss,
                    precision_macro=_coerce_float(item.get("eval_precision_macro")),
                    recall_macro=_coerce_float(item.get("eval_recall_macro")),
                    f1_macro=_coerce_float(item.get("eval_f1_macro")),
                )
            )

    if not rows:
        logger.warning("训练日志回收完成，但 %s 中没有可写入的训练/验证指标。", trainer_state_path)
        return False

    artifact_store.append_artifact_rows(
        model_layout=model_layout,
        artifact_name=TRAIN_METRICS_ARTIFACT_NAME,
        fieldnames=_TRAIN_METRICS_FIELDNAMES,
        rows=rows,
    )
    return True


__all__ = [
    "build_train_result",
    "load_trainer_state_payload",
    "recover_trainer_history_metrics",
]