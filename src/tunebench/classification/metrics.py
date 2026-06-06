"""分类任务指标计算工具。"""

from __future__ import annotations

from typing import Any

import numpy as np


LABEL_METRIC_KEY_PREFIX = "label_metric__"


def softmax(logits: np.ndarray) -> np.ndarray:
    """对 logits 计算 softmax。"""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def compute_macro_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    """计算 macro precision、recall 与 f1。"""
    label_values = sorted(set(labels) | set(predictions))
    if not label_values:
        return {
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
        }

    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for label_value in label_values:
        true_positive = sum(1 for gold, pred in zip(labels, predictions) if gold == label_value and pred == label_value)
        false_positive = sum(1 for gold, pred in zip(labels, predictions) if gold != label_value and pred == label_value)
        false_negative = sum(1 for gold, pred in zip(labels, predictions) if gold == label_value and pred != label_value)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    divisor = len(label_values)
    return {
        "precision_macro": float(sum(precisions) / divisor),
        "recall_macro": float(sum(recalls) / divisor),
        "f1_macro": float(sum(f1_scores) / divisor),
    }


def compute_label_metrics(labels: list[int], predictions: list[int]) -> dict[int, dict[str, float | int]]:
    """计算逐标签指标。"""
    label_values = sorted(set(labels) | set(predictions))
    label_metrics: dict[int, dict[str, float | int]] = {}
    for label_value in label_values:
        true_positive = sum(1 for gold, pred in zip(labels, predictions) if gold == label_value and pred == label_value)
        false_positive = sum(1 for gold, pred in zip(labels, predictions) if gold != label_value and pred == label_value)
        false_negative = sum(1 for gold, pred in zip(labels, predictions) if gold == label_value and pred != label_value)
        support = sum(1 for gold in labels if gold == label_value)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        label_metrics[label_value] = {
            "support": int(support),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_score),
        }
    return label_metrics


def compute_classification_metrics_bundle(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, float], dict[int, dict[str, float | int]]]:
    """同时返回聚合指标与逐标签指标。"""
    probabilities = softmax(logits)
    predictions = np.argmax(probabilities, axis=-1)
    label_list = labels.tolist()
    prediction_list = predictions.tolist()
    metrics = compute_macro_metrics(label_list, prediction_list)
    metrics["avg_confidence"] = float(np.max(probabilities, axis=-1).mean())
    return metrics, compute_label_metrics(label_list, prediction_list)


def compute_classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """仅返回聚合分类指标。"""
    metrics, _ = compute_classification_metrics_bundle(logits, labels)
    return metrics


def flatten_label_metrics(label_metrics: dict[int, dict[str, float | int]]) -> dict[str, float]:
    """将逐标签指标拍平成 Trainer 可写回的扁平字典。"""
    flattened_metrics: dict[str, float] = {}
    for label_id, metrics in label_metrics.items():
        for metric_name, metric_value in metrics.items():
            flattened_metrics[f"{LABEL_METRIC_KEY_PREFIX}{label_id}__{metric_name}"] = float(metric_value)
    return flattened_metrics


def extract_label_metrics_from_flattened(metrics: dict[str, Any]) -> dict[int, dict[str, float | int]]:
    """从扁平指标字典中恢复逐标签指标。"""
    label_metrics: dict[int, dict[str, float | int]] = {}
    for metric_name, metric_value in metrics.items():
        if not metric_name.startswith(LABEL_METRIC_KEY_PREFIX):
            continue
        metric_suffix = metric_name.removeprefix(LABEL_METRIC_KEY_PREFIX)
        label_id_text, metric_key = metric_suffix.split("__", 1)
        label_id = int(label_id_text)
        label_metrics.setdefault(label_id, {})[metric_key] = int(metric_value) if metric_key == "support" else float(metric_value)
    return label_metrics