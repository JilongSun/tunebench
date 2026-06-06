"""结构化输出评测工具。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from tunebench.util import get_logger


_ALLOWED_CONFIDENCE_VALUES = (0.3, 0.6, 0.9)
_TOP_LEVEL_KEYS = {"reasoning", "intents", "intent_relations"}
_INTENT_ITEM_KEYS = {"intent", "confidence", "follow_up_question"}
_LEADING_THINK_BLOCK_PATTERN = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


logger = get_logger("classification.structured_output_metrics")


@dataclass(frozen=True, slots=True)
class StructuredOutputAssessment:
    """描述一条结构化输出的解析与校验结果。"""

    json_valid: bool
    reasoning: str | None
    reasoning_char_count: int | None
    reasoning_length_valid: bool
    predicted_intents: tuple[str, ...]
    primary_intent: str | None
    confidence_values: tuple[float, ...]
    max_confidence: float | None
    confidence_enum_valid: bool
    confidence_range_valid: bool
    errors: tuple[str, ...]


def _build_compact_label_mapping(label_names: Sequence[str]) -> dict[str, str]:
    """构建移除空白后的标签映射，仅保留可唯一回填的标准标签。"""
    grouped_labels: dict[str, list[str]] = {}
    for label_name in label_names:
        stripped_label = label_name.strip()
        if not stripped_label:
            continue
        compact_label = _compact_label_whitespace(stripped_label)
        grouped_labels.setdefault(compact_label, []).append(stripped_label)

    return {
        compact_label: candidates[0]
        for compact_label, candidates in grouped_labels.items()
        if len(set(candidates)) == 1
    }


def assess_structured_output(raw_output: str, label_names: Sequence[str]) -> StructuredOutputAssessment:
    """解析单条模型输出并生成结构化评测结果。"""
    allowed_labels = {label for label in label_names if label.strip()}
    compact_label_mapping = _build_compact_label_mapping(label_names)
    semantic_errors: list[str] = []

    try:
        payload = _extract_json_payload(raw_output)
    except Exception as exc:
        logger.warning("结构化输出解析失败: %s", exc)
        return StructuredOutputAssessment(
            json_valid=False,
            reasoning=None,
            reasoning_char_count=None,
            reasoning_length_valid=False,
            predicted_intents=(),
            primary_intent=None,
            confidence_values=(),
            max_confidence=None,
            confidence_enum_valid=False,
            confidence_range_valid=False,
            errors=(str(exc),),
        )

    schema_errors: list[str] = []
    extra_keys = set(payload) - _TOP_LEVEL_KEYS
    if extra_keys:
        schema_errors.append(f"顶层存在非法字段: {sorted(extra_keys)}")

    reasoning_value = payload.get("reasoning")
    normalized_reasoning: str | None = None
    if not isinstance(reasoning_value, str):
        schema_errors.append("reasoning 字段必须是字符串。")
    else:
        normalized_reasoning = reasoning_value.strip()

    intents_payload = payload.get("intents")
    predicted_intent_candidates: list[tuple[str, float | None]] = []
    confidence_values: list[float] = []
    if not isinstance(intents_payload, list):
        schema_errors.append("intents 字段必须是数组。")
    else:
        for index, item in enumerate(intents_payload, start=1):
            if not isinstance(item, dict):
                schema_errors.append(f"intents[{index}] 必须是对象。")
                continue

            extra_item_keys = set(item) - _INTENT_ITEM_KEYS
            if extra_item_keys:
                schema_errors.append(f"intents[{index}] 存在非法字段: {sorted(extra_item_keys)}")

            intent_value = item.get("intent")
            normalized_labels: list[str] = []
            if not isinstance(intent_value, list) or not all(isinstance(candidate, str) for candidate in intent_value):
                schema_errors.append(f"intents[{index}].intent 必须是字符串数组。")
            else:
                for raw_label in intent_value:
                    label = raw_label.strip()
                    if not label:
                        continue
                    normalized_label = _normalize_label_name(label, allowed_labels, compact_label_mapping)
                    if normalized_label is None:
                        semantic_errors.append(f"intents[{index}] 包含未注册标签: {label}")
                        continue
                    normalized_labels.append(normalized_label)

            confidence_raw = item.get("confidence")
            confidence_value: float | None = None
            if isinstance(confidence_raw, (int, float)) and not isinstance(confidence_raw, bool):
                confidence_value = float(confidence_raw)
                confidence_values.append(confidence_value)
            else:
                schema_errors.append(f"intents[{index}].confidence 必须是数值。")

            follow_up_question = item.get("follow_up_question")
            if not isinstance(follow_up_question, str):
                schema_errors.append(f"intents[{index}].follow_up_question 必须是字符串。")

            for label in normalized_labels:
                predicted_intent_candidates.append((label, confidence_value))

    reasoning_char_count = len(normalized_reasoning) if normalized_reasoning is not None else None
    reasoning_length_valid = reasoning_char_count is not None and 0 < reasoning_char_count <= 120
    confidence_range_valid = bool(confidence_values) and all(0.0 <= value <= 1.0 for value in confidence_values)
    confidence_enum_valid = bool(confidence_values) and all(_matches_allowed_confidence(value) for value in confidence_values)

    predicted_intents = tuple(sorted({label for label, _ in predicted_intent_candidates}))
    primary_intent = _resolve_primary_intent(predicted_intent_candidates)
    max_confidence = max(confidence_values) if confidence_values else None

    return StructuredOutputAssessment(
        json_valid=not schema_errors,
        reasoning=normalized_reasoning,
        reasoning_char_count=reasoning_char_count,
        reasoning_length_valid=reasoning_length_valid,
        predicted_intents=predicted_intents,
        primary_intent=primary_intent,
        confidence_values=tuple(confidence_values),
        max_confidence=max_confidence,
        confidence_enum_valid=confidence_enum_valid,
        confidence_range_valid=confidence_range_valid,
        errors=tuple(schema_errors + semantic_errors),
    )


def compute_intent_metrics_bundle(
    *,
    gold_intents: Sequence[set[str]],
    predicted_intents: Sequence[set[str]],
    label_names: Sequence[str],
) -> tuple[dict[str, float], dict[str, dict[str, float | int]]]:
    """基于多标签 intent 集合计算宏平均与逐标签指标。"""
    if len(gold_intents) != len(predicted_intents):
        raise ValueError("gold_intents 与 predicted_intents 长度不一致。")

    if not label_names:
        return {"precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}, {}

    label_metrics: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for label_name in label_names:
        true_positive = sum(1 for gold, pred in zip(gold_intents, predicted_intents) if label_name in gold and label_name in pred)
        false_positive = sum(1 for gold, pred in zip(gold_intents, predicted_intents) if label_name not in gold and label_name in pred)
        false_negative = sum(1 for gold, pred in zip(gold_intents, predicted_intents) if label_name in gold and label_name not in pred)
        support = sum(1 for gold in gold_intents if label_name in gold)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        label_metrics[label_name] = {
            "support": int(support),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_score),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    divisor = len(label_names)
    return (
        {
            "precision_macro": float(sum(precisions) / divisor),
            "recall_macro": float(sum(recalls) / divisor),
            "f1_macro": float(sum(f1_scores) / divisor),
        },
        label_metrics,
    )


def sanitize_structured_output_text(content: str) -> str:
    """清洗结构化输出前的包裹内容，优先剥离前置 think 块。"""
    normalized_content = content.strip()
    while True:
        sanitized_content = _LEADING_THINK_BLOCK_PATTERN.sub("", normalized_content, count=1)
        if sanitized_content == normalized_content:
            break
        normalized_content = sanitized_content.lstrip()

    if normalized_content.startswith("```"):
        lines = normalized_content.splitlines()
        if len(lines) >= 3 and lines[-1].startswith("```"):
            normalized_content = "\n".join(lines[1:-1]).strip()

    return normalized_content.strip()


def _extract_json_payload(content: str) -> dict[str, Any]:
    normalized_content = sanitize_structured_output_text(content)

    try:
        payload = json.loads(normalized_content)
    except json.JSONDecodeError:
        start = normalized_content.find("{")
        end = normalized_content.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise ValueError("输出内容无法解析为 JSON 对象。")
        payload = json.loads(normalized_content[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("输出 JSON 顶层必须是对象。")
    return payload


def _matches_allowed_confidence(value: float) -> bool:
    return any(abs(value - candidate) < 1e-9 for candidate in _ALLOWED_CONFIDENCE_VALUES)


def _normalize_label_name(
    label: str,
    allowed_labels: set[str],
    compact_label_mapping: dict[str, str],
) -> str | None:
    if label in allowed_labels:
        return label
    return compact_label_mapping.get(_compact_label_whitespace(label))


def _compact_label_whitespace(label: str) -> str:
    return re.sub(r"\s+", "", label)


def _resolve_primary_intent(predicted_intent_candidates: Sequence[tuple[str, float | None]]) -> str | None:
    if not predicted_intent_candidates:
        return None

    ranked_candidates = sorted(
        predicted_intent_candidates,
        key=lambda item: (-1.0 if item[1] is None else -item[1], item[0]),
    )
    return ranked_candidates[0][0]


__all__ = [
    "StructuredOutputAssessment",
    "assess_structured_output",
    "compute_intent_metrics_bundle",
    "sanitize_structured_output_text",
]