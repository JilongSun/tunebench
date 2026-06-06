"""LlamaFactory reasoning policy 工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .model_profiles import ResolvedLlamaFactoryModel


ReasoningControl = Literal["template_only", "template_and_suffix", "native_enable_thinking"]


def resolve_reasoning_suffix_value(reasoning_suffix_style: str | None, reasoning_mode: str | None) -> str | None:
    """解析 reasoning 模式对应的 Qwen3 软开关后缀。"""
    if reasoning_suffix_style != "qwen3":
        return None
    if reasoning_mode == "no_think":
        return "/no_think"
    if reasoning_mode == "think":
        return "/think"
    return None


@dataclass(frozen=True, slots=True)
class ReasoningPolicy:
    """描述一次 train/eval/chat 实际生效的 reasoning 控制策略。"""

    template: str | None
    effective_reasoning_mode: str | None
    reasoning_suffix_style: str | None
    reasoning_suffix_value: str | None
    reasoning_control: ReasoningControl
    default_reasoning_mode: str | None = None
    reasoning_mode_source: str | None = None
    prompt_engine: str | None = None
    enable_thinking: bool | None = None

    def to_payload(self) -> dict[str, Any]:
        """序列化为兼容当前 metadata/plan 的字典。"""
        payload: dict[str, Any] = {
            "template": self.template,
            "effective_reasoning_mode": self.effective_reasoning_mode,
            "reasoning_suffix_style": self.reasoning_suffix_style,
            "reasoning_suffix_value": self.reasoning_suffix_value,
            "reasoning_control": self.reasoning_control,
        }
        if self.default_reasoning_mode is not None:
            payload["default_reasoning_mode"] = self.default_reasoning_mode
        if self.reasoning_mode_source is not None:
            payload["reasoning_mode_source"] = self.reasoning_mode_source
        if self.prompt_engine is not None:
            payload["prompt_engine"] = self.prompt_engine
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        return payload


def _normalize_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized_value = value.strip()
    return normalized_value or None


def build_train_reasoning_policy(
    resolved_model: ResolvedLlamaFactoryModel,
    *,
    reasoning_mode_source: str,
) -> ReasoningPolicy:
    """基于模型画像构建训练期 reasoning policy。"""
    reasoning_suffix_value = resolve_reasoning_suffix_value(
        resolved_model.reasoning_suffix_style,
        resolved_model.reasoning_mode,
    )
    return ReasoningPolicy(
        template=resolved_model.template,
        effective_reasoning_mode=resolved_model.reasoning_mode,
        reasoning_suffix_style=resolved_model.reasoning_suffix_style,
        reasoning_suffix_value=reasoning_suffix_value,
        reasoning_control="template_and_suffix" if reasoning_suffix_value is not None else "template_only",
        default_reasoning_mode=resolved_model.variant.default_reasoning_mode,
        reasoning_mode_source=reasoning_mode_source,
    )


def build_reasoning_policy_from_metadata(metadata: dict[str, Any]) -> ReasoningPolicy:
    """从训练 metadata 构建内部评测/聊天使用的 reasoning policy。"""
    backend_config = metadata.get("backend_config")
    if not isinstance(backend_config, dict):
        raise ValueError("metadata.backend_config 缺失或类型无效。")

    template = _normalize_optional_text(backend_config.get("template"))
    effective_reasoning_mode = _normalize_optional_text(backend_config.get("reasoning_mode"))
    reasoning_suffix_style = _normalize_optional_text(backend_config.get("reasoning_suffix_style"))
    reasoning_suffix_value = resolve_reasoning_suffix_value(reasoning_suffix_style, effective_reasoning_mode)
    return ReasoningPolicy(
        template=template,
        effective_reasoning_mode=effective_reasoning_mode,
        reasoning_suffix_style=reasoning_suffix_style,
        reasoning_suffix_value=reasoning_suffix_value,
        reasoning_control="template_and_suffix" if reasoning_suffix_value is not None else "template_only",
    )


def build_chat_reasoning_policy(
    *,
    prompt_engine: str,
    template: str | None,
    reasoning_mode: str | None,
    reasoning_suffix_style: str | None,
    enable_thinking: bool | None,
) -> ReasoningPolicy:
    """构建 chat 使用的 reasoning policy。"""
    reasoning_suffix_value = resolve_reasoning_suffix_value(reasoning_suffix_style, reasoning_mode)
    if prompt_engine == "native":
        reasoning_control: ReasoningControl = "native_enable_thinking"
    else:
        reasoning_control = "template_and_suffix" if reasoning_suffix_value is not None else "template_only"

    return ReasoningPolicy(
        prompt_engine=prompt_engine,
        template=template,
        effective_reasoning_mode=reasoning_mode,
        reasoning_suffix_style=reasoning_suffix_style,
        reasoning_suffix_value=reasoning_suffix_value,
        reasoning_control=reasoning_control,
        enable_thinking=enable_thinking if prompt_engine == "native" else None,
    )


__all__ = [
    "ReasoningControl",
    "ReasoningPolicy",
    "build_chat_reasoning_policy",
    "build_reasoning_policy_from_metadata",
    "build_train_reasoning_policy",
    "resolve_reasoning_suffix_value",
]