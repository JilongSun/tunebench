"""LlamaFactory 模型画像注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ReasoningMode = Literal["think", "no_think"]
REASONING_MODES: tuple[ReasoningMode, ...] = ("think", "no_think")
ReasoningSuffixStyle = Literal["qwen3"]
ModelLoaderFamily = Literal["causal_lm", "conditional_generation"]


@dataclass(frozen=True, slots=True)
class LlamaFactoryModelProfile:
    """描述一个可直接交给 LlamaFactory 的模型配置。"""

    model_name_or_path: str
    template: str


@dataclass(frozen=True, slots=True)
class LlamaFactoryModelVariant:
    """描述一个 LlamaFactory 模型变体。"""

    model_key: str
    display_name: str
    default_reasoning_mode: ReasoningMode | None
    profiles: dict[ReasoningMode, LlamaFactoryModelProfile]
    reasoning_suffix_style: ReasoningSuffixStyle | None = None
    loader_family: ModelLoaderFamily = "causal_lm"
    supports_multimodal_wrapper: bool = False
    supports_native_chat_template: bool = True
    availability_note: str | None = None

    @property
    def supported_reasoning_modes(self) -> tuple[ReasoningMode, ...]:
        """返回当前模型支持的推理模式。"""
        return tuple(mode for mode in REASONING_MODES if mode in self.profiles)

    @property
    def is_available(self) -> bool:
        """返回当前模型是否已补齐可执行映射。"""
        return bool(self.profiles)


@dataclass(frozen=True, slots=True)
class ResolvedLlamaFactoryModel:
    """描述解析后的 LlamaFactory 可执行模型信息。"""

    variant: LlamaFactoryModelVariant
    reasoning_mode: ReasoningMode
    model_name_or_path: str
    template: str
    reasoning_suffix_style: ReasoningSuffixStyle | None
    loader_family: ModelLoaderFamily
    supports_multimodal_wrapper: bool
    supports_native_chat_template: bool


_MODEL_CATALOG: dict[str, LlamaFactoryModelVariant] = {
    "qwen3_4b": LlamaFactoryModelVariant(
        model_key="qwen3_4b",
        display_name="Qwen3-4B",
        default_reasoning_mode="no_think",
        reasoning_suffix_style="qwen3",
        profiles={
            "think": LlamaFactoryModelProfile(model_name_or_path="Qwen/Qwen3-4B", template="qwen3"),
            "no_think": LlamaFactoryModelProfile(
                model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
                template="qwen3_nothink",
            ),
        },
    ),
    "qwen3_32b": LlamaFactoryModelVariant(
        model_key="qwen3_32b",
        display_name="Qwen3-32B",
        default_reasoning_mode="no_think",
        reasoning_suffix_style="qwen3",
        profiles={
            "think": LlamaFactoryModelProfile(model_name_or_path="Qwen/Qwen3-32B", template="qwen3"),
            "no_think": LlamaFactoryModelProfile(
                model_name_or_path="Qwen/Qwen3-32B",
                template="qwen3_nothink",
            ),
        },
    ),
    "qwen3_5_4b": LlamaFactoryModelVariant(
        model_key="qwen3_5_4b",
        display_name="Qwen3.5-4B",
        default_reasoning_mode="no_think",
        loader_family="conditional_generation",
        supports_multimodal_wrapper=True,
        profiles={
            "think": LlamaFactoryModelProfile(model_name_or_path="Qwen/Qwen3.5-4B", template="qwen3_5"),
            "no_think": LlamaFactoryModelProfile(model_name_or_path="Qwen/Qwen3.5-4B", template="qwen3_5_nothink"),
        },
        availability_note="Qwen3.5-4B 训练侧默认走 qwen3_5_nothink 模板，不启用 /no_think 软开关。",
    ),
    "qwen3_6_27b": LlamaFactoryModelVariant(
        model_key="qwen3_6_27b",
        display_name="Qwen3.6-27B",
        default_reasoning_mode=None,
        loader_family="conditional_generation",
        supports_multimodal_wrapper=True,
        profiles={},
        availability_note="当前版本未在 LlamaFactory 官方常量中检索到可验证映射，暂不开放执行。",
    ),
}


def list_model_keys() -> tuple[str, ...]:
    """返回已登记的模型键。"""
    return tuple(sorted(_MODEL_CATALOG))


def get_model_variant(model_key: str) -> LlamaFactoryModelVariant:
    """按模型键返回注册表项。"""
    try:
        return _MODEL_CATALOG[model_key]
    except KeyError as exc:
        available_model_keys = ", ".join(list_model_keys())
        raise ValueError(f"未注册的 LlamaFactory 模型键: {model_key}；当前可用模型: {available_model_keys}") from exc


def resolve_model_variant(model_key: str, reasoning_mode: ReasoningMode | None) -> ResolvedLlamaFactoryModel:
    """解析模型键、思考模式以及底层模板配置。"""
    variant = get_model_variant(model_key)
    if not variant.is_available:
        message = f"模型 {model_key} 当前尚未补齐可执行映射。"
        if variant.availability_note:
            message = f"{message}{variant.availability_note}"
        raise ValueError(message)

    effective_reasoning_mode = reasoning_mode or variant.default_reasoning_mode
    if effective_reasoning_mode is None:
        raise ValueError(f"模型 {model_key} 缺少默认 reasoning_mode，请显式指定。")

    profile = variant.profiles.get(effective_reasoning_mode)
    if profile is None:
        supported_modes = ", ".join(variant.supported_reasoning_modes)
        message = (
            f"模型 {model_key} 不支持 reasoning_mode={effective_reasoning_mode}；"
            f"当前支持: {supported_modes or '无'}"
        )
        if variant.availability_note:
            message = f"{message}。{variant.availability_note}"
        raise ValueError(message)

    return ResolvedLlamaFactoryModel(
        variant=variant,
        reasoning_mode=effective_reasoning_mode,
        model_name_or_path=profile.model_name_or_path,
        template=profile.template,
        reasoning_suffix_style=variant.reasoning_suffix_style,
        loader_family=variant.loader_family,
        supports_multimodal_wrapper=variant.supports_multimodal_wrapper,
        supports_native_chat_template=variant.supports_native_chat_template,
    )


__all__ = [
    "LlamaFactoryModelProfile",
    "LlamaFactoryModelVariant",
    "ModelLoaderFamily",
    "REASONING_MODES",
    "ReasoningMode",
    "ReasoningSuffixStyle",
    "ResolvedLlamaFactoryModel",
    "get_model_variant",
    "list_model_keys",
    "resolve_model_variant",
]