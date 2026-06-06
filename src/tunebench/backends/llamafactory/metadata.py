"""LlamaFactory 推理 metadata 解析工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ModelLoaderFamily, resolve_model_variant
from .prompting import build_instruction


_QWEN_REASONING_SUFFIX_TEMPLATE_NAMES = frozenset({"qwen3", "qwen3_nothink"})


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    """读取训练 metadata。"""
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_backend_config(metadata: dict[str, Any]) -> dict[str, Any]:
    """解析 metadata 中的 backend_config。"""
    backend_config = metadata.get("backend_config")
    if not isinstance(backend_config, dict):
        raise ValueError("metadata.backend_config 缺失或类型无效。")
    return backend_config


def resolve_model_name_or_path(metadata: dict[str, Any]) -> str:
    """解析训练时记录的基座模型路径。"""
    backend_config = resolve_backend_config(metadata)
    model_name_or_path = backend_config.get("model_name_or_path")
    if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
        raise ValueError("metadata.backend_config.model_name_or_path 缺失。")
    return model_name_or_path.strip()


def resolve_model_key(metadata: dict[str, Any]) -> str | None:
    """解析训练时记录的 model_key。"""
    backend_config = resolve_backend_config(metadata)
    model_key = backend_config.get("model_key")
    if not isinstance(model_key, str):
        return None
    normalized_model_key = model_key.strip()
    return normalized_model_key or None


def resolve_loader_family(metadata: dict[str, Any]) -> ModelLoaderFamily:
    """解析训练时记录的装载族；旧 run 缺失时按 model_key 回退推断。"""
    backend_config = resolve_backend_config(metadata)
    loader_family = backend_config.get("loader_family")
    if isinstance(loader_family, str):
        normalized_loader_family = loader_family.strip().lower()
        if normalized_loader_family in {"causal_lm", "conditional_generation"}:
            return normalized_loader_family  # type: ignore[return-value]

    model_key = backend_config.get("model_key")
    reasoning_mode = backend_config.get("reasoning_mode")
    if isinstance(model_key, str) and model_key.strip():
        resolved_model = resolve_model_variant(
            model_key.strip(),
            reasoning_mode.strip() if isinstance(reasoning_mode, str) and reasoning_mode.strip() else None,
        )
        return resolved_model.loader_family
    return "causal_lm"


def resolve_supports_multimodal_wrapper(metadata: dict[str, Any]) -> bool:
    """解析训练时记录的多模态 wrapper 能力；旧 run 缺失时按 model_key 回退推断。"""
    backend_config = resolve_backend_config(metadata)
    supports_multimodal_wrapper = backend_config.get("supports_multimodal_wrapper")
    if isinstance(supports_multimodal_wrapper, bool):
        return supports_multimodal_wrapper

    model_key = backend_config.get("model_key")
    reasoning_mode = backend_config.get("reasoning_mode")
    if isinstance(model_key, str) and model_key.strip():
        resolved_model = resolve_model_variant(
            model_key.strip(),
            reasoning_mode.strip() if isinstance(reasoning_mode, str) and reasoning_mode.strip() else None,
        )
        return resolved_model.supports_multimodal_wrapper
    return False


def resolve_reasoning_mode(metadata: dict[str, Any]) -> str | None:
    """解析训练时记录的 reasoning_mode。"""
    backend_config = resolve_backend_config(metadata)
    reasoning_mode = backend_config.get("reasoning_mode")
    return reasoning_mode if isinstance(reasoning_mode, str) else None


def resolve_template_name(metadata: dict[str, Any]) -> str | None:
    """解析训练时记录的 template。"""
    backend_config = resolve_backend_config(metadata)
    template_name = backend_config.get("template")
    if not isinstance(template_name, str):
        return None
    normalized_template_name = template_name.strip()
    return normalized_template_name or None


def resolve_default_instruction(metadata: dict[str, Any]) -> str:
    """优先使用 metadata 中已固化的 instruction，否则回退到兼容逻辑。"""
    metadata_instruction = metadata.get("instruction")
    if isinstance(metadata_instruction, str) and metadata_instruction.strip():
        return metadata_instruction.strip()

    backend_config = resolve_backend_config(metadata)
    custom_instruction = backend_config.get("instruction")
    if isinstance(custom_instruction, str) and custom_instruction.strip():
        return custom_instruction.strip()

    label_names = metadata.get("label_names")
    if not isinstance(label_names, list) or not all(isinstance(label_name, str) for label_name in label_names):
        raise ValueError("metadata.label_names 缺失或类型无效。")
    return build_instruction(tuple(label_names))


def infer_reasoning_suffix_style_from_template_name(template_name: str | None) -> str | None:
    """按模板名推断 reasoning 后缀风格。"""
    if template_name is None:
        return None
    normalized_template_name = template_name.strip().lower()
    if normalized_template_name in _QWEN_REASONING_SUFFIX_TEMPLATE_NAMES:
        return "qwen3"
    return None


def resolve_reasoning_suffix_style(metadata: dict[str, Any]) -> str | None:
    """解析 metadata 中记录的 reasoning 后缀风格。"""
    backend_config = resolve_backend_config(metadata)
    reasoning_suffix_style = backend_config.get("reasoning_suffix_style")
    if isinstance(reasoning_suffix_style, str):
        normalized_reasoning_suffix_style = reasoning_suffix_style.strip().lower()
        return normalized_reasoning_suffix_style or None

    return infer_reasoning_suffix_style_from_template_name(resolve_template_name(metadata))


__all__ = [
    "infer_reasoning_suffix_style_from_template_name",
    "load_metadata",
    "resolve_backend_config",
    "resolve_default_instruction",
    "resolve_loader_family",
    "resolve_model_key",
    "resolve_model_name_or_path",
    "resolve_reasoning_mode",
    "resolve_reasoning_suffix_style",
    "resolve_supports_multimodal_wrapper",
    "resolve_template_name",
]