"""LlamaFactory 推理运行时装载工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams.data_args import DataArguments
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase

from tunebench.artifacts import ModelArtifactLayout
from tunebench.util import get_logger

from .metadata import (
    resolve_loader_family,
    resolve_model_name_or_path,
    resolve_supports_multimodal_wrapper,
    resolve_template_name,
)
from .models import ModelLoaderFamily


logger = get_logger("backends.llamafactory.loaders")


@dataclass(frozen=True, slots=True)
class LoadedInferenceRuntime:
    """描述一次推理所需的已加载运行时对象。"""

    model: Any
    tokenizer: PreTrainedTokenizerBase
    template: Any | None
    processor: Any | None
    loader_family: ModelLoaderFamily
    supports_multimodal_wrapper: bool


def _normalize_generation_config(model: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return

    generation_config.do_sample = False
    for attribute_name in ("temperature", "top_p", "top_k"):
        if hasattr(generation_config, attribute_name):
            setattr(generation_config, attribute_name, None)


def _resolve_tokenizer_source(model_dir: Path, model_name_or_path: str) -> str:
    if (model_dir / "tokenizer_config.json").exists():
        return str(model_dir)
    return model_name_or_path


def _resolve_processor_source(model_dir: Path, model_name_or_path: str) -> str:
    if any((model_dir / file_name).exists() for file_name in ("preprocessor_config.json", "processor_config.json")):
        return str(model_dir)
    return model_name_or_path


def prepare_template(
    *,
    tokenizer: PreTrainedTokenizerBase,
    template_name: str | None,
    enable_thinking: bool | None = None,
) -> Any:
    """按 LlamaFactory 官方模板逻辑修正 tokenizer 并返回模板对象。"""
    data_args_kwargs: dict[str, Any] = {"template": template_name}
    if enable_thinking is not None:
        data_args_kwargs["enable_thinking"] = enable_thinking
    data_args = DataArguments(**data_args_kwargs)
    return get_template_and_fix_tokenizer(cast(Any, tokenizer), data_args)


def _template_requires_processor(template: Any) -> bool:
    mm_plugin = getattr(template, "mm_plugin", None)
    return any(getattr(mm_plugin, attribute_name, None) is not None for attribute_name in ("image_token", "video_token", "audio_token"))


def _load_optional_processor(model_name_or_path: str) -> Any | None:
    start_time = perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        logger.info(
            "可选 processor 加载完成: model=%s, processor_type=%s, elapsed_seconds=%.2f",
            model_name_or_path,
            type(processor).__name__,
            perf_counter() - start_time,
        )
        return processor
    except Exception as exc:
        logger.debug(
            "加载可选 processor 失败，将继续使用无 processor 模式: model=%s, elapsed_seconds=%.2f, error=%s",
            model_name_or_path,
            perf_counter() - start_time,
            exc,
        )
        return None


def _finalize_runtime(
    *,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    template: Any | None,
    processor: Any | None,
    loader_family: ModelLoaderFamily,
    supports_multimodal_wrapper: bool,
) -> LoadedInferenceRuntime:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    _normalize_generation_config(model)
    model.eval()
    return LoadedInferenceRuntime(
        model=model,
        tokenizer=tokenizer,
        template=template,
        processor=processor,
        loader_family=loader_family,
        supports_multimodal_wrapper=supports_multimodal_wrapper,
    )


def _resolve_model_loader(loader_family: ModelLoaderFamily) -> Any:
    if loader_family == "conditional_generation":
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def _should_load_processor(*, template: Any | None, supports_multimodal_wrapper: bool) -> bool:
    if template is None:
        return False
    if supports_multimodal_wrapper:
        return True
    return _template_requires_processor(template)


def load_model_and_tokenizer(
    *,
    artifact_type: str,
    metadata: dict[str, Any],
    model_layout: ModelArtifactLayout,
) -> LoadedInferenceRuntime:
    """按 artifact_type 加载推理模型与 tokenizer。"""
    model_name_or_path = resolve_model_name_or_path(metadata)
    template_name = resolve_template_name(metadata)
    loader_family = resolve_loader_family(metadata)
    supports_multimodal_wrapper = resolve_supports_multimodal_wrapper(metadata)
    model_loader = _resolve_model_loader(loader_family)
    logger.info(
        "开始加载推理运行时: artifact_type=%s, model_name_or_path=%s, loader_family=%s, supports_multimodal_wrapper=%s",
        artifact_type,
        model_name_or_path,
        loader_family,
        supports_multimodal_wrapper,
    )

    if artifact_type == "merged":
        tokenizer_start_time = perf_counter()
        tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(
                _resolve_tokenizer_source(model_layout.merged_model_dir, model_name_or_path),
                use_fast=True,
                trust_remote_code=True,
            ),
        )
        logger.info("merged tokenizer 加载完成: elapsed_seconds=%.2f", perf_counter() - tokenizer_start_time)
        template = prepare_template(tokenizer=tokenizer, template_name=template_name)
        processor = (
            _load_optional_processor(_resolve_processor_source(model_layout.merged_model_dir, model_name_or_path))
            if _should_load_processor(template=template, supports_multimodal_wrapper=supports_multimodal_wrapper)
            else None
        )
        model_start_time = perf_counter()
        model = model_loader.from_pretrained(
            str(model_layout.merged_model_dir),
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        logger.info(
            "merged model 加载完成: model_type=%s, elapsed_seconds=%.2f",
            type(model).__name__,
            perf_counter() - model_start_time,
        )
        return _finalize_runtime(
            model=model,
            tokenizer=tokenizer,
            template=template,
            processor=processor,
            loader_family=loader_family,
            supports_multimodal_wrapper=supports_multimodal_wrapper,
        )

    if artifact_type == "lora":
        tokenizer_start_time = perf_counter()
        tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(
                _resolve_tokenizer_source(model_layout.lora_dir, model_name_or_path),
                use_fast=True,
                trust_remote_code=True,
            ),
        )
        logger.info("lora tokenizer 加载完成: elapsed_seconds=%.2f", perf_counter() - tokenizer_start_time)
        template = prepare_template(tokenizer=tokenizer, template_name=template_name)
        processor = (
            _load_optional_processor(_resolve_processor_source(model_layout.lora_dir, model_name_or_path))
            if _should_load_processor(template=template, supports_multimodal_wrapper=supports_multimodal_wrapper)
            else None
        )
        base_model_start_time = perf_counter()
        base_model = model_loader.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        logger.info(
            "LoRA base model 加载完成: model_type=%s, elapsed_seconds=%.2f",
            type(base_model).__name__,
            perf_counter() - base_model_start_time,
        )
        peft_start_time = perf_counter()
        model = PeftModel.from_pretrained(base_model, str(model_layout.lora_dir))
        logger.info(
            "LoRA adapter 注入完成: model_type=%s, elapsed_seconds=%.2f",
            type(model).__name__,
            perf_counter() - peft_start_time,
        )
        return _finalize_runtime(
            model=model,
            tokenizer=tokenizer,
            template=template,
            processor=processor,
            loader_family=loader_family,
            supports_multimodal_wrapper=supports_multimodal_wrapper,
        )

    raise ValueError(f"artifact_type={artifact_type} 非法，仅支持 merged 或 lora。")


def load_external_model_and_tokenizer(
    model_name_or_path: str,
    *,
    template_name: str | None = None,
    enable_thinking: bool | None = None,
    loader_family: ModelLoaderFamily = "causal_lm",
    supports_multimodal_wrapper: bool = False,
) -> LoadedInferenceRuntime:
    """直接从外部本地模型目录加载推理模型与 tokenizer。"""
    normalized_model_name_or_path = model_name_or_path.strip()
    if not normalized_model_name_or_path:
        raise ValueError("external_model_path 不能为空字符串。")

    tokenizer_start_time = perf_counter()
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            normalized_model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
        ),
    )
    logger.info("external tokenizer 加载完成: model=%s, elapsed_seconds=%.2f", normalized_model_name_or_path, perf_counter() - tokenizer_start_time)
    template = None
    processor = None
    if template_name is not None:
        template = prepare_template(
            tokenizer=tokenizer,
            template_name=template_name,
            enable_thinking=enable_thinking,
        )
        processor = (
            _load_optional_processor(normalized_model_name_or_path)
            if _should_load_processor(template=template, supports_multimodal_wrapper=supports_multimodal_wrapper)
            else None
        )
    model_loader = _resolve_model_loader(loader_family)
    model_start_time = perf_counter()
    model = model_loader.from_pretrained(
        normalized_model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    logger.info(
        "external model 加载完成: model=%s, model_type=%s, elapsed_seconds=%.2f",
        normalized_model_name_or_path,
        type(model).__name__,
        perf_counter() - model_start_time,
    )
    return _finalize_runtime(
        model=model,
        tokenizer=tokenizer,
        template=template,
        processor=processor,
        loader_family=loader_family,
        supports_multimodal_wrapper=supports_multimodal_wrapper,
    )


__all__ = [
    "LoadedInferenceRuntime",
    "load_external_model_and_tokenizer",
    "load_model_and_tokenizer",
    "prepare_template",
]