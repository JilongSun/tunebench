"""LlamaFactory 推理兼容导出层。"""

from .generation import (
    GeneratedOutput,
    apply_qwen_reasoning_suffix,
    build_messages,
    encode_prompt_ids_with_chat_template,
    encode_prompt_ids_with_llamafactory,
    generate_outputs_from_prompt_ids,
)
from .loaders import (
    LoadedInferenceRuntime,
    load_external_model_and_tokenizer,
    load_model_and_tokenizer,
    prepare_template,
)
from .metadata import (
    infer_reasoning_suffix_style_from_template_name,
    load_metadata,
    resolve_backend_config,
    resolve_default_instruction,
    resolve_loader_family,
    resolve_model_name_or_path,
    resolve_reasoning_mode,
    resolve_reasoning_suffix_style,
    resolve_supports_multimodal_wrapper,
    resolve_template_name,
)


__all__ = [
    "GeneratedOutput",
    "LoadedInferenceRuntime",
    "apply_qwen_reasoning_suffix",
    "build_messages",
    "encode_prompt_ids_with_chat_template",
    "encode_prompt_ids_with_llamafactory",
    "generate_outputs_from_prompt_ids",
    "infer_reasoning_suffix_style_from_template_name",
    "load_external_model_and_tokenizer",
    "load_metadata",
    "load_model_and_tokenizer",
    "prepare_template",
    "resolve_backend_config",
    "resolve_default_instruction",
    "resolve_loader_family",
    "resolve_model_name_or_path",
    "resolve_reasoning_mode",
    "resolve_reasoning_suffix_style",
    "resolve_supports_multimodal_wrapper",
    "resolve_template_name",
]