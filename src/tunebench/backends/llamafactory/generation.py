"""LlamaFactory prompt 编码与生成工具。"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers import PreTrainedTokenizerBase

from .loaders import LoadedInferenceRuntime
from .metadata import infer_reasoning_suffix_style_from_template_name


_UNKNOWN_MODEL_MAX_LENGTH_THRESHOLD = 1_000_000
_QWEN_REASONING_SUFFIX_BY_MODE = {
    "no_think": "/no_think",
    "think": "/think",
}


@dataclass(frozen=True, slots=True)
class GeneratedOutput:
    """描述单条生成结果及其关键元信息。"""

    text: str
    finish_reason: str
    prompt_token_count: int
    generated_token_count: int


def _resolve_context_window(model: Any, tokenizer: PreTrainedTokenizerBase) -> int | None:
    for attribute_name in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"):
        value = getattr(getattr(model, "config", None), attribute_name, None)
        if isinstance(value, int) and value > 0:
            return value

    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < _UNKNOWN_MODEL_MAX_LENGTH_THRESHOLD:
        return tokenizer_max_length
    return None


def apply_qwen_reasoning_suffix(
    text: str,
    *,
    reasoning_suffix_style: str | None,
    reasoning_mode: str | None,
) -> str:
    """按 Qwen3 软开关约定，在用户消息末尾追加 /think 或 /no_think。"""
    normalized_text = text.strip()
    if not normalized_text or reasoning_suffix_style != "qwen3":
        return normalized_text
    if reasoning_mode is None:
        return normalized_text

    expected_suffix = _QWEN_REASONING_SUFFIX_BY_MODE.get(reasoning_mode)
    if expected_suffix is None:
        return normalized_text

    suffix_stripped_text = normalized_text
    for suffix in _QWEN_REASONING_SUFFIX_BY_MODE.values():
        if suffix_stripped_text.endswith(suffix):
            suffix_stripped_text = suffix_stripped_text[: -len(suffix)].rstrip()
            break
    return f"{suffix_stripped_text}{expected_suffix}"


def build_messages(
    *,
    instruction: str,
    text: str,
    reasoning_suffix_style: str | None = None,
    template_name: str | None = None,
    reasoning_mode: str | None = None,
    apply_reasoning_suffix: bool = False,
) -> list[dict[str, str]]:
    """构建与 LlamaFactory Alpaca prompt/query 一致的单轮消息。"""
    normalized_instruction = instruction.strip()
    normalized_text = text.strip()
    if apply_reasoning_suffix:
        effective_reasoning_suffix_style = reasoning_suffix_style
        if effective_reasoning_suffix_style is None and template_name is not None:
            effective_reasoning_suffix_style = infer_reasoning_suffix_style_from_template_name(template_name)
        normalized_text = apply_qwen_reasoning_suffix(
            normalized_text,
            reasoning_suffix_style=effective_reasoning_suffix_style,
            reasoning_mode=reasoning_mode,
        )
    user_parts = [part for part in (normalized_instruction, normalized_text) if part]
    if not user_parts:
        raise ValueError("instruction 和 text 不能同时为空。")
    return [{"role": "user", "content": "\n".join(user_parts)}]


def _build_prompt_ids_with_chat_template(
    *,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    chat_template_kwargs: dict[str, Any],
) -> list[int]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("当前 tokenizer 不支持 apply_chat_template，无法使用原生 enable_thinking 控制。")

    prompt_ids = apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    while True:
        if isinstance(prompt_ids, Mapping):
            if "input_ids" not in prompt_ids:
                raise ValueError("apply_chat_template 返回了缺少 input_ids 的编码结果。")
            prompt_ids = prompt_ids["input_ids"]
            continue
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.tolist()
            continue
        if hasattr(prompt_ids, "input_ids") and not isinstance(prompt_ids, (list, tuple)):
            prompt_ids = getattr(prompt_ids, "input_ids")
            continue
        if hasattr(prompt_ids, "tolist") and not isinstance(prompt_ids, (list, tuple, str, bytes)):
            prompt_ids = cast(Any, prompt_ids).tolist()
            continue
        break
    if isinstance(prompt_ids, tuple):
        prompt_ids = list(prompt_ids)
    while isinstance(prompt_ids, list) and prompt_ids and isinstance(prompt_ids[0], (list, tuple)):
        prompt_ids = list(prompt_ids[0])
    if not isinstance(prompt_ids, list):
        raise ValueError("apply_chat_template 返回了无法识别的 token 序列类型。")
    return [int(token_id) for token_id in prompt_ids]


def _finalize_prompt_ids(
    prompt_ids: list[int],
    *,
    max_sequence_length: int | None,
    context_window: int | None,
) -> list[int]:
    if max_sequence_length is not None and len(prompt_ids) > max_sequence_length:
        prompt_ids = prompt_ids[-max_sequence_length:]
    if context_window is not None and len(prompt_ids) >= context_window:
        raise ValueError(
            f"输入 token 长度 {len(prompt_ids)} 已达到或超过模型上下文上限 {context_window}，"
            "请手动缩短输入或显式设置 --max-sequence-length。"
        )
    return prompt_ids


def encode_prompt_ids_with_llamafactory(
    *,
    runtime: LoadedInferenceRuntime,
    messages_list: list[list[dict[str, str]]],
    max_sequence_length: int | None,
) -> list[list[int]]:
    """使用 LlamaFactory 模板链将消息编码为 prompt token。"""
    if runtime.template is None:
        raise ValueError("当前运行时未加载 LlamaFactory template，无法使用 llamafactory prompt 引擎。")
    context_window = _resolve_context_window(runtime.model, runtime.tokenizer)
    prompt_id_batch: list[list[int]] = []
    for messages in messages_list:
        normalized_messages = [{"role": message["role"], "content": str(message["content"])} for message in messages]
        processed_messages = runtime.template.mm_plugin.process_messages(normalized_messages, [], [], [], runtime.processor)
        paired_messages = processed_messages + [{"role": "assistant", "content": ""}]
        prompt_ids, _ = runtime.template.encode_oneturn(runtime.tokenizer, paired_messages, None, None)
        prompt_ids, _ = runtime.template.mm_plugin.process_token_ids(
            prompt_ids,
            None,
            [],
            [],
            [],
            runtime.tokenizer,
            runtime.processor,
        )
        prompt_id_batch.append(
            _finalize_prompt_ids(
                prompt_ids,
                max_sequence_length=max_sequence_length,
                context_window=context_window,
            )
        )
    return prompt_id_batch


def encode_prompt_ids_with_chat_template(
    *,
    runtime: LoadedInferenceRuntime,
    messages_list: list[list[dict[str, str]]],
    max_sequence_length: int | None,
    chat_template_kwargs: dict[str, Any],
) -> list[list[int]]:
    """使用模型原生 chat template 将消息编码为 prompt token。"""
    context_window = _resolve_context_window(runtime.model, runtime.tokenizer)
    prompt_id_batch: list[list[int]] = []
    for messages in messages_list:
        normalized_messages = [{"role": message["role"], "content": str(message["content"])} for message in messages]
        prompt_ids = _build_prompt_ids_with_chat_template(
            tokenizer=runtime.tokenizer,
            messages=normalized_messages,
            chat_template_kwargs=chat_template_kwargs,
        )
        prompt_id_batch.append(
            _finalize_prompt_ids(
                prompt_ids,
                max_sequence_length=max_sequence_length,
                context_window=context_window,
            )
        )
    return prompt_id_batch


def generate_outputs_from_prompt_ids(
    *,
    runtime: LoadedInferenceRuntime,
    prompt_id_batches: list[list[int]],
    max_new_tokens: int | None,
    batch_size: int,
    progress_callback: Callable[[int, int, int, int, float], None] | None = None,
) -> tuple[list[GeneratedOutput], float]:
    """按已编码的 prompt token 批量执行生成并返回输出文本与总耗时。"""
    device = next(runtime.model.parameters()).device
    outputs: list[GeneratedOutput] = []
    total_latency_ms = 0.0
    if runtime.template is not None:
        stop_token_ids = runtime.template.get_stop_token_ids(runtime.tokenizer)
    else:
        eos_token_id = runtime.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("当前 tokenizer 缺少 eos_token_id，无法确定停止 token。")
        stop_token_ids = [eos_token_id]
    pad_token_id = cast(int, runtime.tokenizer.pad_token_id)
    context_window = _resolve_context_window(runtime.model, runtime.tokenizer)
    total_sample_count = len(prompt_id_batches)
    total_batch_count = (total_sample_count + batch_size - 1) // batch_size if total_sample_count else 0
    completed_sample_count = 0
    completed_batch_count = 0
    if progress_callback is not None:
        progress_callback(0, total_sample_count, 0, total_batch_count, 0.0)
    for start_index in range(0, len(prompt_id_batches), batch_size):
        prompt_id_batch = prompt_id_batches[start_index : start_index + batch_size]

        max_prompt_length = max(len(prompt_ids) for prompt_ids in prompt_id_batch)
        effective_max_new_tokens = max_new_tokens
        if effective_max_new_tokens is None:
            if context_window is None:
                raise ValueError("无法推断模型上下文上限；请显式传入 --max-new-tokens。")
            effective_max_new_tokens = max(context_window - max_prompt_length, 1)
        batch_input_ids: list[list[int]] = []
        batch_attention_masks: list[list[int]] = []
        for prompt_ids in prompt_id_batch:
            padding_length = max_prompt_length - len(prompt_ids)
            batch_input_ids.append(([pad_token_id] * padding_length) + prompt_ids)
            batch_attention_masks.append(([0] * padding_length) + ([1] * len(prompt_ids)))

        input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(batch_attention_masks, dtype=torch.long, device=device)

        started_at = time.perf_counter()
        with torch.inference_mode():
            generated = runtime.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=effective_max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=stop_token_ids,
            )
        batch_latency_ms = (time.perf_counter() - started_at) * 1000.0
        total_latency_ms += batch_latency_ms

        generated_only = generated[:, max_prompt_length:]
        decoded_outputs = runtime.tokenizer.batch_decode(generated_only, skip_special_tokens=True)
        stop_token_id_set = set(stop_token_ids)
        for prompt_ids, generated_token_ids, decoded_output in zip(
            prompt_id_batch,
            generated_only.tolist(),
            decoded_outputs,
            strict=False,
        ):
            trimmed_generated_token_ids = list(generated_token_ids)
            finish_reason = "length"
            for index, token_id in enumerate(trimmed_generated_token_ids):
                if token_id in stop_token_id_set:
                    trimmed_generated_token_ids = trimmed_generated_token_ids[:index]
                    finish_reason = "stop"
                    break
            outputs.append(
                GeneratedOutput(
                    text=decoded_output.strip(),
                    finish_reason=finish_reason,
                    prompt_token_count=len(prompt_ids),
                    generated_token_count=len(trimmed_generated_token_ids),
                )
            )
        completed_sample_count += len(prompt_id_batch)
        completed_batch_count += 1
        if progress_callback is not None:
            progress_callback(
                completed_sample_count,
                total_sample_count,
                completed_batch_count,
                total_batch_count,
                batch_latency_ms,
            )

    return outputs, total_latency_ms


__all__ = [
    "GeneratedOutput",
    "apply_qwen_reasoning_suffix",
    "build_messages",
    "encode_prompt_ids_with_chat_template",
    "encode_prompt_ids_with_llamafactory",
    "generate_outputs_from_prompt_ids",
]