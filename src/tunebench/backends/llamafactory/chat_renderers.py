"""LlamaFactory chat prompt renderer。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tunebench.contracts import ChatSpec

from .generation import build_messages, encode_prompt_ids_with_chat_template, encode_prompt_ids_with_llamafactory
from .loaders import LoadedInferenceRuntime


@dataclass(frozen=True, slots=True)
class RenderedChatPrompt:
    """描述一次 chat prompt 渲染结果。"""

    prompt_engine: str
    prompt_id_batches: list[list[int]]
    instruction: str
    template_name: str | None
    reasoning_mode: str | None
    reasoning_suffix_style: str | None


class ChatPromptRenderer(Protocol):
    """描述 chat prompt 渲染器协议。"""

    prompt_engine: str

    def render(
        self,
        *,
        runtime: LoadedInferenceRuntime,
        spec: ChatSpec,
        instruction: str,
        template_name: str | None,
        reasoning_mode: str | None,
        reasoning_suffix_style: str | None,
    ) -> RenderedChatPrompt:
        """渲染消息并返回编码后的 prompt token。"""


@dataclass(frozen=True, slots=True)
class LlamaFactoryChatPromptRenderer:
    """使用 LlamaFactory 模板链渲染 chat prompt。"""

    prompt_engine: str = "llamafactory"

    def render(
        self,
        *,
        runtime: LoadedInferenceRuntime,
        spec: ChatSpec,
        instruction: str,
        template_name: str | None,
        reasoning_mode: str | None,
        reasoning_suffix_style: str | None,
    ) -> RenderedChatPrompt:
        messages = build_messages(
            instruction=instruction,
            text=spec.message,
            template_name=template_name,
            reasoning_mode=reasoning_mode,
            reasoning_suffix_style=reasoning_suffix_style,
            apply_reasoning_suffix=reasoning_mode is not None,
        )
        prompt_id_batches = encode_prompt_ids_with_llamafactory(
            runtime=runtime,
            messages_list=[messages],
            max_sequence_length=spec.max_sequence_length,
        )
        return RenderedChatPrompt(
            prompt_engine=self.prompt_engine,
            prompt_id_batches=prompt_id_batches,
            instruction=instruction,
            template_name=template_name,
            reasoning_mode=reasoning_mode,
            reasoning_suffix_style=reasoning_suffix_style,
        )


@dataclass(frozen=True, slots=True)
class NativeChatTemplateRenderer:
    """使用模型原生 chat template 渲染 external chat prompt。"""

    prompt_engine: str = "native"

    def render(
        self,
        *,
        runtime: LoadedInferenceRuntime,
        spec: ChatSpec,
        instruction: str,
        template_name: str | None,
        reasoning_mode: str | None,
        reasoning_suffix_style: str | None,
    ) -> RenderedChatPrompt:
        messages = build_messages(
            instruction=instruction,
            text=spec.message,
        )
        chat_template_kwargs: dict[str, object] = {}
        if spec.enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = spec.enable_thinking
        prompt_id_batches = encode_prompt_ids_with_chat_template(
            runtime=runtime,
            messages_list=[messages],
            max_sequence_length=spec.max_sequence_length,
            chat_template_kwargs=chat_template_kwargs,
        )
        return RenderedChatPrompt(
            prompt_engine=self.prompt_engine,
            prompt_id_batches=prompt_id_batches,
            instruction=instruction,
            template_name=template_name,
            reasoning_mode=reasoning_mode,
            reasoning_suffix_style=reasoning_suffix_style,
        )


def select_external_chat_prompt_renderer(spec: ChatSpec) -> ChatPromptRenderer:
    """按 external chat 语义选择 prompt 渲染器。"""
    if spec.prompt_engine == "llamafactory":
        return LlamaFactoryChatPromptRenderer()
    return NativeChatTemplateRenderer()