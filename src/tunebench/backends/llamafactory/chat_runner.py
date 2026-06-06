"""LlamaFactory 单轮聊天后端。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from tunebench.artifacts import ModelPathManager, get_model_path_manager
from tunebench.contracts import ChatResult, ChatSpec, RunPlan
from tunebench.util import get_logger

from .chat_renderers import LlamaFactoryChatPromptRenderer, select_external_chat_prompt_renderer
from .generation import generate_outputs_from_prompt_ids
from .loaders import load_external_model_and_tokenizer, load_model_and_tokenizer
from .metadata import (
    load_metadata,
    resolve_default_instruction,
    resolve_loader_family,
    resolve_reasoning_mode,
    resolve_reasoning_suffix_style,
    resolve_supports_multimodal_wrapper,
    resolve_template_name,
)
from .policies import build_chat_reasoning_policy


logger = get_logger("backends.llamafactory.chat_runner")

_LLAMAFACTORY_BACKEND = "llamafactory"
class LlamaFactoryClassificationChatRunner:
    """负责 LlamaFactory 分类模型的单轮聊天推理。"""

    def __init__(self, model_path_manager: ModelPathManager | None = None) -> None:
        self.model_path_manager = model_path_manager or get_model_path_manager()

    def _require_internal_task_name(self, spec: ChatSpec) -> str:
        if spec.task_name is None or not spec.task_name.strip():
            raise ValueError("内部模型推理要求提供 task_name。")
        return spec.task_name

    def _require_internal_run_id(self, spec: ChatSpec) -> str:
        if spec.run_id is None or not spec.run_id.strip():
            raise ValueError("内部模型推理要求提供 run_id。")
        return spec.run_id

    def _has_internal_prompt_source(self, spec: ChatSpec) -> bool:
        return bool(spec.task_name and spec.task_name.strip() and spec.run_id and spec.run_id.strip())

    def _load_internal_prompt_metadata(self, spec: ChatSpec) -> dict[str, Any]:
        task_name = self._require_internal_task_name(spec)
        run_id = self._require_internal_run_id(spec)
        model_layout = self.model_path_manager.build_layout(_LLAMAFACTORY_BACKEND, task_name, run_id)
        return load_metadata(model_layout.metadata_path)

    def _resolve_instruction(self, spec: ChatSpec, metadata: dict[str, Any]) -> str:
        if spec.instruction is not None:
            normalized_instruction = spec.instruction.strip()
            if not normalized_instruction:
                raise ValueError("instruction 不能为空字符串。")
            return normalized_instruction
        return resolve_default_instruction(metadata)

    def _resolve_external_instruction(self, spec: ChatSpec) -> str:
        if spec.instruction is not None:
            normalized_instruction = spec.instruction.strip()
            if not normalized_instruction:
                raise ValueError("instruction 不能为空字符串。")
            return normalized_instruction

        if not self._has_internal_prompt_source(spec):
            return ""

        metadata = self._load_internal_prompt_metadata(spec)
        return resolve_default_instruction(metadata)

    def _resolve_external_reasoning_mode(self, spec: ChatSpec) -> str | None:
        if spec.reasoning_mode is not None:
            return spec.reasoning_mode
        if not self._has_internal_prompt_source(spec):
            return None
        metadata = self._load_internal_prompt_metadata(spec)
        return resolve_reasoning_mode(metadata)

    def _resolve_external_template_name(self, spec: ChatSpec) -> str | None:
        if spec.template_name is not None:
            return spec.template_name
        if not self._has_internal_prompt_source(spec):
            return None
        metadata = self._load_internal_prompt_metadata(spec)
        return resolve_template_name(metadata)

    def _resolve_external_reasoning_suffix_style(self, spec: ChatSpec) -> str | None:
        if spec.reasoning_suffix_style is not None:
            return spec.reasoning_suffix_style
        if not self._has_internal_prompt_source(spec):
            return None
        metadata = self._load_internal_prompt_metadata(spec)
        return resolve_reasoning_suffix_style(metadata)

    def _resolve_internal_reasoning_mode(self, spec: ChatSpec, metadata: dict[str, Any]) -> str | None:
        metadata_reasoning_mode = resolve_reasoning_mode(metadata)
        if spec.reasoning_mode is None:
            return metadata_reasoning_mode
        if metadata_reasoning_mode is not None and spec.reasoning_mode != metadata_reasoning_mode:
            raise ValueError(
                f"内部 run chat 的 --reasoning-mode={spec.reasoning_mode} 与训练 metadata 中的 "
                f"reasoning_mode={metadata_reasoning_mode} 不一致。"
            )
        return spec.reasoning_mode

    def _resolve_external_loader_family(self, spec: ChatSpec) -> str:
        if not self._has_internal_prompt_source(spec):
            return "causal_lm"
        metadata = self._load_internal_prompt_metadata(spec)
        return resolve_loader_family(metadata)

    def _resolve_external_supports_multimodal_wrapper(self, spec: ChatSpec) -> bool:
        if not self._has_internal_prompt_source(spec):
            return False
        metadata = self._load_internal_prompt_metadata(spec)
        return resolve_supports_multimodal_wrapper(metadata)

    def build_plan(self, spec: ChatSpec) -> RunPlan:
        if spec.external_model_path is not None:
            prompt_source_metadata = self._load_internal_prompt_metadata(spec) if self._has_internal_prompt_source(spec) else None
            effective_prompt_engine = spec.prompt_engine or "native"
            template_name = self._resolve_external_template_name(spec) if effective_prompt_engine == "llamafactory" else None
            reasoning_mode = self._resolve_external_reasoning_mode(spec) if effective_prompt_engine == "llamafactory" else None
            reasoning_suffix_style = (
                self._resolve_external_reasoning_suffix_style(spec) if effective_prompt_engine == "llamafactory" else None
            )
            reasoning_policy = build_chat_reasoning_policy(
                prompt_engine=effective_prompt_engine,
                template=template_name,
                reasoning_mode=reasoning_mode,
                reasoning_suffix_style=reasoning_suffix_style,
                enable_thinking=spec.enable_thinking,
            )
            return RunPlan(
                stage="chat",
                summary="执行外部本地 GPT 类模型的单轮聊天推理。",
                inputs=asdict(spec),
                outputs={
                    "external_model_path": spec.external_model_path,
                    "prompt_source_metadata": (
                        str(
                            self.model_path_manager.build_layout(
                                _LLAMAFACTORY_BACKEND,
                                self._require_internal_task_name(spec),
                                self._require_internal_run_id(spec),
                            ).metadata_path
                        )
                        if prompt_source_metadata is not None
                        else None
                    ),
                    "reasoning_policy": reasoning_policy.to_payload(),
                },
                notes=[
                    "外部本地模型固定走 llamafactory backend 的本地加载推理路径。",
                    "external chat 默认使用 native prompt-engine；显式传 --prompt-engine=llamafactory 时才走 LlamaFactory 模板链。",
                    "若未传 instruction，可额外提供 task_name/run_id，从内部已训练模型 metadata 复用 prompt。",
                    (
                        f"当前 external chat 使用 prompt_engine={effective_prompt_engine}，"
                        f"reasoning_control={reasoning_policy.reasoning_control}。"
                    ),
                ],
            )

        task_name = self._require_internal_task_name(spec)
        run_id = self._require_internal_run_id(spec)
        model_layout = self.model_path_manager.build_layout(_LLAMAFACTORY_BACKEND, task_name, run_id)
        metadata = load_metadata(model_layout.metadata_path)
        reasoning_mode = self._resolve_internal_reasoning_mode(spec, metadata)
        reasoning_suffix_style = resolve_reasoning_suffix_style(metadata)
        template_name = resolve_template_name(metadata)
        reasoning_policy = build_chat_reasoning_policy(
            prompt_engine="llamafactory",
            template=template_name,
            reasoning_mode=reasoning_mode,
            reasoning_suffix_style=reasoning_suffix_style,
            enable_thinking=None,
        )
        return RunPlan(
            stage="chat",
            summary="执行 LlamaFactory 模型的单轮聊天推理。",
            inputs=asdict(spec),
            outputs={
                "metadata": str(model_layout.metadata_path),
                "merged_model_dir": str(model_layout.merged_model_dir),
                "lora_dir": str(model_layout.lora_dir),
                "reasoning_policy": reasoning_policy.to_payload(),
            },
            notes=[
                "若未传 instruction，则会基于 metadata.label_names 自动构建统一分类 instruction。",
                "当前仅支持 instruction + 单轮 message 的一次性生成。",
                (
                    f"当前 run chat 使用 template={template_name}，reasoning_mode={reasoning_mode}，"
                    f"reasoning_control={reasoning_policy.reasoning_control}。"
                ),
            ],
        )

    def run(self, spec: ChatSpec) -> ChatResult:
        try:
            if spec.external_model_path is not None:
                logger.info("开始外部本地模型 chat 推理: model=%s", spec.external_model_path)
                renderer = select_external_chat_prompt_renderer(spec)
                template_name = self._resolve_external_template_name(spec) if renderer.prompt_engine == "llamafactory" else None
                reasoning_mode = self._resolve_external_reasoning_mode(spec) if renderer.prompt_engine == "llamafactory" else None
                reasoning_suffix_style = (
                    self._resolve_external_reasoning_suffix_style(spec) if renderer.prompt_engine == "llamafactory" else None
                )
                runtime = load_external_model_and_tokenizer(
                    spec.external_model_path,
                    template_name=template_name,
                    enable_thinking=spec.enable_thinking,
                    loader_family=self._resolve_external_loader_family(spec),
                    supports_multimodal_wrapper=self._resolve_external_supports_multimodal_wrapper(spec),
                )
                instruction = self._resolve_external_instruction(spec)
                rendered_prompt = renderer.render(
                    runtime=runtime,
                    spec=spec,
                    instruction=instruction,
                    template_name=template_name,
                    reasoning_mode=reasoning_mode,
                    reasoning_suffix_style=reasoning_suffix_style,
                )
                generated_outputs, _ = generate_outputs_from_prompt_ids(
                    runtime=runtime,
                    prompt_id_batches=rendered_prompt.prompt_id_batches,
                    max_new_tokens=spec.max_new_tokens,
                    batch_size=1,
                )
                generated_output = generated_outputs[0] if generated_outputs else None
                output_text = generated_output.text if generated_output is not None else ""
                payload = {
                    "backend": spec.backend,
                    "external_model_path": spec.external_model_path,
                    "prompt_engine": rendered_prompt.prompt_engine,
                    "task_name": spec.task_name,
                    "run_id": spec.run_id,
                    "message": spec.message,
                    "instruction": rendered_prompt.instruction,
                    "template": rendered_prompt.template_name,
                    "reasoning_mode": rendered_prompt.reasoning_mode,
                    "reasoning_suffix_style": rendered_prompt.reasoning_suffix_style,
                    "enable_thinking": spec.enable_thinking,
                    "output_text": output_text,
                    "finish_reason": generated_output.finish_reason if generated_output is not None else "",
                    "prompt_token_count": generated_output.prompt_token_count if generated_output is not None else 0,
                    "generated_token_count": generated_output.generated_token_count if generated_output is not None else 0,
                }
                return ChatResult(
                    stage="chat",
                    success=True,
                    message="外部本地模型单轮聊天推理完成。",
                    output_text=output_text,
                    payload=payload,
                )

            task_name = self._require_internal_task_name(spec)
            run_id = self._require_internal_run_id(spec)
            logger.info(
                "开始 LlamaFactory chat 推理: task=%s, run_id=%s, artifact_type=%s",
                task_name,
                run_id,
                spec.artifact_type,
            )
            model_layout = self.model_path_manager.build_layout(_LLAMAFACTORY_BACKEND, task_name, run_id)
            metadata = load_metadata(model_layout.metadata_path)
            runtime = load_model_and_tokenizer(
                artifact_type=spec.artifact_type,
                metadata=metadata,
                model_layout=model_layout,
            )
            instruction = self._resolve_instruction(spec, metadata)
            template_name = resolve_template_name(metadata)
            reasoning_mode = self._resolve_internal_reasoning_mode(spec, metadata)
            reasoning_suffix_style = resolve_reasoning_suffix_style(metadata)
            rendered_prompt = LlamaFactoryChatPromptRenderer().render(
                runtime=runtime,
                spec=spec,
                instruction=instruction,
                template_name=template_name,
                reasoning_mode=reasoning_mode,
                reasoning_suffix_style=reasoning_suffix_style,
            )
            generated_outputs, _ = generate_outputs_from_prompt_ids(
                runtime=runtime,
                prompt_id_batches=rendered_prompt.prompt_id_batches,
                max_new_tokens=spec.max_new_tokens,
                batch_size=1,
            )
            generated_output = generated_outputs[0] if generated_outputs else None
            output_text = generated_output.text if generated_output is not None else ""
            payload = {
                "backend": spec.backend,
                "prompt_engine": rendered_prompt.prompt_engine,
                "task_name": task_name,
                "run_id": run_id,
                "artifact_type": spec.artifact_type,
                "message": spec.message,
                "instruction": instruction,
                "template": rendered_prompt.template_name,
                "reasoning_mode": rendered_prompt.reasoning_mode,
                "reasoning_suffix_style": rendered_prompt.reasoning_suffix_style,
                "output_text": output_text,
                "finish_reason": generated_output.finish_reason if generated_output is not None else "",
                "prompt_token_count": generated_output.prompt_token_count if generated_output is not None else 0,
                "generated_token_count": generated_output.generated_token_count if generated_output is not None else 0,
            }
            return ChatResult(
                stage="chat",
                success=True,
                message="LlamaFactory 单轮聊天推理完成。",
                output_text=output_text,
                payload=payload,
            )
        except Exception as exc:
            logger.error("LlamaFactory chat 推理失败: %s", exc, exc_info=True)
            return ChatResult(
                stage="chat",
                success=False,
                message=f"LlamaFactory 单轮聊天推理失败: {exc}",
            )