"""MCP 与 workflow 之间的协议适配。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from tunebench.workflow.models import (
    BuildStructuredTargetRequest,
    EvaluateModelRequest,
    GenerateReasoningRequest,
    PrepareDatasetRequest,
    TrainModelRequest,
    WorkflowCreateRequest,
    WorkflowRuntimeConfig,
)


class _PayloadConvertible(Protocol):
    def to_payload(self) -> dict[str, Any]: ...


def build_workflow_create_request(
    task_name: str,
    backend: str,
    runtime: dict[str, Any] | None = None,
    run_id: str | None = None,
    enabled_stages: Sequence[str] | None = None,
    review_required_stages: Sequence[str] | None = None,
) -> WorkflowCreateRequest:
    return WorkflowCreateRequest(
        task_name=task_name,
        backend=backend,
        runtime=WorkflowRuntimeConfig.from_payload(runtime),
        run_id=run_id,
        enabled_stages=_normalize_stage_names(enabled_stages),
        review_required_stages=_normalize_stage_names(review_required_stages),
    )


def build_prepare_dataset_request(
    input_path: str,
    dataset_version: str,
    text_key: str,
    label_key: str,
    output_path: str | None = None,
    output_format: str = "jsonl",
    sheet_name: str = "0",
    validation_ratio: float = 0.0,
    split_seed: int = 42,
    is_test: bool = False,
    allowed_labels: Sequence[str] | None = None,
) -> PrepareDatasetRequest:
    return PrepareDatasetRequest(
        input_path=input_path,
        dataset_version=dataset_version,
        text_key=text_key,
        label_key=label_key,
        output_path=output_path,
        output_format=output_format,
        sheet_name=sheet_name,
        validation_ratio=validation_ratio,
        split_seed=split_seed,
        is_test=is_test,
        allowed_labels=tuple(str(value) for value in (allowed_labels or [])),
    )


def build_generate_reasoning_request(
    source_dataset_version: str,
    target_dataset_version: str,
    teacher_model: str,
    endpoint_url: str,
    label_profile: str = "l1_5class",
    prompt_version: str = "reasoning_v1",
    api_key_env_var: str = "TUNEBENCH_REASONING_API_KEY",
    max_concurrency: int = 5,
    request_timeout_seconds: float = 60.0,
    max_attempts: int = 2,
    enable_model_verify: bool = False,
    resume: bool = False,
    sample_limit: int | None = None,
    splits: Sequence[str] | None = None,
) -> GenerateReasoningRequest:
    return GenerateReasoningRequest(
        source_dataset_version=source_dataset_version,
        target_dataset_version=target_dataset_version,
        teacher_model=teacher_model,
        endpoint_url=endpoint_url,
        label_profile=label_profile,
        prompt_version=prompt_version,
        api_key_env_var=api_key_env_var,
        max_concurrency=max_concurrency,
        request_timeout_seconds=request_timeout_seconds,
        max_attempts=max_attempts,
        enable_model_verify=enable_model_verify,
        resume=resume,
        sample_limit=sample_limit,
        splits=tuple(str(value) for value in (splits or ["train", "validation"])),
    )


def build_structured_target_request(
    source_dataset_version: str,
    target_dataset_version: str,
    confidence: float = 0.9,
    splits: Sequence[str] | None = None,
) -> BuildStructuredTargetRequest:
    return BuildStructuredTargetRequest(
        source_dataset_version=source_dataset_version,
        target_dataset_version=target_dataset_version,
        confidence=confidence,
        splits=tuple(str(value) for value in (splits or ["train", "validation"])),
    )


def build_train_model_request(
    dataset_version: str,
    model_name: str | None = None,
    model_key: str | None = None,
    instruction: str | None = None,
    reasoning_mode: str | None = None,
    resume_lora: str | None = None,
    export_dir: str | None = None,
    num_labels: int | None = None,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    num_train_epochs: int = 3,
    max_sequence_length: int = 256,
    warmup_ratio: float = 0.0,
    seed: int = 42,
    lora: dict[str, Any] | None = None,
) -> TrainModelRequest:
    return TrainModelRequest.from_payload(
        {
            "dataset_version": dataset_version,
            "model_name": model_name,
            "model_key": model_key,
            "instruction": instruction,
            "reasoning_mode": reasoning_mode,
            "resume_lora": resume_lora,
            "export_dir": export_dir,
            "num_labels": num_labels,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_train_epochs": num_train_epochs,
            "max_sequence_length": max_sequence_length,
            "warmup_ratio": warmup_ratio,
            "seed": seed,
            "lora": lora,
        }
    )


def build_evaluate_model_request(
    dataset_version: str,
    artifact_type: str = "merged",
    batch_size: int = 8,
    max_sequence_length: int | None = None,
    max_new_tokens: int | None = None,
    prompt_engine: str | None = None,
    enable_thinking: bool | None = None,
    export_xlsx: bool = True,
) -> EvaluateModelRequest:
    return EvaluateModelRequest.from_payload(
        {
            "dataset_version": dataset_version,
            "artifact_type": artifact_type,
            "batch_size": batch_size,
            "max_sequence_length": max_sequence_length,
            "max_new_tokens": max_new_tokens,
            "prompt_engine": prompt_engine,
            "enable_thinking": enable_thinking,
            "export_xlsx": export_xlsx,
        }
    )


def to_payload(value: _PayloadConvertible) -> dict[str, Any]:
    return value.to_payload()


def build_stage_log_payload(stage_run_id: str, content: str) -> dict[str, Any]:
    return {
        "stage_run_id": stage_run_id,
        "content": content,
    }


def _normalize_stage_names(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values)