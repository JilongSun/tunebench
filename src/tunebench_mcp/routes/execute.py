"""阶段操作执行路由：数据准备、推理增强、结构化目标、训练、评测。

这些是 TuneBench 的核心 operation 执行能力。
"""

from __future__ import annotations

from typing import Any

from mcp_use.server import MCPRouter

from tunebench_mcp.adapters import (
    build_evaluate_model_request,
    build_generate_reasoning_request,
    build_prepare_dataset_request,
    build_structured_target_request,
    build_train_model_request,
    to_payload,
)
from tunebench_mcp.routes.shared import get_workflow_service
from tunebench.util import get_logger

logger = get_logger("mcp.execute")

router = MCPRouter(
    prefix="execute",
    tags=["stage-operations"],
)


@router.tool()
async def run_prepare_dataset(
    workflow_id: str,
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
    allowed_labels: list[str] | None = None,
) -> dict[str, Any]:
    """启动数据准备环节。

    用于把原始表格整理为标准数据版本。`text_key` 和 `label_key`
    指定输入列，`validation_ratio` 与 `is_test` 控制数据切分用途。
    """
    logger.info("启动 prepare_dataset: workflow=%s, dataset_version=%s", workflow_id, dataset_version)
    service = await get_workflow_service()
    stage_run = await service.run_prepare_dataset(
        workflow_id,
        build_prepare_dataset_request(
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
            allowed_labels=allowed_labels,
        ),
    )
    return to_payload(stage_run)


@router.tool()
async def run_generate_reasoning(
    workflow_id: str,
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
    splits: list[str] | None = None,
) -> dict[str, Any]:
    """启动 reasoning 数据增强环节。

    该接口基于已有数据版本调用教师模型生成 reasoning。`teacher_model`
    与 `endpoint_url` 指定增强来源，`splits` 用于限制处理的数据切分。
    """
    logger.info("启动 generate_reasoning: workflow=%s, target=%s", workflow_id, target_dataset_version)
    service = await get_workflow_service()
    stage_run = await service.run_generate_reasoning(
        workflow_id,
        build_generate_reasoning_request(
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
            splits=splits,
        ),
    )
    return to_payload(stage_run)


@router.tool()
async def run_build_structured_target(
    workflow_id: str,
    source_dataset_version: str,
    target_dataset_version: str,
    confidence: float = 0.9,
    splits: list[str] | None = None,
) -> dict[str, Any]:
    """启动 structured target 构建环节。

    用于把 reasoning 数据继续转换为更稳定的结构化训练目标。
    `confidence` 用于控制结构化目标的置信阈值。
    """
    logger.info("启动 build_structured_target: workflow=%s, target=%s", workflow_id, target_dataset_version)
    service = await get_workflow_service()
    stage_run = await service.run_build_structured_target(
        workflow_id,
        build_structured_target_request(
            source_dataset_version=source_dataset_version,
            target_dataset_version=target_dataset_version,
            confidence=confidence,
            splits=splits,
        ),
    )
    return to_payload(stage_run)


@router.tool()
async def run_train_model(
    workflow_id: str,
    dataset_version: str,
    run_id: str,
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
) -> dict[str, Any]:
    """启动训练环节。

    用于发起分类训练或继续训练。`dataset_version` 指定训练数据版本，
    `run_id` 是本次训练要写入的模型版本标识，`model_name` / `model_key`
    用于选择底座模型，`resume_lora` 用于继续训练。
    """
    logger.info("启动 train_model: workflow=%s, dataset_version=%s, run_id=%s", workflow_id, dataset_version, run_id)
    service = await get_workflow_service()
    stage_run = await service.run_train_model(
        workflow_id,
        build_train_model_request(
            dataset_version=dataset_version,
            run_id=run_id,
            model_name=model_name,
            model_key=model_key,
            instruction=instruction,
            reasoning_mode=reasoning_mode,
            resume_lora=resume_lora,
            export_dir=export_dir,
            num_labels=num_labels,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            max_sequence_length=max_sequence_length,
            warmup_ratio=warmup_ratio,
            seed=seed,
            lora=lora,
        ),
    )
    return to_payload(stage_run)


@router.tool()
async def run_evaluate_model(
    workflow_id: str,
    dataset_version: str,
    run_id: str,
    artifact_type: str = "merged",
    batch_size: int = 8,
    max_sequence_length: int | None = None,
    max_new_tokens: int | None = None,
    prompt_engine: str | None = None,
    enable_thinking: bool | None = None,
    export_xlsx: bool = True,
) -> dict[str, Any]:
    """启动评测环节。

    用于对训练产物执行独立评测。`run_id` 指定评测目标模型版本，`artifact_type` 指定评测对象，
    `prompt_engine` 与 `enable_thinking` 主要影响 LlamaFactory/Qwen 路径的评测渲染方式。
    """
    logger.info("启动 evaluate_model: workflow=%s, dataset_version=%s, run_id=%s", workflow_id, dataset_version, run_id)
    service = await get_workflow_service()
    stage_run = await service.run_evaluate_model(
        workflow_id,
        build_evaluate_model_request(
            dataset_version=dataset_version,
            run_id=run_id,
            artifact_type=artifact_type,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
            max_new_tokens=max_new_tokens,
            prompt_engine=prompt_engine,
            enable_thinking=enable_thinking,
            export_xlsx=export_xlsx,
        ),
    )
    return to_payload(stage_run)
