"""TuneBench MCP Server。"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context, FastMCP

from .adapters import (
    build_evaluate_model_request,
    build_generate_reasoning_request,
    build_prepare_dataset_request,
    build_stage_log_payload,
    build_structured_target_request,
    build_train_model_request,
    build_workflow_create_request,
    to_payload,
)
from tunebench.util import get_logger

if TYPE_CHECKING:
    from tunebench.workflow.service import WorkflowService


logger = get_logger("mcp.server")


def _read_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f"环境变量 {name} 必须是整数。") from exc


def _normalize_http_path(value: str) -> str:
    if not value:
        return "/mcp"
    return value if value.startswith("/") else f"/{value}"


_MCP_HOST = os.getenv("TUNEBENCH_MCP_HOST", "127.0.0.1")
_MCP_PORT = _read_env_int("TUNEBENCH_MCP_PORT", 8888)
_MCP_HTTP_PATH = _normalize_http_path(os.getenv("TUNEBENCH_MCP_PATH", "/mcp"))


@asynccontextmanager
async def _server_lifespan(_server: FastMCP) -> AsyncIterator[WorkflowService]:
    """管理 MCP 生命周期内共享的 workflow 服务。"""
    from tunebench.workflow.service import WorkflowService

    service = WorkflowService()
    await service.initialize()
    yield service


mcp = FastMCP(
    "TuneBench",
    instructions=(
        "提供 TuneBench workflow 的创建、operation 执行、状态查询和日志读取能力。"
    ),
    host=_MCP_HOST,
    port=_MCP_PORT,
    streamable_http_path=_MCP_HTTP_PATH,
    json_response=True,
    lifespan=_server_lifespan,
)


def _get_workflow_service(ctx: Context) -> WorkflowService:
    """从 MCP 上下文中提取 workflow 服务。"""
    from tunebench.workflow.service import WorkflowService

    service = ctx.request_context.lifespan_context
    if not isinstance(service, WorkflowService):
        raise RuntimeError("workflow 服务未正确初始化。")
    return service


@mcp.tool()
async def preview_workflow(
    task_name: str,
    backend: str,
    ctx: Context,
    runtime: dict[str, Any] | None = None,
    enabled_stages: list[str] | None = None,
) -> dict[str, Any]:
    """预览 workflow 计划。

    用于在真正创建 workflow 前确认后端、启用 operation 范围和运行时配置。
    """
    await ctx.info(f"预览 workflow: task={task_name}, backend={backend}")
    service = _get_workflow_service(ctx)
    preview = await service.preview_workflow(
        build_workflow_create_request(
            task_name=task_name,
            backend=backend,
            runtime=runtime,
            enabled_stages=enabled_stages,
        )
    )
    return to_payload(preview)


@mcp.tool()
async def create_workflow(
    task_name: str,
    backend: str,
    ctx: Context,
    runtime: dict[str, Any] | None = None,
    enabled_stages: list[str] | None = None,
) -> dict[str, Any]:
    """创建新的 workflow。

    该接口只创建实验容器，不再绑定某个固定 run_id。
    后续每次 operation 执行都显式携带自己的输入与输出版本标识。
    """
    await ctx.info(f"创建 workflow: task={task_name}, backend={backend}")
    service = _get_workflow_service(ctx)
    snapshot = await service.create_workflow(
        build_workflow_create_request(
            task_name=task_name,
            backend=backend,
            runtime=runtime,
            enabled_stages=enabled_stages,
        )
    )
    return to_payload(snapshot)


@mcp.tool()
async def run_prepare_dataset(
    workflow_id: str,
    input_path: str,
    dataset_version: str,
    text_key: str,
    label_key: str,
    ctx: Context,
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
    await ctx.info(f"启动 prepare_dataset: workflow={workflow_id}, dataset_version={dataset_version}")
    service = _get_workflow_service(ctx)
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


@mcp.tool()
async def run_generate_reasoning(
    workflow_id: str,
    source_dataset_version: str,
    target_dataset_version: str,
    teacher_model: str,
    endpoint_url: str,
    ctx: Context,
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
    await ctx.info(f"启动 generate_reasoning: workflow={workflow_id}, target={target_dataset_version}")
    service = _get_workflow_service(ctx)
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


@mcp.tool()
async def run_build_structured_target(
    workflow_id: str,
    source_dataset_version: str,
    target_dataset_version: str,
    ctx: Context,
    confidence: float = 0.9,
    splits: list[str] | None = None,
) -> dict[str, Any]:
    """启动 structured target 构建环节。

    用于把 reasoning 数据继续转换为更稳定的结构化训练目标。
    `confidence` 用于控制结构化目标的置信阈值。
    """
    await ctx.info(f"启动 build_structured_target: workflow={workflow_id}, target={target_dataset_version}")
    service = _get_workflow_service(ctx)
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


@mcp.tool()
async def run_train_model(
    workflow_id: str,
    dataset_version: str,
    run_id: str,
    ctx: Context,
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
    await ctx.info(f"启动 train_model: workflow={workflow_id}, dataset_version={dataset_version}, run_id={run_id}")
    service = _get_workflow_service(ctx)
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


@mcp.tool()
async def run_evaluate_model(
    workflow_id: str,
    dataset_version: str,
    run_id: str,
    ctx: Context,
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
    await ctx.info(f"启动 evaluate_model: workflow={workflow_id}, dataset_version={dataset_version}, run_id={run_id}")
    service = _get_workflow_service(ctx)
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


@mcp.tool()
async def get_workflow_state(workflow_id: str, ctx: Context, event_limit: int = 50) -> dict[str, Any]:
    """读取 workflow 当前状态。

    返回 workflow 主状态、各次 operation 运行记录和最近事件，适合外部 agent 轮询。
    """
    service = _get_workflow_service(ctx)
    snapshot = await service.get_workflow_state(workflow_id, event_limit=event_limit)
    return to_payload(snapshot)


@mcp.tool()
async def tail_stage_log(stage_run_id: str, ctx: Context, max_bytes: int = 8192) -> dict[str, Any]:
    """读取环节日志尾部。

    用于在不下载完整日志文件的情况下查看最近执行输出。
    """
    service = _get_workflow_service(ctx)
    log_tail = await service.tail_stage_log(stage_run_id, max_bytes=max_bytes)
    return build_stage_log_payload(stage_run_id, log_tail)


def main() -> int:
    """MCP server 启动入口。"""
    logger.info(
        "启动 MCP server transport=streamable-http host=%s port=%s path=%s",
        _MCP_HOST,
        _MCP_PORT,
        _MCP_HTTP_PATH,
    )
    mcp.run(transport="streamable-http")
    return 0