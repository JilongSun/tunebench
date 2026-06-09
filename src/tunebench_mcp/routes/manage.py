"""Workflow 生命周期管理路由：创建、预览、状态查询、日志读取。

负责 workflow 的增删改查与状态管理。
"""

from __future__ import annotations

import logging
from typing import Any

from mcp_use.server import MCPRouter

from tunebench_mcp.adapters import (
    build_stage_log_payload,
    build_workflow_create_request,
    to_payload,
)
from tunebench_mcp.routes.shared import get_workflow_service

logger = logging.getLogger("mcp.manage")

router = MCPRouter(
    prefix="manage",
    tags=["workflow-management"],
)


@router.tool()
async def preview_workflow(
    task_name: str,
    backend: str,
    runtime: dict[str, Any] | None = None,
    enabled_stages: list[str] | None = None,
) -> dict[str, Any]:
    """预览 workflow 计划。

    用于在真正创建 workflow 前确认后端、启用 operation 范围和运行时配置。
    """
    logger.info("预览 workflow: task=%s, backend=%s", task_name, backend)
    service = await get_workflow_service()
    preview = await service.preview_workflow(
        build_workflow_create_request(
            task_name=task_name,
            backend=backend,
            runtime=runtime,
            enabled_stages=enabled_stages,
        )
    )
    return to_payload(preview)


@router.tool()
async def create_workflow(
    task_name: str,
    backend: str,
    runtime: dict[str, Any] | None = None,
    enabled_stages: list[str] | None = None,
) -> dict[str, Any]:
    """创建新的 workflow。

    该接口只创建实验容器，不再绑定某个固定 run_id。
    后续每次 operation 执行都显式携带自己的输入与输出版本标识。
    """
    logger.info("创建 workflow: task=%s, backend=%s", task_name, backend)
    service = await get_workflow_service()
    snapshot = await service.create_workflow(
        build_workflow_create_request(
            task_name=task_name,
            backend=backend,
            runtime=runtime,
            enabled_stages=enabled_stages,
        )
    )
    return to_payload(snapshot)


@router.tool()
async def get_workflow_state(
    workflow_id: str, event_limit: int = 50
) -> dict[str, Any]:
    """读取 workflow 当前状态。

    返回 workflow 主状态、各次 operation 运行记录和最近事件，适合外部 agent 轮询。
    """
    logger.info("查询 workflow 状态: workflow=%s", workflow_id)
    service = await get_workflow_service()
    snapshot = await service.get_workflow_state(workflow_id, event_limit=event_limit)
    return to_payload(snapshot)


@router.tool()
async def tail_stage_log(
    stage_run_id: str, max_bytes: int = 8192
) -> dict[str, Any]:
    """读取环节日志尾部。

    用于在不下载完整日志文件的情况下查看最近执行输出。
    """
    logger.info("读取日志尾部: stage_run=%s", stage_run_id)
    service = await get_workflow_service()
    log_tail = await service.tail_stage_log(stage_run_id, max_bytes=max_bytes)
    return build_stage_log_payload(stage_run_id, log_tail)

# TODO: 后续可添加 delete_workflow、list_workflows 等 CRUD 工具

