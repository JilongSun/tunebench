"""路由模块共享的辅助函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.fastmcp import Context
from tunebench.workflow.service import WorkflowService

if TYPE_CHECKING:
    pass


def _get_workflow_service(ctx: Context) -> WorkflowService:
    """从 MCP 上下文中提取 workflow 服务。"""
    service = ctx.request_context.lifespan_context
    if not isinstance(service, WorkflowService):
        raise RuntimeError("workflow 服务未正确初始化。")
    return service
