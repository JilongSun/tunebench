"""路由模块共享的辅助函数。

通过闭包管理 WorkflowService 单例，工具函数不再需要 ctx 传参。
"""

from __future__ import annotations

import logging

from tunebench.workflow.service import WorkflowService

logger = logging.getLogger("mcp.shared")

_service: WorkflowService | None = None


async def get_workflow_service() -> WorkflowService:
    """获取 WorkflowService 单例（延迟初始化）。"""
    global _service
    if _service is None:
        _service = WorkflowService()
        await _service.initialize()
        logger.info("WorkflowService 已初始化。")
    return _service
