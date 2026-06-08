"""TuneBench MCP Server。

按路由划分：
- execute: 阶段操作执行（数据准备、推理增强、结构化目标、训练、评测）
- manage: Workflow 生命周期管理（创建、预览、状态查询、日志）
- monitor: 系统监控工具（如显卡使用查询）
- assets: MCP Resources（asset 数据暴露）
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from tunebench_mcp.routes import assets, execute, manage, utility
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


# ── 注册各路由 ──────────────────────────────────────────────────────────
def _register_routes() -> None:
    """统一注册所有路由模块。"""
    manage.register(mcp)
    execute.register(mcp)
    utility.register(mcp)
    assets.register(mcp)


_register_routes()


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