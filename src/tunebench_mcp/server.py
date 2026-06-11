"""TuneBench MCP Server。

按路由划分：
- execute: 阶段操作执行（数据准备、推理增强、结构化目标、训练、评测）
- manage: Workflow 生命周期管理（创建、预览、状态查询、日志）
- utility: 辅助工具（如显卡使用查询）
- assets: MCP Resources（asset 数据暴露）
"""

from __future__ import annotations

import os

from mcp_use.server import MCPServer

from tunebench_mcp.routes import assets, execute, manage, utility
from tunebench.util import get_logger
from tunebench.util.logging import setup_logging


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
_MCP_DEBUG = os.getenv("TUNEBENCH_MCP_DEBUG", "false").lower() in ("true", "1", "yes")


mcp = MCPServer(
    name="TuneBench",
    version="0.1.0",
    instructions=(
        "提供 TuneBench workflow 的创建、operation 执行、状态查询和日志读取能力。"
    ),
    host=_MCP_HOST,
    port=_MCP_PORT,
    mcp_path=_MCP_HTTP_PATH,
)


# ── 注册各路由 ──────────────────────────────────────────────────────────
mcp.include_router(manage.router)
mcp.include_router(execute.router)
mcp.include_router(utility.router)
mcp.include_router(assets.router)


def main(debug: bool = False) -> int:
    """MCP server 启动入口。

    :param debug: 是否开启调试模式（启用 MCP inspector 页面）。
    """
    setup_logging()
    debug_mode = debug or _MCP_DEBUG
    logger.info(
        "启动 MCP server transport=streamable-http host=%s port=%s path=%s debug=%s",
        _MCP_HOST,
        _MCP_PORT,
        _MCP_HTTP_PATH,
        debug_mode,
    )
    if debug_mode:
        logger.info("调试模式已启用，可通过 http://%s:%s/inspector 访问 MCP Inspector", _MCP_HOST, _MCP_PORT)
    mcp.run(transport="streamable-http", debug=debug_mode)
    return 0