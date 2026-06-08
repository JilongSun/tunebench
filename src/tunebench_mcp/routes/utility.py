"""系统监控路由：系统资源查询等监控工具。

当前包含显卡使用查询等监控能力。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    """将辅助工具注册到 MCP 实例。"""

    @mcp.tool()
    def get_gpu_status() -> dict[str, Any]:
        """查询当前显卡使用情况。

        返回 GPU 数量、显存占用、利用率等信息。
        """
        # TODO: 接入 nvidia-smi 或 pynvml 获取真实数据
        return {
            "gpu_count": 0,
            "gpus": [],
            "note": "此工具尚未接入真实 GPU 数据，返回模拟结果。",
        }

    # TODO: 后续可添加其他系统工具，如磁盘空间查询、进程监控等

