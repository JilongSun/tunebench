"""辅助工具路由：系统资源查询等辅助工具。

当前包含显卡使用查询等辅助能力。
"""

from __future__ import annotations

from typing import Any

from mcp_use.server import MCPRouter

router = MCPRouter(
    prefix="utility",
    tags=["utilities"],
)


@router.tool()
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

# TODO: 后续可添加其他辅助工具，如环境检查、路径转换、可用后端列表查询等

