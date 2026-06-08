"""用于在 Python 调试器中调用 TuneBench CLI。"""

from __future__ import annotations

import os
import shlex
from collections.abc import Sequence


# 调试训练时可手动指定物理显卡，值应为外层机器上的物理卡号。
# 当前约定仅允许 4-7 号卡；设为 None 表示不改写当前环境。
DEBUG_CUDA_VISIBLE_DEVICES = "5"


if DEBUG_CUDA_VISIBLE_DEVICES is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = DEBUG_CUDA_VISIBLE_DEVICES
os.environ["TUNEBENCH_REASONING_API_KEY"] = (
    "your_api_key_here"  # 替换为实际的 API Key，或确保在环境变量中已设置。
)
from tunebench import cli  # noqa: E402


# 可在调试时直接修改这行，模拟命令行参数。
DEBUG_COMMAND = "your_cli_command_here"  # 替换为实际的 CLI 命令。


def run_cli_args(args: Sequence[str]) -> int:
    """以参数列表方式调用 CLI 主入口。"""
    return cli.main(args)


def run_cli_command(command: str) -> int:
    """以命令字符串方式调用 CLI 主入口。"""
    return run_cli_args(shlex.split(command))


def main() -> int:
    """调试入口。"""
    return run_cli_command(DEBUG_COMMAND)


if __name__ == "__main__":
    raise SystemExit(main())
