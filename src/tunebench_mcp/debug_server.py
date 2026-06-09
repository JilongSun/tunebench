#!/usr/bin/env python3
"""MCP Server 调试启动脚本。

用法：
    1. 在 server.py 中设置断点
    2. 在 VS Code 中对此文件按 F5 调试运行

注意：
    - 使用 .tb311 环境运行（在 VS Code 调试配置中指定 Python 解释器）
    - 确保已安装项目依赖
    - 环境变量 TUNEBENCH_MCP_HOST/PORT/PATH 可在启动前设置
"""

import sys
from pathlib import Path
from tunebench_mcp.server import main

# 将 src 加入 Python 路径
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


if __name__ == "__main__":
    # debug=True 启用 MCP Inspector，可通过 http://localhost:8888/inspector 访问
    sys.exit(main(debug=True))
