"""供部署脚本调用的 MCP 启动入口。"""

from .server import main


if __name__ == "__main__":
    raise SystemExit(main())