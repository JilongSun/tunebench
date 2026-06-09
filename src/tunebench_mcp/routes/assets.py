"""Assets 路由：通过 MCP Resources 能力暴露 asset 数据。

利用 MCP 的 resources 机制，将 assets 目录下的数据以资源 URI 的形式暴露给客户端。
"""

from __future__ import annotations

from pathlib import Path

from mcp_use.server import MCPRouter

# assets 目录位于项目根目录
_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"

router = MCPRouter(
    prefix="assets",
    tags=["resources"],
)


@router.resource("asset://models/list")
def list_model_assets() -> str:
    """列出所有可用的模型资产目录。"""
    models_dir = _ASSETS_DIR / "models"
    if not models_dir.is_dir():
        return "models 目录不存在。"
    entries = [str(p.relative_to(models_dir)) for p in models_dir.iterdir()]
    return "\n".join(entries) if entries else "models 目录为空。"


@router.resource("asset://data/list")
def list_data_assets() -> str:
    """列出所有可用的数据资产。"""
    data_dir = _ASSETS_DIR / "data"
    if not data_dir.is_dir():
        return "data 目录不存在。"
    entries = [str(p.relative_to(data_dir)) for p in data_dir.rglob("*") if p.is_file()]
    return "\n".join(entries) if entries else "data 目录为空。"


@router.resource("asset://read/{file_path}")
def read_asset_file(file_path: str) -> str:
    """读取 asset 目录下的指定文件内容。

    `file_path` 为相对于 assets 目录的路径。
    """
    target = (_ASSETS_DIR / file_path).resolve()
    # 安全校验：防止路径穿越
    if not str(target).startswith(str(_ASSETS_DIR.resolve())):
        return "错误：访问路径超出 assets 目录范围。"
    if not target.is_file():
        return f"文件不存在: {file_path}"
    return target.read_text(encoding="utf-8", errors="replace")

# TODO: 后续可根据需要添加更多资源模板，如 workflows 状态资源等

