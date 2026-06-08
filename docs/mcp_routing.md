# MCP Server 路由架构

本文档描述 TuneBench MCP Server 的路由划分、模块职责和扩展方式。

## 概述

MCP Server 将 TuneBench 的 workflow 能力暴露为 MCP 工具和资源，供外部 agent 调用。为便于后续功能扩展，采用路由划分架构：

```
tunebench_mcp/
├── server.py           # Server 实例、lifespan、路由注册汇总
├── adapters.py         # MCP 参数 → workflow 请求对象的协议适配
├── __main__.py         # 启动入口
├── debug_server.py     # 调试启动脚本
└── routes/             # 路由模块
    ├── __init__.py     # 路由包入口
    ├── shared.py       # 路由间共享辅助函数
    ├── execute.py      # 阶段操作执行路由
    ├── manage.py       # Workflow 生命周期管理路由
    ├── utility.py      # 辅助工具路由
    └── assets.py       # MCP Resources 路由
```

## 路由划分

### execute — 阶段操作执行

负责 TuneBench 核心 operation 的启动与执行。

| 工具名 | 说明 |
|--------|------|
| `run_prepare_dataset` | 启动数据准备环节 |
| `run_generate_reasoning` | 启动 reasoning 数据增强环节 |
| `run_build_structured_target` | 启动 structured target 构建环节 |
| `run_train_model` | 启动训练环节 |
| `run_evaluate_model` | 启动评测环节 |

这些工具的执行链路为：`MCP 工具 → adapters.py 构建请求对象 → WorkflowService → 子进程执行`。

### manage — Workflow 生命周期管理

负责 workflow 容器的增删改查与状态管理。

| 工具名 | 说明 |
|--------|------|
| `preview_workflow` | 预览 workflow 计划 |
| `create_workflow` | 创建新的 workflow |
| `get_workflow_state` | 读取 workflow 当前状态 |
| `tail_stage_log` | 读取环节日志尾部 |

**待扩展**：`delete_workflow`、`list_workflows` 等 CRUD 工具。

### utility — 辅助工具

提供各类辅助性质的工具，不限于监控。

| 工具名 | 说明 |
|--------|------|
| `get_gpu_status` | 查询当前显卡使用情况（模拟方法，待接入真实数据） |

**待扩展**：环境检查、路径转换、可用后端列表查询等辅助工具。

### assets — MCP Resources

通过 MCP Resources 机制暴露 assets 目录下的数据资产。

| 资源 URI | 说明 |
|----------|------|
| `asset://models/list` | 列出所有可用的模型资产目录 |
| `asset://data/list` | 列出所有可用的数据资产文件 |
| `asset://read/{file_path}` | 读取 assets 目录下指定文件内容 |

**待扩展**：workflows 状态资源等。

## 注册机制

每个路由模块导出一个 `register(mcp: FastMCP) -> None` 函数，在函数内部使用 `@mcp.tool()` 或 `@mcp.resource()` 装饰器注册工具或资源。

`server.py` 在模块加载时调用 `_register_routes()` 统一注册所有路由：

```python
def _register_routes() -> None:
    manage.register(mcp)
    execute.register(mcp)
    utility.register(mcp)
    assets.register(mcp)

_register_routes()
```

## 添加新路由

1. 在 `routes/` 下创建新模块文件（如 `routes/new_route.py`）。
2. 实现 `register(mcp: FastMCP) -> None` 函数。
3. 在 `routes/__init__.py` 中导入并导出新模块。
4. 在 `server.py` 的 `_register_routes()` 中调用 `new_route.register(mcp)`。

## 执行链路

以 `run_train_model` 为例，完整调用链如下：

```
外部 MCP Client
    ↓
server.py (FastMCP 实例，接收请求)
    ↓
routes/execute.py (run_train_model 工具函数)
    ↓
routes/shared.py (_get_workflow_service 获取 WorkflowService)
    ↓
adapters.py (build_train_model_request 构建请求对象)
    ↓
WorkflowService.run_train_model()
    ↓
子进程执行训练
```

## 共享层

`routes/shared.py` 提供路由间共用的辅助函数，当前包含：

- `_get_workflow_service(ctx)`：从 MCP 上下文提取 WorkflowService 实例。

`adapters.py` 提供 MCP 参数到 workflow 请求对象的转换函数，保持与 `tunebench.workflow.models` 的协议对齐。
