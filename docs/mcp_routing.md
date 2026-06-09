# MCP Server 路由架构

本文档描述 TuneBench MCP Server 的路由划分、模块职责和扩展方式。

## 概述

基于 mcp-use 框架，MCP Server 将 TuneBench 的 workflow 能力暴露为 MCP 工具和资源，供外部 agent 调用。采用 `MCPServer` + `MCPRouter` 的模块化路由架构：

```
tunebench_mcp/
├── server.py           # MCPServer 实例 + router 汇总
├── adapters.py         # MCP 参数 → workflow 请求对象的协议适配
├── __main__.py         # 启动入口
├── debug_server.py     # 调试启动脚本
└── routes/             # 路由模块（每个导出 MCPRouter 实例）
    ├── __init__.py     # 路由包入口
    ├── shared.py       # 路由间共享（WorkflowService 闭包单例）
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
| `execute_run_prepare_dataset` | 启动数据准备环节 |
| `execute_run_generate_reasoning` | 启动 reasoning 数据增强环节 |
| `execute_run_build_structured_target` | 启动 structured target 构建环节 |
| `execute_run_train_model` | 启动训练环节 |
| `execute_run_evaluate_model` | 启动评测环节 |

执行链路：`MCP 工具 → adapters.py 构建请求对象 → WorkflowService → 子进程执行`。

### manage — Workflow 生命周期管理

负责 workflow 容器的增删改查与状态管理。

| 工具名 | 说明 |
|--------|------|
| `manage_preview_workflow` | 预览 workflow 计划 |
| `manage_create_workflow` | 创建新的 workflow |
| `manage_get_workflow_state` | 读取 workflow 当前状态 |
| `manage_tail_stage_log` | 读取环节日志尾部 |

**待扩展**：`delete_workflow`、`list_workflows` 等 CRUD 工具。

### utility — 辅助工具

提供各类辅助性质的工具，不限于监控。

| 工具名 | 说明 |
|--------|------|
| `utility_get_gpu_status` | 查询当前显卡使用情况（模拟方法，待接入真实数据） |

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

每个路由模块创建一个 `MCPRouter` 实例，使用 `@router.tool()` 或 `@router.resource()` 装饰器定义工具/资源。

`server.py` 创建 `MCPServer` 后，通过 `include_router` 统一注册：

```python
from mcp_use.server import MCPServer
from .routes.execute import router as execute_router
from .routes.manage import router as manage_router
from .routes.utility import router as utility_router
from .routes.assets import router as assets_router

mcp = MCPServer(name="TuneBench", version="0.1.0")

mcp.include_router(manage_router)
mcp.include_router(execute_router)
mcp.include_router(utility_router)
mcp.include_router(assets_router)
```

`MCPRouter` 的 `name` 属性会自动作为工具名前缀（如 `execute_run_train_model`）。

## 添加新路由

1. 在 `routes/` 下创建新模块文件（如 `routes/new_route.py`）。
2. 创建 `router = MCPRouter(name="new_route", ...)` 实例，用装饰器定义工具/资源。
3. 在 `server.py` 中导入并调用 `mcp.include_router(new_route_router)`。

## 执行链路

以 `run_train_model` 为例，完整调用链如下：

```
外部 MCP Client
    ↓
server.py (MCPServer 实例，接收请求)
    ↓
routes/execute.py (execute_run_train_model 工具函数)
    ↓
routes/shared.py (get_workflow_service 闭包获取 WorkflowService 单例)
    ↓
adapters.py (build_train_model_request 构建请求对象)
    ↓
WorkflowService.run_train_model()
    ↓
子进程执行训练
```

## 共享层

`routes/shared.py` 通过闭包管理 `WorkflowService` 单例，工具函数直接 `await get_workflow_service()` 获取，不再依赖 MCP Context 传参。

`adapters.py` 提供 MCP 参数到 workflow 请求对象的转换函数，保持与 `tunebench.workflow.models` 的协议对齐。

## Server + App 双模式

当前以纯 Server 运行：

```python
mcp.run(transport="streamable-http", host=..., port=...)
```

未来如需挂载到 FastAPI 应用，利用 `mcp.app`（底层 ASGI 应用）即可：

```python
from fastapi import FastAPI
app = FastAPI(...)
# 通过 lifespan 注入 mcp.session_manager._task_group
app.mount("/agent", mcp.app)
```

路由定义在 `MCPRouter` 中，两种模式共用同一套工具定义。
