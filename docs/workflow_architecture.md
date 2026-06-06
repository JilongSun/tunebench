# Workflow 架构说明

本文档描述 TuneBench 新增的 workflow 编排层，目标是为 MCP、状态查询、人工审核断点和异步任务执行提供统一基础。

## 设计目标

workflow 层解决的是“如何编排一条实验链路”，而不是替代既有训练与评测实现。

这层的职责包括：

- 维护 workflow 主状态。
- 维护环节运行记录与事件流。
- 统一管理人工审核断点。
- 以子进程方式启动具体环节执行。
- 为 MCP 查询与外部调度提供稳定的状态与日志入口。

## 模块结构

新增模块位于 `src/tunebench/workflow/`，主要包括：

- `models.py`：workflow、stage run、事件、运行时配置和环节请求契约。
- `paths.py`：workflow 状态文件、日志、request/result 文件路径管理。
- `store/base.py`：状态存储抽象接口。
- `store/memory.py`：内存态存储实现，适合测试与本地调试。
- `store/sqlite.py`：默认状态存储实现，使用 SQLAlchemy async + SQLite。
- `service.py`：workflow 应用层服务，负责创建 workflow、启动环节、审核状态推进和日志查询。
- `worker.py`：子进程 worker 入口，在独立进程中执行具体环节。

与之对应，`src/tunebench_mcp/` 负责把 workflow 服务暴露为 FastMCP 工具，并通过 `Makefile` 与 `scripts/` 提供部署入口。

## 状态模型

### workflow 状态

- `draft`：workflow 已创建，尚未启动任何环节。
- `running`：某个环节正在执行。
- `awaiting_review`：环节执行成功，等待人工审核。
- `ready_next`：上一环节已审核通过，可以进入下一环节。
- `failed`：某个环节执行失败。
- `rejected`：人工审核拒绝，workflow 被拦截。
- `completed`：workflow 已完成。

### 环节状态

- `pending`
- `running`
- `succeeded`
- `failed`
- `awaiting_review`
- `approved`
- `rejected`
- `skipped`

## 存储层

workflow 状态存储采用“统一抽象 + 多实现”模式：

- 默认实现为 SQLite。
- 备用实现为内存存储。

SQLite 存储包含三类核心记录：

- workflow 主记录
- 环节运行记录
- workflow 事件记录

这三类记录分别承载：

- workflow 基本信息、环节拓扑、运行时环境和主状态。
- 每次环节执行的 request、plan、result、pid、日志文件与时间戳。
- 状态变更、审核动作、环节切换和异常事件。

## 执行模型

workflow 控制层采用异步编排，环节执行采用子进程隔离：

1. `WorkflowService` 创建环节 request 文件。
2. 服务层为子进程注入环境变量，例如 `CUDA_VISIBLE_DEVICES`。
3. 服务层通过 `asyncio.create_subprocess_exec` 启动 worker。
4. worker 进程在独立解释器中导入 TuneBench 环节实现并执行。
5. 环节结果写入 `result.json`，日志写入 `output.log`。
6. 服务层回写状态存储，并决定进入审核、下一环节或失败状态。

这个模型的核心目的是保证训练相关模块只在子进程中加载，从而满足 GPU 环境变量必须在 `torch` 导入前设置的约束。

## 环节范围

workflow 统一支持以下环节：

- `prepare_dataset`
- `generate_reasoning`
- `build_structured_target`
- `train_model`
- `evaluate_model`

这五个环节现在都已经接入同一套状态存储、子进程执行和审核推进框架。

其中：

- `prepare_dataset`、`train_model`、`evaluate_model` 复用现有数据处理与后端执行器。
- `generate_reasoning`、`build_structured_target` 复用分类共享层执行器。

所有环节都可以通过 workflow 层完成：

- request 持久化
- worker 子进程启动
- result 回写
- 日志查询
- 审核通过或拒绝

## 与 CLI 的关系

workflow 层不替代 CLI。

两者关系如下：

- CLI 负责单次命令执行。
- workflow 层负责多次命令之间的状态编排与审核推进。
- worker 进程会在独立解释器中调用 TuneBench 既有执行对象，而不是让 MCP 主进程直接加载训练模块。

## 与 MCP 的关系

workflow 层是 MCP 的应用服务基础。后续 MCP 接口可以围绕以下能力展开：

- 创建 workflow
- 启动指定环节
- 查询 workflow 状态
- 查询环节日志
- 审核通过或拒绝某个环节

这样可以把"任务控制"和"环节执行"拆开，让人工审核、外部 agent 轮询和状态持久化共享同一套底层语义。

当前仓库已经提供 `src/tunebench_mcp/` 作为正式 MCP 接入层，并通过 `make run` / `make stop` 管理 streamable HTTP MCP 服务。
