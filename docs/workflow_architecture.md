# Workflow 架构说明

本文档描述 TuneBench workflow 编排层的正式语义。当前设计把 workflow 视为实验容器，而不是不可回退的线性 stage 链。

## 设计目标

workflow 层解决的是“如何在一个实验容器里重复执行、留痕和查询多个 operation”，而不是替代既有训练与评测实现。

这层的职责包括：

- 维护 workflow 主状态。
- 维护每次 operation 的运行记录与事件流。
- 以子进程方式启动具体 operation 执行。
- 在启动前校验输入资源是否存在、输出是否会覆盖已有产物。
- 为 MCP 查询与外部调度提供稳定的状态与日志入口。

## 核心语义

### Workflow 是容器，不是固定流水线

创建 workflow 时，只确定：

- `task_name`
- `backend`
- `runtime`
- 当前允许使用的 operation 列表

workflow 不再持有固定 `run_id`，也不再表达“当前必须执行哪个下一环节”。

### Operation 是可重复执行的动作

当前支持的 operation 包括：

- `prepare_dataset`
- `generate_reasoning`
- `build_structured_target`
- `train_model`
- `evaluate_model`

这些 operation 可以在同一个 workflow 中多次执行。调度依据不再是线性顺序，而是本次请求声明的输入与输出版本。

例如：

- 训练完成后，仍然可以再次执行 `prepare_dataset`，只要新的 `dataset_version` 不会覆盖旧产物。
- 可以先准备训练集，再训练，再补做测试集准备，再执行评测。
- 是否允许启动，取决于输入资源存在性和输出覆盖风险，而不是“是不是下一步”。

## 状态模型

### workflow 状态

- `idle`：当前没有运行中的 operation。
- `running`：当前至少有一个 operation 正在执行。
- `failed`：最近一次 operation 失败，且当前没有运行中的 operation。

### operation 状态

- `pending`
- `running`
- `succeeded`
- `failed`

旧的 `awaiting_review`、`approved`、`rejected` 仍保留在兼容枚举里，但不再进入正式主流程。

## 存储层

workflow 状态存储采用“统一抽象 + 多实现”模式：

- 默认实现为 SQLite。
- 备用实现为内存存储。

当前核心记录包括：

- workflow 主记录
- operation 运行记录
- workflow 事件记录

其中：

- workflow 主记录保存容器级属性与聚合状态。
- operation 运行记录保存每次执行的 request、result、输入输出摘要、pid、日志与时间戳。
- workflow 事件记录保存启动、结束和状态变化事件。

SQLite 仍保留少量旧字段占位，以避免现有表结构直接失配，但这些字段已经不再代表正式领域语义。

## 执行模型

workflow 控制层采用异步编排，operation 执行采用子进程隔离：

1. `WorkflowService` 创建 operation request 文件。
2. 服务层为子进程注入环境变量，例如 `CUDA_VISIBLE_DEVICES`。
3. 服务层通过 `asyncio.create_subprocess_exec` 启动 worker。
4. worker 进程在独立解释器中导入 TuneBench 具体执行器并运行。
5. operation 结果写入 `result.json`，日志写入 `output.log`。
6. 服务层回写状态存储，并根据最近执行结果更新 workflow 聚合状态。

这个模型的核心目的是保证训练相关模块只在子进程中加载，从而满足 GPU 环境变量必须在 `torch` 导入前设置的约束。

## 启动校验策略

workflow 层当前在启动 operation 前执行三类检查：

- workflow 是否启用了该 operation。
- 当前 workflow 是否还有运行中的 operation。
- 本次 operation 的输入资源是否存在，输出资源是否会覆盖已有产物。

具体包括：

- `prepare_dataset`：目标 `dataset_version` 不能已存在。
- `generate_reasoning` / `build_structured_target`：源数据版本必须存在，目标数据版本不能已存在。
- `train_model`：输入 `dataset_version` 必须存在，目标 `run_id` 不能已存在。
- `evaluate_model`：输入 `dataset_version` 与目标 `run_id` 必须存在，且评测输出不能覆盖已有文件。

版本命名本身目前只做 skill 与文档推荐，不在 workflow 代码里做额外格式强制。

## 与 CLI 的关系

workflow 层不替代 CLI。

两者关系如下：

- CLI 负责单次命令执行。
- workflow 层负责跨多次命令的实验容器编排、状态留痕和异步调度。
- worker 进程会在独立解释器中调用 TuneBench 既有执行对象，而不是让 MCP 主进程直接加载训练模块。

## 与 MCP 的关系

workflow 层是 MCP 的应用服务基础。当前 MCP 接口围绕以下能力展开：

- 创建 workflow
- 启动指定 operation
- 查询 workflow 状态
- 查询 operation 日志

这样可以把“任务控制”和“具体执行”拆开，让外部 agent 通过显式的资源版本参数组织实验分支，而不是依赖隐式的线性 stage 推进。
