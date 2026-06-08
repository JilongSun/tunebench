# MCP 工具接口说明

本文档说明每个 MCP 工具的作用、在整体流程中的位置，以及关键参数含义。

## 整体调用流程

```
外部 Agent → MCP 工具 → Workflow 引擎 → 底层训练/评测模块
```

你调用 MCP 工具后，请求会被转换为 workflow operation，由 workflow 引擎负责状态管理、子进程执行和资源校验。具体的训练、评测、数据处理由底层模块执行。

## 工具清单

### `preview_workflow`

**流程位置**：创建 workflow 前的预览步骤。

**作用**：在真正创建 workflow 前，预览可用 operation 范围与运行时配置。

**关键参数**：

- `task_name`：任务名。
- `backend`：后端类型，通常为 `bert` 或 `llamafactory`。
- `runtime`：运行时配置载荷（GPU 设备、环境变量等），详见 [复杂参数详解 - runtime](./complex-params.md#runtime---运行时配置)。
- `enabled_stages`：本次允许使用的 operation 列表。

### `create_workflow`

**流程位置**：workflow 创建入口。

**作用**：创建 workflow 主记录，作为实验容器的控制平面入口。

**关键参数**：

- 与 `preview_workflow` 基本一致。

### `run_prepare_dataset`

**流程位置**：第一个数据环节。

**作用**：触发数据准备环节，生成训练集、验证集或测试集版本。

**关键参数**：详见 [复杂参数详解 - 数据准备参数](./complex-params.md#数据准备参数run_prepare_dataset-顶层参数)。

核心必填参数：`workflow_id`、`input_path`、`dataset_version`、`text_key`、`label_key`。

### `run_generate_reasoning`

**流程位置**：数据增强 operation，要求源数据版本已存在。

**作用**：对已有数据版本生成 reasoning 增强结果。

**关键参数**：详见 [复杂参数详解 - Reasoning 生成参数](./complex-params.md#reasoning-生成参数run_generate_reasoning-顶层参数)。

核心必填参数：`workflow_id`、`source_dataset_version`、`target_dataset_version`、`teacher_model`、`endpoint_url`。

### `run_build_structured_target`

**流程位置**：数据格式化 operation，要求源数据版本已存在。

**作用**：把 reasoning 数据继续转换为结构化目标，供下游训练使用。

**关键参数**：详见 [复杂参数详解 - 结构化目标构建参数](./complex-params.md#结构化目标构建参数run_build_structured_target-顶层参数)。

核心必填参数：`workflow_id`、`source_dataset_version`、`target_dataset_version`。

### `run_train_model`

**流程位置**：训练 operation，要求训练数据版本已存在。

**作用**：启动分类训练或继续训练。

**关键参数**：详见 [复杂参数详解 - 训练超参数](./complex-params.md#训练超参数run_train_model-顶层参数)。

- `lora`：LoRA 配置载荷，详见 [复杂参数详解 - lora](./complex-params.md#lora---lora-配置)。
- `run_id`：本次训练要写入的模型版本标识。
- 后端约束：BERT 需 `model_name`，LlamaFactory 需 `model_key`；`instruction` 仅 LlamaFactory 可用。

### `run_evaluate_model`

**流程位置**：评测 operation，要求待评测 `run_id` 和评测数据版本都已存在。

**作用**：对训练产物执行独立评测，输出指标与明细。

**关键参数**：详见 [复杂参数详解 - 评测参数](./complex-params.md#评测参数run_evaluate_model-顶层参数)。

- `prompt_engine`、`enable_thinking`、`max_new_tokens` 仅 LlamaFactory 后端可用。
- `enable_thinking` 仅在 `prompt_engine = "native"` 时有效。
- `run_id`：本次要评测的模型版本标识。

### `get_workflow_state`

**流程位置**：状态查询，随时可用。

**作用**：查询 workflow 当前主状态、operation 运行记录和最近事件。

**关键参数**：

- `workflow_id`：workflow ID。
- `event_limit`：返回最近多少条事件。

### `tail_stage_log`

**流程位置**：日志查询，随时可用。

**作用**：读取某个 operation 日志尾部，便于查看最新执行输出。

**关键参数**：

- `stage_run_id`：环节运行记录 ID。
- `max_bytes`：读取日志尾部的最大字节数。
