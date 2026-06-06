# MCP 工具接口说明

本文档说明每个 MCP 工具的作用、在整体流程中的位置，以及关键参数含义。

## 整体调用流程

```
外部 Agent → MCP 工具 → Workflow 引擎 → 底层训练/评测模块
```

你调用 MCP 工具后，请求会被转换为 workflow 操作，由 workflow 引擎负责状态管理、子进程执行和审核推进。具体的训练、评测、数据处理由底层模块执行。

## 工具清单

### `preview_workflow`

**流程位置**：创建 workflow 前的预览步骤。

**作用**：在真正创建 workflow 前，预览环节拓扑、运行标识和审核断点设置。

**关键参数**：

- `task_name`：任务名。
- `backend`：后端类型，通常为 `bert` 或 `llamafactory`。
- `runtime`：运行时配置载荷，用于表达环境或执行上下文。
- `enabled_stages`：本次启用的环节列表。
- `review_required_stages`：执行成功后需要人工审核的环节列表。

### `create_workflow`

**流程位置**：workflow 创建入口。

**作用**：创建 workflow 主记录，作为整条实验链路的控制平面入口。

**关键参数**：

- 与 `preview_workflow` 基本一致。
- `run_id`：可选的运行标识；未显式提供时可由服务层生成。

### `run_prepare_dataset`

**流程位置**：第一个数据环节。

**作用**：触发数据准备环节，生成训练集、验证集或测试集版本。

**关键参数**：

- `workflow_id`：所属 workflow。
- `input_path`：输入表格路径。
- `dataset_version`：目标数据版本号。
- `text_key`、`label_key`：文本列与标签列字段名。
- `validation_ratio`：验证集切分比例。
- `is_test`：是否生成测试数据。

### `run_generate_reasoning`

**流程位置**：数据增强环节，依赖 `prepare_dataset` 完成。

**作用**：对已有数据版本生成 reasoning 增强结果。

**关键参数**：

- `source_dataset_version`：源数据版本。
- `target_dataset_version`：目标增强版本。
- `teacher_model`、`endpoint_url`：教师模型及其服务地址。
- `splits`：处理哪些数据切分，默认是训练集和验证集。

### `run_build_structured_target`

**流程位置**：数据格式化环节，依赖 `generate_reasoning` 完成。

**作用**：把 reasoning 数据继续转换为结构化目标，供下游训练使用。

**关键参数**：

- `source_dataset_version`：源 reasoning 数据版本。
- `target_dataset_version`：目标结构化版本。
- `confidence`：结构化目标的置信阈值。

### `run_train_model`

**流程位置**：训练环节，依赖数据准备完成。

**作用**：启动分类训练或继续训练。

**关键参数**：

- `dataset_version`：训练所用数据版本。
- `model_name` / `model_key`：模型路径或注册键。
- `resume_lora`：继续训练时的已有 LoRA 路径。
- `instruction`、`reasoning_mode`：LlamaFactory 路径下的重要训练语义参数。
- `lora`：LoRA 配置载荷。

### `run_evaluate_model`

**流程位置**：评测环节，依赖 `train_model` 完成。

**作用**：对训练产物执行独立评测，输出指标与明细。

**关键参数**：

- `dataset_version`：评测集版本。
- `artifact_type`：评测哪类产物，如 `merged`。
- `prompt_engine`、`enable_thinking`：LlamaFactory / Qwen 评测时的重要渲染控制参数。

### `approve_stage`

**流程位置**：审核断点操作。

**作用**：人工审核通过指定环节，使 workflow 可以继续推进。

**关键参数**：

- `stage_run_id`：被审核的环节运行记录 ID。

### `reject_stage`

**流程位置**：审核断点操作。

**作用**：人工审核拒绝指定环节，并给出拒绝原因。

**关键参数**：

- `stage_run_id`：被拒绝的环节运行记录 ID。
- `reason`：拒绝说明。

### `get_workflow_state`

**流程位置**：状态查询，随时可用。

**作用**：查询 workflow 当前主状态、环节状态和最近事件。

**关键参数**：

- `workflow_id`：workflow ID。
- `event_limit`：返回最近多少条事件。

### `tail_stage_log`

**流程位置**：日志查询，随时可用。

**作用**：读取某个环节日志尾部，便于查看最新执行输出。

**关键参数**：

- `stage_run_id`：环节运行记录 ID。
- `max_bytes`：读取日志尾部的最大字节数。
