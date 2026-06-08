---
name: tunebench-finetuning
description: 'Use when an external agent needs to use TuneBench MCP to orchestrate model fine-tuning workflows. Covers workflow containers, operation execution, resource versioning, and practical usage patterns.'
argument-hint: '要完成的微调任务或需要调用的 MCP 工具'
user-invocable: true
---

# TuneBench 微调 MCP 技能

## 你能用它做什么

TuneBench MCP 让外部 agent 能够编排和执行模型微调的完整流程。你可以通过一组标准化的 MCP 工具，完成从数据准备到模型评测的全链路操作，并把不同数据版本、训练版本和评测动作留在同一个 workflow 容器里。

核心能力包括：

- **数据准备**：将原始表格数据切分为训练集、验证集、测试集。
- **Reasoning 增强**：调用教师模型为数据生成推理链。
- **结构化目标构建**：将增强后的数据转换为训练可用的结构化格式。
- **模型训练**：启动分类训练或基于已有产物的继续训练。
- **模型评测**：对训练产物执行独立评测，获取指标与明细。
- **流程编排**：通过 workflow 机制串联上述环节，保持状态持久化和执行留痕。

## 核心概念

### Workflow（工作流）

Workflow 是一个实验容器，而不是不可回退的 stage 链。创建 workflow 时，你需要指定：

- 任务名称
- 后端类型（`bert` 或 `llamafactory`）
- 本次允许使用的 operation 列表

Workflow 创建后，你可以按需要重复触发不同 operation，随时查询状态与日志。

### Operation（执行动作）

Workflow 中可以执行多个 operation，当前支持：

| 环节 | 说明 |
|------|------|
| `prepare_dataset` | 数据准备：输入原始表格，输出训练/验证/测试集版本 |
| `generate_reasoning` | Reasoning 增强：调用教师模型生成推理链 |
| `build_structured_target` | 结构化目标构建：将 reasoning 数据转为训练格式 |
| `train_model` | 模型训练：启动训练或继续训练 |
| `evaluate_model` | 模型评测：对训练产物执行评测 |

这些 operation 不再按固定顺序推进。是否允许执行，取决于：

- 输入资源是否存在
- 输出资源是否会覆盖已有产物
- 当前 workflow 是否已有运行中的 operation

例如，训练完成后仍然可以回到 `prepare_dataset` 生成新的测试数据版本。

### 后端类型

- **bert**：适用于分类微调主路径，支持训练、继续训练、评测和单条推理。
- **llamafactory**：适用于分类训练、导出、评测和单轮对话，支持 Qwen 系列模型。

## 使用流程

### 1. 预览并创建 Workflow

```
preview_workflow → 确认 operation 范围和运行时配置 → create_workflow
```

先用 `preview_workflow` 预览即将创建的 workflow 结构，确认无误后调用 `create_workflow` 正式创建。创建时不再绑定固定 `run_id`。

**GPU 选择**：在创建 workflow 之前，务必向用户确认使用哪张 GPU 卡。当前环境可用显卡为 **4、5、6、7**（四选一）。确认后，将选中的显卡编号写入 `runtime.visible_devices`，例如 `"visible_devices": ["4"]`。如果用户没有明确指定，默认使用 `"4"`。

### 2. 按需执行 operation

```
run_prepare_dataset → run_generate_reasoning → run_build_structured_target → run_train_model → run_evaluate_model
```

根据当前实验需要调用对应的 `run_*` 工具触发执行。常见组合包括：

- `run_prepare_dataset` → `run_train_model`
- `run_prepare_dataset` → `run_generate_reasoning` → `run_build_structured_target` → `run_train_model`
- `run_train_model` 之后再补做一次 `run_prepare_dataset`，然后执行 `run_evaluate_model`

`run_train_model` 和 `run_evaluate_model` 必须显式提供 `run_id`。

### 3. 监控与查询

```
get_workflow_state → tail_stage_log
```

- 用 `get_workflow_state` 随时查询 workflow 整体状态和最近事件。
- 用 `tail_stage_log` 查看某个 operation 的最新日志输出。

### 版本命名建议

以下命名建议用于帮助 agent 生成更可读、更稳定的资源版本，但当前不会被程序强制校验：

- `dataset_version` 建议体现主版本与用途，例如 `v0013_train_raw`、`v0013_reasoning`、`v0013_test`。
- `run_id` 建议体现训练世代或派生关系，例如 `r0013_base`、`r0013_ft01`、`r0014_continue01`。
- 同一个 workflow 内推荐持续使用有语义的版本命名，避免 `test1`、`tmp2` 这类后续难以追踪的标识。

## 注意事项

- workflow 不再强制线性顺序，但 operation 仍然受输入存在性与输出覆盖风险约束。
- 每个 `run_*` 工具调用后会异步执行，执行结果通过 `get_workflow_state` 或 `tail_stage_log` 查询。
- `run_train_model` 与 `run_evaluate_model` 的 `run_id` 属于本次 operation，而不属于 workflow 本体。
- 当前主要支持中文分类微调任务，其他任务类型的支持范围以实际接口返回为准。

## 详细参考

- [MCP 工具接口说明](./references/mcp-interfaces.md) — 每个 MCP 工具的流程位置和关键参数概览
- [复杂参数详解](./references/complex-params.md) — `runtime`、`lora` 等嵌套配置对象的完整字段、默认值、约束与后端差异
- [资产目录结构](./references/asset-structure.md) — `task_name`、`run_id`、`dataset_version` 等参数在文件系统上的组织方式
- [项目边界](./references/project-boundaries.md) — 当前支持的能力范围和限制