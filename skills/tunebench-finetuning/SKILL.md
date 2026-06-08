---
name: tunebench-finetuning
description: 'Use when an external agent needs to use TuneBench MCP to orchestrate model fine-tuning workflows. Covers workflow concepts, available MCP tools, stage progression, review gates, and practical usage patterns.'
argument-hint: '要完成的微调任务或需要调用的 MCP 工具'
user-invocable: true
---

# TuneBench 微调 MCP 技能

## 你能用它做什么

TuneBench MCP 让外部 agent 能够编排和执行模型微调的完整流程。你可以通过一组标准化的 MCP 工具，完成从数据准备到模型评测的全链路操作，并在关键环节设置人工审核断点。

核心能力包括：

- **数据准备**：将原始表格数据切分为训练集、验证集、测试集。
- **Reasoning 增强**：调用教师模型为数据生成推理链。
- **结构化目标构建**：将增强后的数据转换为训练可用的结构化格式。
- **模型训练**：启动分类训练或基于已有产物的继续训练。
- **模型评测**：对训练产物执行独立评测，获取指标与明细。
- **流程编排**：通过 workflow 机制串联上述环节，保持状态持久化和执行留痕。
- **人工审核**：在指定环节设置审核断点，由人工决定是否继续推进。

## 核心概念

### Workflow（工作流）

Workflow 是一条完整的微调实验链路。创建 workflow 时，你需要指定：

- 任务名称
- 后端类型（`bert` 或 `llamafactory`）
- 本次启用的环节列表
- 哪些环节需要人工审核

Workflow 创建后，你可以逐个触发环节执行，随时查询状态，并在审核断点处进行人工决策。

### 环节（Stage）

Workflow 由多个环节组成，当前支持的环节包括：

| 环节 | 说明 |
|------|------|
| `prepare_dataset` | 数据准备：输入原始表格，输出训练/验证/测试集版本 |
| `generate_reasoning` | Reasoning 增强：调用教师模型生成推理链 |
| `build_structured_target` | 结构化目标构建：将 reasoning 数据转为训练格式 |
| `train_model` | 模型训练：启动训练或继续训练 |
| `evaluate_model` | 模型评测：对训练产物执行评测 |

环节按顺序推进，每个环节执行完成后会记录状态和日志。

### 审核断点（Review Gate）

创建 workflow 时，可以指定某些环节执行成功后需要人工审核。到达审核断点时：

- 你可以查看该环节的输出日志
- 审核通过（approve）后，workflow 继续推进
- 审核拒绝（reject）后，workflow 在该环节终止，并记录拒绝原因

### 后端类型

- **bert**：适用于分类微调主路径，支持训练、继续训练、评测和单条推理。
- **llamafactory**：适用于分类训练、导出、评测和单轮对话，支持 Qwen 系列模型。

## 使用流程

### 1. 预览并创建 Workflow

```
preview_workflow → 确认环节拓扑和审核设置 → create_workflow
```

先用 `preview_workflow` 预览即将创建的 workflow 结构，确认无误后调用 `create_workflow` 正式创建。

**GPU 选择**：在创建 workflow 之前，务必向用户确认使用哪张 GPU 卡。当前环境可用显卡为 **4、5、6、7**（四选一）。确认后，将选中的显卡编号写入 `runtime.visible_devices`，例如 `"visible_devices": ["4"]`。如果用户没有明确指定，默认使用 `"4"`。

### 2. 按顺序执行环节

```
run_prepare_dataset → run_generate_reasoning → run_build_structured_target → run_train_model → run_evaluate_model
```

根据创建时指定的环节列表，依次调用对应的 `run_*` 工具触发执行。每个环节执行完成后，状态会自动更新。

### 3. 监控与审核

```
get_workflow_state → tail_stage_log → approve_stage / reject_stage
```

- 用 `get_workflow_state` 随时查询 workflow 整体状态和最近事件。
- 用 `tail_stage_log` 查看某个环节的最新日志输出。
- 当环节到达审核断点时，用 `approve_stage` 或 `reject_stage` 进行决策。

## 注意事项

- 环节之间有依赖关系，不要跳过前置环节直接执行后续环节。
- 每个 `run_*` 工具调用后会异步执行，执行结果通过 `get_workflow_state` 或 `tail_stage_log` 查询。
- 审核断点是可选的，创建 workflow 时未指定则所有环节自动推进。
- 当前主要支持中文分类微调任务，其他任务类型的支持范围以实际接口返回为准。

## 详细参考

- [MCP 工具接口说明](./references/mcp-interfaces.md) — 每个 MCP 工具的流程位置和关键参数概览
- [复杂参数详解](./references/complex-params.md) — `runtime`、`lora` 等嵌套配置对象的完整字段、默认值、约束与后端差异
- [资产目录结构](./references/asset-structure.md) — `task_name`、`run_id`、`dataset_version` 等参数在文件系统上的组织方式
- [项目边界](./references/project-boundaries.md) — 当前支持的能力范围和限制