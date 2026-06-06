# 项目边界

本文档说明 TuneBench MCP 当前支持的能力范围和限制，帮助外部 agent 判断需求是否在可用范围内。

## 项目定位

TuneBench 是面向中文分类微调的工程，不是通用生成任务平台。

当前稳定范围包括：

- 分类数据准备。
- reasoning 数据增强。
- structured target 构建。
- 分类训练与继续训练。
- 独立评测。
- 单条分类推理或单轮对话。
- workflow 编排与 MCP 协议接入。

## 后端边界

### BERT

- 适用于分类微调主路径。
- 支持训练、继续训练、训练期验证、独立评测、单条分类推理。
- 不支持开放对话能力。

### LlamaFactory

- 适用于分类训练、导出、独立评测和单轮对话。
- 当前已校验模型键包括：`qwen3_4b`、`qwen3_32b`、`qwen3_5_4b`。
- 外部模型对话能力主要落在这一后端。

## 对话边界

- 对话能力不等同于通用智能体框架。
- 已训练模型对话与外部模型对话的参数约束不同。
- 外部模型对话当前要求走 `llamafactory` 后端。
- Qwen3 与 Qwen3.5 在 no_think 语义上仍需区别对待。

## Workflow 边界

Workflow 负责跨环节编排，不替代单次命令执行。

当前统一支持的环节包括：

- `prepare_dataset`
- `generate_reasoning`
- `build_structured_target`
- `train_model`
- `evaluate_model`

Workflow 负责：

- 状态持久化。
- 子进程启动。
- 日志留痕。
- 人工审核断点。

Workflow 不负责：

- 在控制进程内直接执行训练模块。
- 重新实现现有训练与评测逻辑。

## MCP 边界

- MCP 是协议接入层，对外暴露 workflow 能力。
- MCP 当前服务的是 workflow 编排，不覆盖所有底层命令。
- 所有训练、评测、数据处理逻辑由底层核心模块执行，MCP 层只做请求转发和状态管理。
