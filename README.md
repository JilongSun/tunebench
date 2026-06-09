# TuneBench

TuneBench 是一个面向中文分类微调场景的命令行工程，用于把数据准备、模型训练、结果评估和产物留痕组织成一套可复用流程。

项目同时支持 BERT 与 LlamaFactory 两类后端，统一管理数据版本、训练 run、评测结果和运行元数据，适合需要持续迭代分类任务的数据与模型资产场景。

## 主要能力

- 数据准备：从 Excel 读取原始表格，清洗并生成训练、验证、测试数据资产。
- 推理增强：支持 reasoning 数据生成与 structured target 构建。
- 模型训练：支持 BERT LoRA 微调、继续训练，以及 LlamaFactory 分类训练与导出。
- 独立评估：支持训练完成后的独立评测、汇总指标和样本级预测明细导出。
- 单轮推理：支持已训练 run 推理，也支持外部本地 GPT 类模型对话。
- 资产管理：统一管理数据版本、模型产物、训练元数据、图表、报表和评测输出。
- MCP 接入：通过 MCP server 暴露 workflow 创建、operation 执行、状态查询和日志读取能力。

## 项目架构

TuneBench 按核心流程层与外部接入层组织：

- `src/tunebench/cli.py`：命令行入口、结果输出与命令分发。
- `src/tunebench/cli_support.py`：命令参数校验与契约对象组装。
- `src/tunebench/contracts/`：数据准备、训练、评测、对话等统一契约定义。
- `src/tunebench/classification/`：分类任务共享的数据处理、reasoning 生成、structured target 构建与训练期评估能力。
- `src/tunebench/backends/`：BERT 与 LlamaFactory 后端实现，以及统一注册表。
- `src/tunebench/artifacts/`：数据版本、模型 run、评测产物与 metadata 路径管理。
- `src/tunebench/util/`：日志与通用辅助能力。
- `src/tunebench_mcp/`：MCP server 与 workflow 工具映射入口。

更完整的模块说明见 `docs/developer_overview.md`。

## 部署与使用

### 环境要求

- Conda（用于创建 Python 3.11.15 环境）
- Poetry
- 可用的 CUDA 训练环境
- 若使用 LlamaFactory 路径，需要可用的 `llamafactory-cli`

### 部署

拉取仓库代码后，在项目根目录执行：

```bash
make build
```

该命令会自动在项目根目录创建 `.tb311` conda 环境（Python 3.11.15），并通过 Poetry 安装所有依赖。

### 使用方式

部署完成后，TuneBench 提供两种使用方式。

#### 方式一：CLI 命令行

直接使用 `tunebench` 或 `tb` 命令执行数据准备、训练、评估和推理：

```bash
# 查看可用命令
tb --help

# 查看具体命令参数
tb train --help
tb evaluate --help
```

日常使用流程：

1. 执行 `prepare-data` 生成训练、验证或测试数据。
2. 按任务需要执行 `generate-reasoning` 与 `build-structured-target`。
3. 执行 `train` 启动训练或继续训练。
4. 执行 `evaluate` 生成独立评测结果。
5. 需要单条推理时，执行 `chat`。

可用命令包括：

- `prepare-data`
- `generate-reasoning`
- `build-structured-target`
- `train`
- `evaluate`
- `chat`
- `plan`

#### 方式二：MCP 服务

TuneBench 提供 streamable HTTP 方式的 MCP 服务，供外部 agent 调用 workflow 创建、operation 执行、状态查询和日志读取能力。

| 操作 | 命令 |
|------|------|
| 启动 | `make run` |
| 停止 | `make stop` |
| 状态 | `make status` |

- 默认地址：`http://127.0.0.1:8888/mcp`
- 日志文件：`runtime/mcp/tunebench_mcp.log`
- PID 文件：`runtime/mcp/tunebench_mcp.pid`
- 外部 agent 配置：`configs/mcp/tunebench.http.json`

#### 调试模式

设置环境变量 `TUNEBENCH_MCP_DEBUG=true` 启动 MCP 服务，可在 `http://127.0.0.1:8888/inspector` 访问 MCP Inspector 页面，用于实时监控和调试 MCP 工具调用。

```bash
# 方式一：通过环境变量
TUNEBENCH_MCP_DEBUG=true make run

# 方式二：使用调试启动脚本（自动开启 debug 模式）
# 在 VS Code 中对 src/tunebench_mcp/debug_server.py 按 F5 调试运行
```

进入 Inspector 页面后，将 MCP URL 改为 `http://127.0.0.1:8888/inspector` 即可开始监控。

如需调整监听地址、端口或路径，可在执行 `make run` 前设置以下环境变量：

- `TUNEBENCH_MCP_HOST`
- `TUNEBENCH_MCP_PORT`
- `TUNEBENCH_MCP_PATH`

## 后端说明

### bert

- 支持训练、继续训练、训练期验证、独立评测与单条分类推理。
- 训练进程要求仅暴露一张 GPU。

### llamafactory

- 支持分类训练、导出、独立评测和单轮对话。
- 训练路径会生成运行时数据集、`dataset_info.json`、`train.yaml`、`export.yaml` 与命令脚本。

已校验的模型键包括：

- `qwen3_4b`
- `qwen3_32b`
- `qwen3_5_4b`

## 产物结构

TuneBench 默认将产物写入 `assets/`：

- `assets/data/classification/`：数据版本资产。
- `assets/models/`：训练 run、LoRA、merged 模型、评测结果与 metadata。
- `assets/workflows/`：workflow 状态、运行 request/result 文件与日志。

每次数据处理、训练和评测都会生成独立目录，用于记录输入来源、参数、指标与结果文件。

## 文档

- `docs/developer_overview.md`：开发者架构总览与模块边界说明。
- `docs/bert_workflow.md`：BERT 工作流说明。
- `docs/path_structure.md`：资产目录与 metadata 布局说明。
- `docs/workflow_architecture.md`：workflow 状态层、子进程执行模型与存储结构说明。
- `docs/roadmap.md`：能力规划说明。
