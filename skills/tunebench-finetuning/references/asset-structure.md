# 资产目录结构

本文档描述 TuneBench 产生的所有文件在磁盘上的组织方式。理解这些结构有助于你将 MCP 工具参数（如 `task_name`、`run_id`、`dataset_version`）映射到具体的文件路径。

## 核心概念映射

| 参数 | 含义 | 出现位置 |
|------|------|----------|
| `task_name` | 任务名称，标识一次完整的微调实验主题 | 数据目录、模型目录 |
| `dataset_version` | 数据版本号，同一任务可有多个数据版本 | 数据目录 |
| `run_id` | 训练运行版本号，同一任务可有多个训练 run | 模型目录 |
| `backend` | 训练后端类型（`bert`/`llamafactory`） | 模型目录 |
| `workflow_id` | Workflow 唯一标识 | Workflow 目录 |
| `stage_run_id` | 单次环节运行记录 ID | Workflow 目录 |

## 总体结构

默认资产根目录为 `assets/`：

```text
assets/
├── data/
│   └── classification/
│       └── <task_name>/
│           └── <dataset_version>/
│               ├── raw/
│               ├── stage/
│               ├── final/
│               └── metadata.json
├── models/
│   └── <backend>/
│       └── classification/
│           └── <task_name>/
│               └── <run_id>/
│                   ├── initial_model/
│                   ├── lora/
│                   ├── checkpoints/
│                   ├── merged/
│                   ├── eval/
│                   ├── llamafactory/  # 仅 llamafactory 后端
│                   └── metadata.json
└── workflows/
    ├── state/
    │   └── workflow_state.sqlite3
    └── <workflow_id>/
        └── stage_runs/
            └── <stage_run_id>/
                ├── request.json
                ├── result.json
                └── output.log
```

## 数据资产目录

**路径模式**：`assets/data/classification/<task_name>/<dataset_version>/`

`task_name` 是任务的逻辑分组，同一任务下可以有多个 `dataset_version`，代表数据的不同迭代版本。

| 目录/文件 | 说明 |
|-----------|------|
| `raw/` | 原始导入数据的留档，不做修改 |
| `stage/` | 中间清洗、增强、转换的产物 |
| `final/` | 训练与评测实际消费的标准化数据 |
| `metadata.json` | 当前数据版本的结构化元数据 |

`final/` 中通常包含：

- `train.json`（或 `train.jsonl`）— 训练集
- `validation.json`（或 `validation.jsonl`）— 验证集
- `test.json`（或 `test.jsonl`）— 测试集

数据在不同环节之间流转时，会经历版本演进：

```
prepare_dataset：  原始表格 → v001（raw → final）
generate_reasoning：v001 → v002（在 stage/final 中增加推理链字段）
build_structured_target：v002 → v003（转换为结构化训练格式）
```

## 模型资产目录

**路径模式**：`assets/models/<backend>/classification/<task_name>/<run_id>/`

`run_id` 标识一次独立的训练运行，同一任务下可以有多个 run，分别对应不同的超参数、数据版本或模型起点。

| 目录/文件 | 说明 |
|-----------|------|
| `initial_model/` | 训练起点的记录信息 |
| `lora/` | LoRA 适配器权重及相关配置 |
| `checkpoints/` | 训练过程中的阶段性 checkpoint |
| `merged/` | 合并导出的完整模型 |
| `eval/` | 训练指标、验证指标、测试结果、图表和报表 |
| `llamafactory/` | LlamaFactory 运行时文件（仅 `llamafactory` 后端） |
| `metadata.json` | 本次训练 run 的结构化摘要 |

`llamafactory/` 子目录内容：

- `dataset/`：转写后的 Alpaca SFT 数据与 `dataset_info.json`
- `train.yaml`：训练配置
- `export.yaml`：导出配置
- `commands.sh`：等价命令脚本，便于排查与复现

## Workflow 资产目录

**路径模式**：`assets/workflows/`

Workflow 目录承载编排控制信息，不承载模型产物。

| 路径 | 说明 |
|------|------|
| `state/workflow_state.sqlite3` | workflow 主状态、环节运行记录与事件流 |
| `<workflow_id>/stage_runs/<stage_run_id>/request.json` | 单次环节启动请求快照 |
| `<workflow_id>/stage_runs/<stage_run_id>/result.json` | 单次环节执行结果快照 |
| `<workflow_id>/stage_runs/<stage_run_id>/output.log` | worker 子进程输出日志 |

## 参数与路径的对应关系

当你调用 MCP 工具时，传入的参数会直接映射到文件系统路径：

- `task_name` + `dataset_version` → `assets/data/classification/{task_name}/{dataset_version}/`
- `backend` + `task_name` + `run_id` → `assets/models/{backend}/classification/{task_name}/{run_id}/`
- `workflow_id` → `assets/workflows/{workflow_id}/`
- `stage_run_id` → `assets/workflows/{workflow_id}/stage_runs/{stage_run_id}/`

这意味着：

- 同一个 `task_name` 下的不同 `dataset_version` 互不干扰。
- 同一个 `task_name` 下的不同 `run_id` 互不干扰。
- `workflow_id` 独立于 `task_name`，但 workflow 中的环节会通过 `task_name` 和 `dataset_version` 关联到具体的数据或模型资产。
