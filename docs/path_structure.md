# 资产路径结构说明

本文档只描述 TuneBench 的数据资产、模型资产、评测产物和 metadata 布局，不承担代码模块设计说明。路径设计的核心目标是：让每次实验都能按数据版本和模型版本独立留痕，避免覆盖，也便于在多个训练后端之间保持一致约束。

## 总体结构

当前默认资产根目录为项目根目录下的 `assets/`，整体结构如下：

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
│                   ├── llamafactory/  # 仅 llamafactory 后端会生成
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

路径模式：`assets/data/classification/<task_name>/<dataset_version>/`

目录语义：

- `raw/`：原始导入数据的留档。
- `stage/`：中间清洗结果和中间转换产物。
- `final/`：训练与评测实际消费的标准化数据。
- `metadata.json`：当前数据版本的结构化元数据。

分类任务中，`final/` 通常包含以下文件：

- `train.json` 或 `train.jsonl`
- `validation.json` 或 `validation.jsonl`
- `test.json` 或 `test.jsonl`

其中：

- `train` 与 `validation` 用于训练期。
- `test` 用于训练后的独立评测。

对于 reasoning 数据生成与 structured target 构建，仍沿用同一数据版本目录约束，只是 `stage/` 与 `final/` 中保存的记录结构会随阶段变化。

## 模型资产目录

路径模式：`assets/models/<backend>/classification/<task_name>/<run_id>/`

目录语义：

- `initial_model/`：训练起点的记录信息。
- `lora/`：LoRA 适配器及相关配置。
- `checkpoints/`：训练过程中的阶段性产物。
- `merged/`：合并导出的完整模型。
- `eval/`：训练指标、验证指标、测试结果、图表和报表。
- `llamafactory/`：LlamaFactory 运行时文件目录，仅 `llamafactory` 后端生成。
- `metadata.json`：本次训练 run 的结构化摘要。

其中 `llamafactory/` 当前会写入：

- `dataset/`：转写后的 Alpaca SFT 数据与 `dataset_info.json`
- `train.yaml`：训练配置
- `export.yaml`：导出配置
- `commands.sh`：等价命令脚本，便于排查与复现

## workflow 资产目录

路径模式：`assets/workflows/`

目录语义：

- `state/workflow_state.sqlite3`：workflow 主状态、环节运行记录与事件流。
- `<workflow_id>/stage_runs/<stage_run_id>/request.json`：单次环节启动请求快照。
- `<workflow_id>/stage_runs/<stage_run_id>/result.json`：单次环节执行结果快照。
- `<workflow_id>/stage_runs/<stage_run_id>/output.log`：worker 子进程输出日志。

这部分目录不承载训练模型产物，而是承载编排控制信息，供 MCP 查询、外部调度和运行留痕使用。

## metadata 约束

### 数据 metadata

数据版本目录下的 `metadata.json` 用于记录数据来源、清洗参数、split 信息和环节摘要。

### 训练 run metadata

模型 run 目录下的 `metadata.json` 当前已经围绕统一 schema 演进，schema 定义位于 `src/tunebench/artifacts/run_metadata.py`。核心字段包括：

- `backend`、`task_name`、`dataset_version`
- `model_name`、`model_key`、`reasoning_mode`
- `run_id`、`output_dir`
- `train_file`、`validation_file`
- `num_labels`、`label_names`、`label_to_id`
- `dataset_stats`
- `hyperparameters`
- `backend_config`
- `train_metrics`、`eval_metrics`
- `instruction`
- `status`

其中：

- `backend_config` 用于保存后端专有配置，例如 template、reasoning policy、loader family、运行时配置路径等。
- `status` 用于表达训练生命周期状态，训练准备、训练完成、导出完成或失败都会写回对应状态。

## 评测产物目录

`eval/` 目录当前包含以下几类产物。

### 训练过程相关

- `train_metrics.csv`
  记录训练 loss、训练期 validation 聚合指标，以及最终 summary。

- `validation_label_metrics.csv`
  记录训练期 validation 的按标签指标。一次 `epoch_evaluate` 会追加一批标签行；训练结束后的 `final_evaluate` 也会以 summary 形式追加。

- `train_loss_plot.png`
  基于训练指标表生成的 loss 图，当前会同时绘制训练 loss 和 validation loss。

- `train_eval_metrics_plot.png`
  基于训练期 validation 聚合指标生成的指标趋势图。

### 独立评测相关

- `test_metrics.csv`
  测试集整体指标摘要。

- `test_label_metrics.csv`
  测试集按标签指标摘要，用于分析各类别表现差异。

- `test_predictions.csv`
  测试集逐样本预测明细。BERT 路径主要包含真实标签、预测标签、是否预测正确和置信度等字段；LlamaFactory 路径还会额外记录 `raw_output`、`cleaned_output`、`finish_reason`、`prompt_token_count`、`generated_token_count` 等生成侧字段。

- `eval_report.xlsx`
  将主要 CSV 结果汇总到一个 Excel 文件中，便于人工查看和共享。

## 当前路径管理实现

统一路径入口位于 `src/tunebench/artifacts/path.py`。

当前关键对象：

- `DatasetPathManager`：管理数据资产目录。
- `ModelPathManager`：管理模型资产目录。
- `DatasetArtifactLayout`：描述某个数据版本的完整路径布局。
- `ModelArtifactLayout`：描述某个训练 run 的完整路径布局。

当前路径管理已经不再为某个单一模型写死目录，而是通过 `backend/task/run_id` 三层结构统一约束 BERT 与 LlamaFactory 两条路径。

## 设计意图

当前这套路径组织解决的是三个问题：

1. 数据版本和模型版本需要严格分离。
2. 每一次训练和评测都必须有独立产物目录。
3. 训练中间结果、最终模型和评测结果要能被统一追踪。

这也是后续继续扩展更多模型键、更多后端能力时，优先复用的基础约束之一。
