# BERT 工作流说明

本文档描述当前 TuneBench 中已经落地的 BERT 路径，帮助开发者快速理解现有闭环是如何组织的。

## 工作流概览

当前 BERT 路径围绕四个动作展开：

1. 数据准备
2. 模型训练
3. 训练期验证
4. 独立评测

这四段流程已经能够串成一条完整实验闭环。

## 1. 数据准备

入口：`prepare-data`

当前数据准备环节已经支持：

- 从 Excel 读取数据。
- 选择文本列和标签列。
- 仅保留指定标签。
- 生成训练集、验证集或测试集。
- 将结果标准化为统一记录结构。
- 为数据版本写入元数据。

当前标准记录格式非常简单：

- `text`
- `label`

这使得后续训练与评测层都不需要依赖源数据的原始列结构。

## 2. 模型训练

入口：`train`

当前训练流程已经支持：

- 基于 BERT 类模型进行 LoRA 微调。
- 自动读取训练集和可选验证集。
- 自动记录训练 run 的元数据。
- 导出 LoRA 权重和合并模型。
- 支持基于既有 LoRA 继续训练。
- 对标签空间和 continue-train 参数做一致性检查。

当前训练产物包括：

- 训练元数据
- checkpoints
- LoRA 导出
- merged 模型
- 训练指标 CSV
- 验证指标图与 loss 图

## 3. 训练期验证

如果数据版本下存在 validation split，训练会自动启用 validation。

当前验证环节输出分成两类：

### 聚合指标

写入 `train_metrics.csv`，区分：

- `train`
- `epoch_evaluate`
- `final_evaluate`

其中：

- `epoch_evaluate` 表示训练过程中的周期性验证。
- `final_evaluate` 表示训练结束后单独执行的最终 summary。

### 按标签指标

写入 `validation_label_metrics.csv`。

这张表的目标不是替代聚合指标，而是补充每个标签的 precision、recall、f1 和 support，方便分析哪些标签难学、哪些标签的数据可能存在问题。

## 4. 独立评测

入口：`evaluate`

独立评测与训练期 validation 的职责不同：

- 训练期 validation 主要用于观测训练过程和辅助模型选择。
- 独立 evaluate 主要用于训练完成后的固定验收。

当前独立评测会输出三类结果：

### 测试集整体指标

写入 `test_metrics.csv`。

### 测试集按标签指标

写入 `test_label_metrics.csv`。

### 测试集样本级预测明细

写入 `test_predictions.csv`，用于查看：

- 真实标签
- 预测标签
- 是否预测正确
- 预测置信度

## 图表与报表

当前训练完成后会自动生成两类图：

- `train_loss_plot.png`
- `train_eval_metrics_plot.png`

独立评测默认会生成 `eval_report.xlsx`，用于汇总关键 CSV 结果。

## 当前 BERT 路径的价值

当前这条路径的意义不只是“能跑通 BERT”，更重要的是它已经沉淀出一批真实有效的工程规则：

- 数据版本和模型版本分离。
- 训练期 validation 与独立 evaluate 明确区分。
- 聚合指标、按标签指标和样本级结果并存。
- 所有关键产物都写回统一资产目录。

这些规则会成为后续扩展到 Qwen 和其他模型路径时的参考基线。
