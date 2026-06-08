# 复杂参数详解

本文档详细说明 MCP 工具中那些以 JSON 对象（`dict`）形式传递的复杂配置参数，包括每个字段的含义、类型、默认值、约束和适用场景。

---

## `runtime` — 运行时配置

**使用位置**：`preview_workflow`、`create_workflow`

**作用**：控制 workflow 子进程的运行时环境，包括 GPU 可见设备、环境变量覆盖和工作目录。

**JSON 结构**：

```json
{
  "visible_devices": ["4"],
  "env_overrides": {"HF_HOME": "/data/hf"},
  "working_dir": "/tmp"
}
```

### 字段说明

| 字段 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `visible_devices` | `string[]` | `[]` | 否 | GPU 设备索引列表，写入 `CUDA_VISIBLE_DEVICES` 环境变量。当前环境仅支持 `["4"]`、`["5"]`、`["6"]`、`["7"]` 四选一（单卡）。空数组表示不限制 |
| `env_overrides` | `object<string, string>` | `{}` | 否 | 环境变量覆盖字典，会 merge 到子进程环境中。键值均为字符串 |
| `working_dir` | `string` | `null` | 否 | 子进程工作目录；不指定时由系统决定 |

### 注意事项

- `visible_devices` 中的元素会被转换为字符串，即使传入数字也会被序列化。
- `env_overrides` 中的键值对会覆盖 `build_env()` 方法自动设置的环境变量（包括 `PATH` 和 `CUDA_VISIBLE_DEVICES`）。
- 整个 `runtime` 参数在 MCP 工具层面是可选的；不传时等价于空配置（无 GPU 限制，无环境变量覆盖）。

---

## `lora` — LoRA 配置

**使用位置**：`run_train_model`

**作用**：配置 LoRA（Low-Rank Adaptation）微调的参数。

**JSON 结构**：

```json
{
  "r": 16,
  "alpha": 32,
  "dropout": 0.05,
  "target_modules": ["q_proj", "v_proj"],
  "bias": "none",
  "modules_to_save": [],
  "use_rslora": false,
  "use_dora": false
}
```

### 字段说明

| 字段 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `r` | `integer` | `8` | 否 | LoRA 秩（rank），控制低秩矩阵的维度。值越大表达能力越强，参数量也越多 |
| `alpha` | `integer` | `16` | 否 | LoRA alpha 缩放因子。实际缩放比例为 `alpha / r` |
| `dropout` | `float` | `0.1` | 否 | LoRA 层的 dropout 率，范围建议 `[0.0, 0.3]` |
| `target_modules` | `string[]` | `[]` | 否 | 应用 LoRA 的目标模块名列表。空数组表示由后端自动选择默认模块 |
| `bias` | `string` | `"none"` | 否 | Bias 处理方式。**枚举约束**：仅接受 `"none"`、`"all"`、`"lora_only"` 三值之一 |
| `modules_to_save` | `string[]` | `[]` | 否 | 需要完整保存（不应用 LoRA）的模块名列表，通常用于分类头等需要全量训练的模块 |
| `use_rslora` | `boolean` | `false` | 否 | 是否启用 Rank-Stabilized LoRA，可改善训练稳定性 |
| `use_dora` | `boolean` | `false` | 否 | 是否启用 DoRA（Weight-Decomposed LoRA），将权重分解为方向和幅度 |

### 注意事项

- 整个 `lora` 参数在 MCP 工具层面是可选的；不传时等价于使用全部默认值。
- `bias` 字段为枚举类型，传入非法值可能导致运行时错误。
- `target_modules` 的具体模块名取决于模型架构（BERT 和 Llama 的模块名不同）。
- `use_rslora` 和 `use_dora` 可以同时启用，具体效果取决于后端实现。

---

## 训练超参数（`run_train_model` 顶层参数）

虽然不属于嵌套对象，但 `run_train_model` 的超参数组合较多，在此一并说明。

### 字段说明

| 字段 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `workflow_id` | `string` | — | **是** | 所属 workflow |
| `dataset_version` | `string` | — | **是** | 训练所用数据版本 |
| `model_name` | `string` | `null` | 条件必填 | BERT 后端在 `start` 模式下必填；`resume_lora` 模式下可不填 |
| `model_key` | `string` | `null` | 条件必填 | LlamaFactory 后端必填（如 `"qwen3_4b"`） |
| `instruction` | `string` | `null` | 否 | **仅 LlamaFactory 后端可用**。分类任务的 instruction 模板，不能为空字符串 |
| `reasoning_mode` | `string` | `null` | 否 | **枚举约束**：`"think"`、`"no_think"` 或 `null`。控制推理链的生成模式 |
| `resume_lora` | `string` | `null` | 否 | 继续训练时的已有 LoRA 路径；指定后为继续训练模式 |
| `export_dir` | `string` | `null` | 否 | 导出目录路径 |
| `num_labels` | `integer` | `null` | 否 | 分类标签数；不指定时从数据自动推断 |
| `learning_rate` | `float` | `2e-5` | 否 | 学习率 |
| `batch_size` | `integer` | `8` | 否 | 批次大小 |
| `num_train_epochs` | `integer` | `3` | 否 | 训练轮数 |
| `max_sequence_length` | `integer` | `256` | 否 | 最大序列长度 |
| `warmup_ratio` | `float` | `0.0` | 否 | 学习率预热比例，范围 `[0.0, 1.0]` |
| `seed` | `integer` | `42` | 否 | 随机种子 |
| `lora` | `object` | `null` | 否 | LoRA 配置对象，详见上方 `lora` 章节 |

### 后端约束

| 约束 | 说明 |
|------|------|
| BERT 后端 + `start` 模式 | 必须提供 `model_name` |
| BERT 后端 | 不支持 `instruction` 参数，传入会报错 |
| LlamaFactory 后端 | 必须提供 `model_key` |
| LlamaFactory 后端 + `instruction` | `instruction` 不能为空字符串 |

---

## 评测参数（`run_evaluate_model` 顶层参数）

### 字段说明

| 字段 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `workflow_id` | `string` | — | **是** | 所属 workflow |
| `dataset_version` | `string` | — | **是** | 评测集版本 |
| `artifact_type` | `string` | `"merged"` | 否 | 评测对象类型，如 `"merged"`（合并模型） |
| `batch_size` | `integer` | `8` | 否 | 批次大小 |
| `max_sequence_length` | `integer` | `null` | 否 | 最大序列长度；BERT 后端默认 256，LlamaFactory 后端需显式指定 |
| `max_new_tokens` | `integer` | `null` | 否 | **仅 LlamaFactory 后端可用**。生成的最大 token 数 |
| `prompt_engine` | `string` | `null` | 否 | **仅 LlamaFactory 后端可用**。**枚举约束**：`"llamafactory"`、`"native"` 或 `null` |
| `enable_thinking` | `boolean` | `null` | 否 | **仅 LlamaFactory + `native` prompt_engine 可用**。启用思考模式 |
| `export_xlsx` | `boolean` | `true` | 否 | 是否导出 Excel 报表 |

### 后端约束

| 约束 | 说明 |
|------|------|
| `max_new_tokens` | 仅 LlamaFactory 后端可用，BERT 后端传入会报错 |
| `prompt_engine` | 仅 LlamaFactory 后端可用，BERT 后端传入会报错 |
| `enable_thinking` | 仅 LlamaFactory 后端可用，且仅在 `prompt_engine` 为 `"native"` 时有效 |
| `enable_thinking` + `prompt_engine = "llamafactory"` | 组合非法，会报错 |
| `enable_thinking` 非 null + `prompt_engine` 为 null | 自动将 `prompt_engine` 设为 `"native"` |

---

## 数据准备参数（`run_prepare_dataset` 顶层参数）

### 字段说明

| 字段 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `workflow_id` | `string` | — | **是** | 所属 workflow |
| `input_path` | `string` | — | **是** | 输入表格文件路径（Excel/CSV 等） |
| `dataset_version` | `string` | — | **是** | 目标数据版本号 |
| `text_key` | `string` | — | **是** | 文本列的字段名 |
| `label_key` | `string` | — | **是** | 标签列的字段名 |
| `output_path` | `string` | `null` | 否 | 自定义输出路径；不指定时按约定路径生成 |
| `output_format` | `string` | `"jsonl"` | 否 | 输出格式 |
| `sheet_name` | `string` | `"0"` | 否 | Excel 的 sheet 名称或索引（`"0"` 表示第一个 sheet） |
| `validation_ratio` | `float` | `0.0` | 否 | 验证集切分比例，范围 `[0.0, 1.0]`；`0.0` 表示不切分验证集 |
| `split_seed` | `integer` | `42` | 否 | 数据切分的随机种子 |
| `is_test` | `boolean` | `false` | 否 | 是否为纯测试集用途（仅生成测试数据） |
| `allowed_labels` | `string[]` | `[]` | 否 | 允许的标签白名单；空数组表示不限制 |

---

## Reasoning 生成参数（`run_generate_reasoning` 顶层参数）

### 字段说明

| 字段 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `workflow_id` | `string` | — | **是** | 所属 workflow |
| `source_dataset_version` | `string` | — | **是** | 源数据版本 |
| `target_dataset_version` | `string` | — | **是** | 目标增强版本 |
| `teacher_model` | `string` | — | **是** | 教师模型名称 |
| `endpoint_url` | `string` | — | **是** | API 端点 URL |
| `label_profile` | `string` | `"l1_5class"` | 否 | 标签配置档案名 |
| `prompt_version` | `string` | `"reasoning_v1"` | 否 | Prompt 模板版本 |
| `api_key_env_var` | `string` | `"TUNEBENCH_REASONING_API_KEY"` | 否 | API Key 的环境变量名 |
| `max_concurrency` | `integer` | `5` | 否 | 最大并发请求数 |
| `request_timeout_seconds` | `float` | `60.0` | 否 | 单次请求超时时间（秒） |
| `max_attempts` | `integer` | `2` | 否 | 最大重试次数 |
| `enable_model_verify` | `boolean` | `false` | 否 | 是否启用模型验证（对生成结果做校验） |
| `resume` | `boolean` | `false` | 否 | 是否断点续传（跳过已处理的记录） |
| `sample_limit` | `integer` | `null` | 否 | 样本数量上限；`null` 表示不限制 |
| `splits` | `string[]` | `["train", "validation"]` | 否 | 要处理的数据切分列表 |

---

## 结构化目标构建参数（`run_build_structured_target` 顶层参数）

### 字段说明

| 字段 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `workflow_id` | `string` | — | **是** | 所属 workflow |
| `source_dataset_version` | `string` | — | **是** | 源 reasoning 数据版本 |
| `target_dataset_version` | `string` | — | **是** | 目标结构化版本 |
| `confidence` | `float` | `0.9` | 否 | 置信度阈值，仅保留置信度 >= 该值的记录 |
| `splits` | `string[]` | `["train", "validation"]` | 否 | 要处理的数据切分列表 |

---

## 参数传递的通用规则

1. **可选参数的省略**：所有标记为"否"的可选参数，在 MCP 调用中可以省略，系统会使用默认值。
2. **类型转换**：传入的 JSON 值会被强制转换为目标类型（如 `int()`、`float()`、`str()`），转换失败会报错。
3. **枚举约束**：标记为"枚举约束"的字段仅接受列出的值，传入其他值可能导致运行时错误。
4. **后端约束**：部分参数与后端类型（`bert`/`llamafactory`）强绑定，违反约束会在 `to_spec()` 阶段抛出 `ValueError`。
5. **null 语义**：对于 `string | null` 类型的字段，`null` 和省略在语义上等价，均表示"使用系统默认行为"。
