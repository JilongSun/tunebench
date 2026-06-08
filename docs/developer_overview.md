# 开发者总览

本文档面向参与 TuneBench 开发的工程师，说明当前正式架构、模块边界、文档分工和重点关注点。

## 文档分工

为避免临时方案文档与正式文档并存，当前文档职责约束如下：

- `README.md`：对外说明项目定位、当前能力、快速使用方式与文档入口。
- `docs/developer_overview.md`：面向开发者说明正式架构、模块边界、稳定约束与阅读顺序。
- `docs/bert_workflow.md`：只描述 BERT 路径的专项工作流，不承担通用架构说明。
- `docs/path_structure.md`：只描述资产目录、评测产物和 metadata 布局，不承担模块设计说明。
- `docs/workflow_architecture.md`：只描述 workflow 编排层、状态存储与子进程执行模型。
- `docs/roadmap.md`：只描述后续目标和推进方向，不回放已完成的临时重构计划。

## 项目能力概览

TuneBench 已经形成面向双后端分类任务的稳定结构，项目已经具备：

- 统一 CLI 入口与共享契约对象。
- 数据清洗、reasoning 生成、structured target 构建。
- BERT LoRA 微调、继续训练、训练期验证与独立评测。
- LlamaFactory 分类训练、导出、独立评测与单轮 chat。
- 聚合指标、按标签指标、样本级预测结果导出。
- 训练图表、评测报表、资产版本管理与训练 metadata schema。

可以这样理解：

- BERT 路径已经是最完整、最稳定的基线路径。
- LlamaFactory 路径已经形成完整闭环，并完成了第一轮结构拆分。
- 多后端共享边界已经稳定到足以承接后续能力扩展，但模型覆盖和语义细节仍会继续收紧。

## 正式代码结构

当前代码主要按“接口层 + 共享分类能力 + 后端实现 + 资产边界”组织：

- `src/tunebench/cli.py`：统一命令入口、日志和结果输出。
- `src/tunebench/cli_support.py`：CLI 参数级业务校验与 spec 组装辅助模块。
- `src/tunebench/workflow/`：workflow 编排、状态存储、异步子进程执行与 operation 级资源校验。
- `src/tunebench/contracts/`：CLI 与后端之间的统一契约对象。
- `src/tunebench/classification/`：分类任务共享的数据加载、指标、reasoning 数据加工和训练期评估回调。
- `src/tunebench/backends/`：后端注册与各训练后端的 plan、train、evaluate、chat 实现。
- `src/tunebench/artifacts/`：资产路径、产物命名、评测文件存储与训练 run metadata schema。
- `src/tunebench/util/`：日志、表格等弱业务耦合的通用工具。
- `src/tunebench_mcp/`：MCP server、workflow 工具暴露与运行入口。

这套结构的目标不是做抽象展示，而是确保双后端路径在共享 CLI、路径与 metadata 约束的前提下仍能独立演进。

## 当前模块边界

### 接口层

接口层当前主要由 `cli.py` 和 `cli_support.py` 组成，职责是：

- 暴露稳定命令入口。
- 做参数文本级和请求语义级校验。
- 组装 `TrainSpec`、`EvalSpec`、`ChatSpec` 等契约对象。
- 统一输出计划、环节结果和 chat 结果。

CLI 不再直接承担大段后端业务逻辑，也不再同时维护多套复杂组合校验。

### workflow 编排层

`workflow/` 负责承接多环节实验链路的编排问题，职责是：

- 为 MCP 和外部调度提供稳定的 workflow 状态语义。
- 维护 workflow 主记录、operation 运行记录和事件流。
- 通过异步子进程执行具体环节，避免控制进程提前加载训练模块。
- 在启动前校验输入资源存在性与输出覆盖风险。

这层不重写训练、评测和数据处理逻辑，而是复用既有环节执行器，并为其补充编排、持久化和查询能力。

### MCP 接入层

`tunebench_mcp/` 负责把 workflow 应用层能力暴露为 MCP 工具，职责是：

- 管理 MCP server 生命周期。
- 将 workflow 服务映射为结构化工具接口。
- 为外部 MCP client 提供创建 workflow、启动 operation、查询状态和读取日志能力。

这层不承担训练逻辑，也不负责状态持久化；它位于 `tunebench` 包之外，作为 workflow 层的协议适配入口。

### 共享分类层

`classification/` 负责被多个后端复用的任务语义，当前主要关注：

- 数据读取与标准化。
- 标签空间与数据约束。
- 通用分类指标与结构化输出指标。
- reasoning 数据生成与 structured target 构建。
- 训练期 validation 回调与结果整理。

### 资产与 metadata 层

`artifacts/` 负责路径和产物语义，当前已经收敛出两个稳定职责：

- 使用路径管理器统一约束数据版本与模型 run 目录。
- 使用 `run_metadata.py` 中的 dataclass schema 统一表达训练 run metadata。

训练过程中写入的 `metadata.json` 不再只是松散字典的临时堆叠，而是围绕统一 schema 与 manifest 构建函数演进。

### 后端层

后端层负责模型族差异与运行编排。当前规则是：

- registry 决定后端选择，CLI 不直接绑死某个实现。
- BERT 与 LlamaFactory 各自维护本后端的训练、评测与聊天流程。
- 多后端共享的请求对象、路径和 metadata 由公共层提供。

## LlamaFactory 后端当前拆分

`src/tunebench/backends/llamafactory/` 当前已经完成第一轮职责拆分，主要模块如下：

- `model_profiles.py`：模型能力单一事实源。
- `policies.py`：reasoning policy 构建与收敛。
- `metadata.py`：metadata 解析与推理时配置恢复。
- `loaders.py`：模型、tokenizer、processor、template 装载。
- `generation.py`：prompt 构建、编码与生成执行。
- `chat_renderers.py`：external chat 下 native 渲染链兼容。
- `validation_recovery.py`：训练后 validation recovery。
- `trainer_state_recovery.py`：trainer state 解析与历史指标回收。
- `artifact_sync.py`：最终 adapter 与 LoRA 目录同步。
- `run_lifecycle.py`：训练生命周期 metadata 写回与导出。
- `train_runner.py`、`eval_runner.py`、`chat_runner.py`：面向流程编排的入口 runner。

其中 `inference.py` 与 `models.py` 当前只承担兼容导出角色，不应继续新增业务逻辑。

## 当前已形成的工程约束

- CLI 仍是当前唯一正式入口，但已经回归接口层职责。
- workflow 层负责跨命令编排，不直接替代 CLI 的单次执行职责。
- 训练、评测和资产写入必须返回结构化结果。
- `artifacts/` 负责路径和产物命名，不再把这部分散落在 `util/` 或后端实现里。
- 后端选择通过 registry 完成，CLI 不直接绑死某个训练实现。
- 验证集与测试集必须明确区分。
- 每次训练都必须留下可追溯 metadata。
- 兼容层文件不继续承载新增逻辑，新增逻辑必须落到对应职责模块。

## 当前仍需关注的点

结构性重构已经完成一轮，但以下问题仍是近期重点：

- Qwen 模型映射还不是全覆盖，`qwen3_6_27b` 仍未开放。
- Qwen3 与 Qwen3.5 的 no_think 机制不同，训练、评测和 chat 仍需要继续保持一致语义。
- external chat 仍同时存在 native 与 LlamaFactory 两条 prompt 渲染链，后续要继续收紧边界和参数语义。
- 现有拆分已经足够支撑继续开发，是否继续抽出更显式的 application service，应由新增需求驱动，而不是提前设计。

## 推荐阅读顺序

建议新加入项目的开发者按以下顺序阅读：

1. `README.md`
2. `docs/developer_overview.md`
3. `docs/workflow_architecture.md`
4. `docs/bert_workflow.md`
5. `docs/path_structure.md`
6. `docs/roadmap.md`

读完后，再进入具体模块代码会更高效。
