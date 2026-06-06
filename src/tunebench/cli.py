"""TuneBench 第一版命令行入口。"""

from __future__ import annotations

import json
from dataclasses import asdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import click
import cloup

from tunebench.backends import REASONING_MODES, get_classification_backend, list_classification_backend_names
from tunebench.cli_support import build_chat_spec, build_eval_spec, build_train_spec, parse_cli_values, parse_sheet_name
from tunebench.classification import ClassificationDataPreparer, TEST_SPLIT_NAME, TRAIN_SPLIT_NAME, VALIDATION_SPLIT_NAME
from tunebench.contracts import ChatResult, DatasetSpec, ReasoningGenerationSpec, StageResult, StructuredTargetBuildSpec
from tunebench.util import get_logger, get_result_logger, setup_logging


logger = get_logger("cli")
result_logger = get_result_logger()
FORMATTER_SETTINGS = cloup.HelpFormatter.settings(col1_max_width=32, col2_min_width=28, max_width=120)


def _log_stage_result(result: StageResult) -> None:
    """统一输出 StageResult。"""
    payload = asdict(result)
    logger.info("执行完成: stage=%s success=%s", result.stage, result.success)
    result_logger.info(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _emit_chat_result(result: ChatResult, *, emit_json: bool) -> None:
    """统一输出 chat 结果。"""
    payload = asdict(result)
    if emit_json:
        result_logger.info(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return

    if result.success:
        result_logger.info("输出: %s", result.output_text)
        return

    result_logger.info("执行失败: %s", result.message)


@cloup.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    formatter_settings=FORMATTER_SETTINGS,
    no_args_is_help=True,
)
def cli_app() -> None:
    """TuneBench 第一版 BERT 数据处理、训练与评测命令行入口。"""


@cli_app.command("prepare-data", help="执行数据清洗与格式转换。")
@cloup.option("-i", "--input-path", required=True, type=click.Path(path_type=Path), help="原始数据路径。")
@cloup.option("-t", "--task-name", default="bert-classification", show_default=True, help="任务名称。")
@cloup.option("-d", "--dataset-version", required=True, help="数据版本号，例如 v20260509_001。")
@cloup.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path),
    help="可选，额外导出最终标准数据的文件路径；资产目录内仍会写入 final 层。",
)
@cloup.option(
    "-f",
    "--output-format",
    type=click.Choice(["json", "jsonl"]),
    default="jsonl",
    show_default=True,
    help="final 层导出格式。",
)
@cloup.option("-n", "--sheet-name", default="0", show_default=True, help="Excel sheet 名称或索引，默认 0。")
@cloup.option("-x", "--text-key", required=True, help="原始数据中的文本字段名（必填）。")
@cloup.option("-l", "--label-key", required=True, help="原始数据中的标签字段名（必填）。")
@cloup.option(
    "-r",
    "--validation-ratio",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help="默认写入 train；可选，按 label 分层切出的 validation 比例，例如 0.1。",
)
@cloup.option("-e", "--split-seed", type=int, default=42, show_default=True, help="自动切分时的随机种子。")
@cloup.option("--is-test", is_flag=True, help="测试集模式；固定写入 test，且与 validation-ratio 互斥。")
@cloup.option(
    "--keep-label",
    "-kl",
    "keep_labels",
    multiple=True,
    help="可重复传入，仅保留指定标签；未命中的标签会在清洗阶段被丢弃。也支持逗号分隔。如果不加这个关键字会选中全部的标签。",
)
def prepare_data_command(
    input_path: Path,
    task_name: str,
    dataset_version: str,
    output_path: Path | None,
    output_format: str,
    sheet_name: str,
    text_key: str,
    label_key: str,
    validation_ratio: float,
    split_seed: int,
    is_test: bool,
    keep_labels: tuple[str, ...],
) -> None:
    """执行数据清洗命令。"""
    if is_test and validation_ratio > 0:
        raise click.UsageError("--is-test 与 --validation-ratio 互斥，请二选一。")

    logger.info("收到命令: prepare-data")
    cleaner = ClassificationDataPreparer()
    spec = DatasetSpec(
        task_name=task_name,
        input_path=input_path,
        dataset_version=dataset_version,
        output_path=output_path,
        text_key=text_key,
        label_key=label_key,
        output_format=output_format,
        sheet_name=parse_sheet_name(sheet_name),
        validation_ratio=validation_ratio,
        split_seed=split_seed,
        is_test=is_test,
        allowed_labels=parse_cli_values(keep_labels),
    )
    _log_stage_result(cleaner.run(spec))


@cli_app.command("generate-reasoning", help="基于标准分类数据生成 reasoning 增强版本。")
@cloup.option_group(
    "输入输出",
    cloup.option("-t", "--task-name", default="bert-classification", show_default=True, help="任务名称。"),
    cloup.option("--source-dataset-version", required=True, help="prepare-data 产出的源数据版本。"),
    cloup.option("--target-dataset-version", required=True, help="reasoning 增强后的目标数据版本。"),
    cloup.option(
        "--split",
        "splits",
        multiple=True,
        type=click.Choice([TRAIN_SPLIT_NAME, VALIDATION_SPLIT_NAME, TEST_SPLIT_NAME]),
        help="可重复传入，指定需要处理的 split；默认处理 train,validation。",
    ),
)
@cloup.option_group(
    "模型与提示",
    cloup.option("--teacher-model", required=True, help="外部大模型名称或路径。"),
    cloup.option(
        "--endpoint-url",
        default="http://192.168.75.109:30114/v1/chat/completions",
        show_default=True,
        help="OpenAI 兼容 chat completions 接口地址。",
    ),
    cloup.option("--label-profile", type=click.Choice(["l1_5class"]), default="l1_5class", show_default=True, help="标签规则配置。"),
    cloup.option("--prompt-version", default="reasoning_v1", show_default=True, help="prompt 模板版本。"),
    cloup.option(
        "--api-key-env-var",
        default="TUNEBENCH_REASONING_API_KEY",
        show_default=True,
        help="Bearer Token 所在的环境变量名；若未设置则不附带 Authorization 头。",
    ),
)
@cloup.option_group(
    "运行控制",
    cloup.option("--max-concurrency", type=click.IntRange(1), default=5, show_default=True, help="异步并发上限。"),
    cloup.option(
        "--request-timeout-seconds",
        type=click.FloatRange(min=0.1),
        default=60.0,
        show_default=True,
        help="单次请求超时时间（秒）。",
    ),
    cloup.option("--max-attempts", type=click.IntRange(1), default=2, show_default=True, help="每条样本最多重试次数。"),
    cloup.option(
        "--enable-model-verify/--disable-model-verify",
        "enable_model_verify",
        default=False,
        help="是否使用大模型对生成的 reasoning 做二次校验；默认关闭。",
    ),
    cloup.option("--resume", is_flag=True, help="断点续跑；已存在于 stage 的 source_index 会跳过。"),
    cloup.option("--sample-limit", type=click.IntRange(1), help="调试用，仅处理每个 split 的前 N 条样本。"),
)
def generate_reasoning_command(
    task_name: str,
    source_dataset_version: str,
    target_dataset_version: str,
    teacher_model: str,
    endpoint_url: str,
    label_profile: str,
    prompt_version: str,
    api_key_env_var: str,
    max_concurrency: int,
    request_timeout_seconds: float,
    max_attempts: int,
    enable_model_verify: bool,
    resume: bool,
    sample_limit: int | None,
    splits: tuple[str, ...],
) -> None:
    """执行 reasoning 数据增强命令。"""
    if source_dataset_version == target_dataset_version:
        raise click.UsageError("--source-dataset-version 与 --target-dataset-version 不能相同。")

    try:
        from tunebench.classification.reasoning_generator import ClassificationReasoningGenerator
    except ModuleNotFoundError as exc:
        if exc.name == "httpx":
            raise click.ClickException("generate-reasoning 依赖 httpx，请先手动安装 httpx>=0.27,<1.0。") from exc
        raise

    logger.info("收到命令: generate-reasoning")
    generator = ClassificationReasoningGenerator()
    normalized_splits = parse_cli_values(splits) if splits else (TRAIN_SPLIT_NAME, VALIDATION_SPLIT_NAME)
    spec = ReasoningGenerationSpec(
        task_name=task_name,
        source_dataset_version=source_dataset_version,
        target_dataset_version=target_dataset_version,
        teacher_model=teacher_model,
        endpoint_url=endpoint_url,
        label_profile=label_profile,
        prompt_version=prompt_version,
        api_key_env_var=api_key_env_var,
        max_concurrency=max_concurrency,
        request_timeout_seconds=request_timeout_seconds,
        max_attempts=max_attempts,
        enable_model_verify=enable_model_verify,
        resume=resume,
        sample_limit=sample_limit,
        splits=normalized_splits,
    )
    _log_stage_result(generator.run(spec))


@cli_app.command("build-structured-target", help="将 reasoning 数据转换为固定结构化 target。")
@cloup.option("-t", "--task-name", default="bert-classification", show_default=True, help="任务名称。")
@cloup.option("--source-dataset-version", required=True, help="reasoning 增强后的源数据版本。")
@cloup.option("--target-dataset-version", required=True, help="结构化 target 的目标数据版本。")
@cloup.option(
    "--confidence",
    type=click.Choice(["0.3", "0.6", "0.9"]),
    default="0.9",
    show_default=True,
    help="单标签结构化 target 默认置信度。",
)
@cloup.option(
    "--split",
    "splits",
    multiple=True,
    type=click.Choice([TRAIN_SPLIT_NAME, VALIDATION_SPLIT_NAME, TEST_SPLIT_NAME]),
    help="可重复传入，指定需要处理的 split；默认处理 train,validation。",
)
def build_structured_target_command(
    task_name: str,
    source_dataset_version: str,
    target_dataset_version: str,
    confidence: str,
    splits: tuple[str, ...],
) -> None:
    """执行结构化 target 构建命令。"""
    if source_dataset_version == target_dataset_version:
        raise click.UsageError("--source-dataset-version 与 --target-dataset-version 不能相同。")

    from tunebench.classification import ClassificationStructuredTargetBuilder

    logger.info("收到命令: build-structured-target")
    builder = ClassificationStructuredTargetBuilder()
    normalized_splits = parse_cli_values(splits) if splits else (TRAIN_SPLIT_NAME, VALIDATION_SPLIT_NAME)
    spec = StructuredTargetBuildSpec(
        task_name=task_name,
        source_dataset_version=source_dataset_version,
        target_dataset_version=target_dataset_version,
        confidence=float(confidence),
        splits=normalized_splits,
    )
    _log_stage_result(builder.run(spec))


@cli_app.command("train", help="执行 BERT 微调。")
@cloup.option_group(
    "通用配置",
    cloup.option("--backend", type=click.Choice(list_classification_backend_names()), default="bert", show_default=True, help="训练后端。"),
    cloup.option("-t", "--task-name", default="bert-classification", show_default=True, help="任务名称。"),
    cloup.option("-m", "--model-name", help="模型名称，例如 bert-base-chinese；start 模式必填，resume 模式可省略。"),
    cloup.option("--model-key", help="llamafactory 后端使用的模型注册键，例如 qwen3_4b。"),
    cloup.option("--instruction", help="llamafactory 后端可选，手动指定分类 instruction；未指定时自动构建。"),
    cloup.option("--reasoning-mode", type=click.Choice(REASONING_MODES), help="llamafactory 后端的思考模式。"),
    cloup.option("-d", "--dataset-version", required=True, help="训练使用的数据版本。"),
    cloup.option("--resume-lora", help="可选，继续训练时指定已有 LoRA 头；优先按当前 task 下的 run_id 解析，找不到时再按外部路径解析。"),
    cloup.option("-r", "--run-id", help="可选，自定义 run_id；不传则自动生成。"),
    cloup.option("-o", "--export-dir", type=click.Path(path_type=Path), help="可选，额外导出训练元数据目录。"),
    cloup.option("-n", "--num-labels", type=click.IntRange(1), help="可选，显式指定标签数并与数据校验。"),
    cloup.option("-S", "--seed", type=int, default=42, show_default=True, help="随机种子。"),
)
@cloup.option_group(
    "训练超参数",
    cloup.option("-l", "--learning-rate", type=float, default=2e-5, show_default=True, help="学习率。"),
    cloup.option("-b", "--batch-size", type=click.IntRange(1), default=8, show_default=True, help="批大小。"),
    cloup.option("-e", "--num-train-epochs", type=click.IntRange(1), default=3, show_default=True, help="训练轮数。"),
    cloup.option("-x", "--max-sequence-length", type=click.IntRange(1), default=256, show_default=True, help="最大序列长度。"),
    cloup.option("-w", "--warmup-ratio", type=click.FloatRange(0.0, 1.0), default=0.0, show_default=True, help="warmup 比例。"),
)
@cloup.option_group(
    "LoRA 配置",
    cloup.option("--lora-r", type=click.IntRange(1), default=8, show_default=True, help="LoRA 秩。"),
    cloup.option("--lora-alpha", type=click.IntRange(1), default=16, show_default=True, help="LoRA alpha。"),
    cloup.option("--lora-dropout", type=click.FloatRange(0.0, 1.0), default=0.1, show_default=True, help="LoRA dropout。"),
    cloup.option(
        "--lora-target-module",
        "lora_target_modules",
        multiple=True,
        help="可重复传入或使用逗号分隔，显式指定 LoRA 注入模块；不传时按模型自动推断。",
    ),
    cloup.option(
        "--lora-bias",
        type=click.Choice(["none", "all", "lora_only"]),
        default="none",
        show_default=True,
        help="LoRA bias 策略。",
    ),
    cloup.option(
        "--lora-modules-to-save",
        "lora_modules_to_save",
        multiple=True,
        help="可重复传入或使用逗号分隔，指定除 LoRA 外额外保存的模块。",
    ),
    cloup.option("--use-rslora", is_flag=True, help="启用 RSLoRA。"),
    cloup.option("--use-dora", is_flag=True, help="启用 DoRA。"),
)
def train_command(
    backend: str,
    task_name: str,
    model_name: str,
    model_key: str | None,
    instruction: str | None,
    reasoning_mode: Literal["think", "no_think"] | None,
    dataset_version: str,
    resume_lora: str | None,
    run_id: str | None,
    export_dir: Path | None,
    num_labels: int | None,
    seed: int,
    learning_rate: float,
    batch_size: int,
    num_train_epochs: int,
    max_sequence_length: int,
    warmup_ratio: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: tuple[str, ...],
    lora_bias: Literal["none", "all", "lora_only"],
    lora_modules_to_save: tuple[str, ...],
    use_rslora: bool,
    use_dora: bool,
) -> None:
    """执行训练命令。"""
    logger.info("收到命令: train")
    backend_runner = get_classification_backend(backend)
    spec = build_train_spec(
        ctx=click.get_current_context(),
        backend=backend,
        task_name=task_name,
        model_name=model_name,
        model_key=model_key,
        instruction=instruction,
        reasoning_mode=reasoning_mode,
        dataset_version=dataset_version,
        resume_lora=resume_lora,
        run_id=run_id,
        export_dir=export_dir,
        num_labels=num_labels,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        max_sequence_length=max_sequence_length,
        warmup_ratio=warmup_ratio,
        seed=seed,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_bias=lora_bias,
        lora_modules_to_save=lora_modules_to_save,
        use_rslora=use_rslora,
        use_dora=use_dora,
    )
    _log_stage_result(backend_runner.run_train(spec))


@cli_app.command("evaluate", help="执行模型评测。")
@cloup.option("--backend", type=click.Choice(list_classification_backend_names()), default="bert", show_default=True, help="评测后端。")
@cloup.option("-t", "--task-name", default="bert-classification", show_default=True, help="任务名称。")
@cloup.option("-r", "--run-id", required=True, help="待评测模型对应的 run_id。")
@cloup.option("-d", "--dataset-version", required=True, help="评测使用的数据版本。")
@cloup.option(
    "-a",
    "--artifact-type",
    type=click.Choice(["merged", "lora"]),
    default="merged",
    show_default=True,
    help="评测产物类型。",
)
@cloup.option("-b", "--batch-size", type=click.IntRange(1), default=8, show_default=True, help="评测批大小。")
@cloup.option("-x", "--max-sequence-length", type=click.IntRange(1), help="评测最大输入长度；llamafactory 默认不主动截断。")
@cloup.option("--max-new-tokens", type=click.IntRange(1), help="llamafactory 后端最大生成 token 数；默认生成到 EOS 或上下文上限。")
@cloup.option("--prompt-engine", type=click.Choice(["llamafactory", "native"]), help="llamafactory 评测 prompt 渲染引擎；未显式指定时，qwen3.5 run 默认走 native，其余默认走 llamafactory。")
@cloup.option("--enable-thinking/--disable-thinking", default=None, help="仅 --prompt-engine=native 时透传给模型原生 chat template 的 enable_thinking 开关。")
@cloup.option("--no-export-xlsx", is_flag=True, help="关闭默认的 XLSX 汇总导出。")
def evaluate_command(
    backend: str,
    task_name: str,
    run_id: str,
    dataset_version: str,
    artifact_type: str,
    batch_size: int,
    max_sequence_length: int | None,
    max_new_tokens: int | None,
    prompt_engine: str | None,
    enable_thinking: bool | None,
    no_export_xlsx: bool,
) -> None:
    """执行评测命令。"""
    logger.info("收到命令: evaluate")
    backend_runner = get_classification_backend(backend)
    spec = build_eval_spec(
        ctx=click.get_current_context(),
        backend=backend,
        task_name=task_name,
        run_id=run_id,
        dataset_version=dataset_version,
        artifact_type=artifact_type,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        max_new_tokens=max_new_tokens,
        prompt_engine=prompt_engine,
        enable_thinking=enable_thinking,
        no_export_xlsx=no_export_xlsx,
    )
    _log_stage_result(backend_runner.run_evaluate(spec))


@cli_app.command("chat", help="执行单轮聊天或单条分类推理。")
@cloup.option_group(
    "模型来源",
    cloup.option("--backend", type=click.Choice(list_classification_backend_names()), default="bert", show_default=True, help="推理后端。"),
    cloup.option("-t", "--task-name", help="已训练模型所属任务名称。"),
    cloup.option("-r", "--run-id", help="已训练模型对应的 run_id。"),
    cloup.option(
        "-a",
        "--artifact-type",
        type=click.Choice(["merged", "lora"]),
        default="merged",
        show_default=True,
        help="已训练模型的推理产物类型。",
    ),
    cloup.option(
        "--external-model-path",
        help="外部本地 GPT 类模型目录；传入后固定走 llamafactory backend。external chat 默认使用原生 native prompt-engine；若未传 --instruction，可额外配合 --task-name/--run-id 复用内部 prompt。",
    ),
    cloup.option(
        "--prompt-engine",
        type=click.Choice(["llamafactory", "native"]),
        help="chat prompt 渲染引擎；external chat 默认 native，run chat 默认 llamafactory。",
    ),
    cloup.option("--template", "template_name", help="仅 --prompt-engine=llamafactory 时使用的外部模型 LlamaFactory 模板名。"),
)
@cloup.option_group(
    "输入参数",
    cloup.option("-m", "--message", required=True, help="单轮输入文本。"),
    cloup.option("--instruction", help="llamafactory 后端可选，自定义 instruction。external chat 未传时，可直接仅发送 message，或额外配合 --task-name/--run-id 复用内部 prompt。"),
)
@cloup.option_group(
    "生成参数",
    cloup.option("-x", "--max-sequence-length", type=click.IntRange(1), help="最大输入长度；llamafactory 默认不主动截断。"),
    cloup.option("--max-new-tokens", type=click.IntRange(1), help="llamafactory 后端最大生成 token 数；默认生成到 EOS 或上下文上限。"),
    cloup.option("--reasoning-mode", type=click.Choice(["think", "no_think"]), help="仅 --prompt-engine=llamafactory 时使用的思考模式。"),
    cloup.option("--reasoning-suffix-style", type=click.Choice(["qwen3"]), help="仅 --prompt-engine=llamafactory 时使用的消息后缀控制风格。"),
    cloup.option("--enable-thinking/--disable-thinking", default=None, help="仅 --prompt-engine=native 时透传给模型原生 chat template 的 enable_thinking 开关。"),
)
@cloup.option_group(
    "输出参数",
    cloup.option("--json", "emit_json", is_flag=True, help="以 JSON 结构输出结果。"),
)
def chat_command(
    backend: str,
    task_name: str | None,
    run_id: str | None,
    artifact_type: str,
    external_model_path: str | None,
    prompt_engine: str | None,
    template_name: str | None,
    message: str,
    instruction: str | None,
    max_sequence_length: int | None,
    max_new_tokens: int | None,
    reasoning_mode: str | None,
    reasoning_suffix_style: str | None,
    enable_thinking: bool | None,
    emit_json: bool,
) -> None:
    """执行 chat 命令。"""
    logger.info("收到命令: chat")
    backend_runner = get_classification_backend(backend)
    spec = build_chat_spec(
        ctx=click.get_current_context(),
        backend=backend,
        task_name=task_name,
        run_id=run_id,
        artifact_type=artifact_type,
        external_model_path=external_model_path,
        prompt_engine=prompt_engine,
        message=message,
        instruction=instruction,
        template_name=template_name,
        reasoning_mode=reasoning_mode,
        reasoning_suffix_style=reasoning_suffix_style,
        enable_thinking=enable_thinking,
        max_sequence_length=max_sequence_length,
        max_new_tokens=max_new_tokens,
    )
    _emit_chat_result(backend_runner.run_chat(spec), emit_json=emit_json)


@cli_app.command("plan", help="检查参数并输出执行计划。")
@cloup.option("--config", required=True, type=click.Path(path_type=Path), help="计划使用的配置文件路径。")
def plan_command(config: Path) -> None:
    """输出执行计划。"""
    logger.info("收到命令: plan")
    plan = {
        "stage": "plan",
        "config": str(config),
        "notes": [
            "当前为第一版骨架，plan 命令后续用于做参数检查、资源估算与执行计划输出。",
        ],
    }
    result_logger.info(json.dumps(plan, ensure_ascii=False, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口。"""
    setup_logging()
    try:
        cli_app.main(args=list(argv) if argv is not None else None, prog_name="tunebench", standalone_mode=False)
    except click.exceptions.Exit as exc:
        logger.error(f"命令执行失败: {exc}")
        return exc.exit_code
    except click.ClickException as exc:
        logger.error(f"命令执行失败: {exc}")
        exc.show()
        return exc.exit_code
    except click.Abort:
        logger.error("命令执行已取消。")
        click.echo("已取消。", err=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
