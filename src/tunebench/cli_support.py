"""CLI 参数校验与 spec 构建辅助工具。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import click
from click.core import ParameterSource

from tunebench.artifacts import get_model_path_manager
from tunebench.contracts import ChatSpec, EvalSpec, LoraConfigSpec, TrainSpec


def parse_sheet_name(value: str) -> str | int:
    """将命令行传入的 sheet 参数解析为 int 或 str。"""
    return int(value) if value.isdigit() else value


def parse_cli_values(values: Sequence[str] | None) -> tuple[str, ...]:
    """将重复传入或逗号分隔的参数规范化为去重后的元组。"""
    if not values:
        return ()

    normalized_values: list[str] = []
    seen_values: set[str] = set()
    for raw_value in values:
        for candidate in raw_value.split(","):
            value = candidate.strip()
            if not value or value in seen_values:
                continue
            normalized_values.append(value)
            seen_values.add(value)
    return tuple(normalized_values)


def validate_run_id_value(option_name: str, value: str | None) -> None:
    """校验 run_id 不能写成路径样式。"""
    if value is None:
        return
    if "/" in value or "\\" in value:
        raise click.UsageError(f"{option_name} 不能包含 '/' 或 '\\'，run_id 必须是项目内标识符而不是路径。")


def validate_new_run_id(backend: str, task_name: str, run_id: str | None) -> None:
    """校验训练使用的 run_id 还未占用。"""
    if run_id is None:
        return

    model_layout = get_model_path_manager().build_layout(backend, task_name, run_id)
    if model_layout.version_dir.exists():
        raise click.UsageError(
            f"--run-id={run_id} 已存在，对应目录为 {model_layout.version_dir}；请更换 run_id 或先清理已有产物。"
        )


def collect_explicit_lora_overrides(ctx: click.Context, raw_values: dict[str, object]) -> dict[str, object]:
    """仅收集用户显式传入的 LoRA 参数，用于 resume 校验。"""
    explicit_overrides: dict[str, object] = {}
    for parameter_name, parameter_value in raw_values.items():
        if ctx.get_parameter_source(parameter_name) == ParameterSource.COMMANDLINE:
            explicit_overrides[parameter_name] = parameter_value
    return explicit_overrides


def build_train_spec(
    *,
    ctx: click.Context,
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
) -> TrainSpec:
    """校验 train 命令参数并构建 TrainSpec。"""
    validate_run_id_value("--run-id", run_id)
    validate_new_run_id(backend, task_name, run_id)
    if backend == "bert" and resume_lora is None and not model_name:
        raise click.UsageError("start 模式必须传入 --model-name。")
    if backend == "llamafactory" and not model_key:
        raise click.UsageError("llamafactory 后端必须传入 --model-key。")
    if backend != "llamafactory" and instruction is not None:
        raise click.UsageError("--instruction 仅支持 --backend=llamafactory。")
    if backend == "llamafactory" and instruction is not None and not instruction.strip():
        raise click.UsageError("--instruction 不能为空字符串。")

    normalized_lora_target_modules = parse_cli_values(lora_target_modules)
    normalized_lora_modules_to_save = parse_cli_values(lora_modules_to_save)
    explicit_lora_overrides = collect_explicit_lora_overrides(
        ctx,
        {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": normalized_lora_target_modules,
            "lora_bias": lora_bias,
            "lora_modules_to_save": normalized_lora_modules_to_save,
            "use_rslora": use_rslora,
            "use_dora": use_dora,
        },
    )
    extra_args_payload: dict[str, object] = {"explicit_lora_overrides": explicit_lora_overrides}
    if instruction is not None:
        extra_args_payload["instruction"] = instruction.strip()

    return TrainSpec(
        backend=backend,
        task_name=task_name,
        model_name=model_name,
        dataset_version=dataset_version,
        model_key=model_key,
        reasoning_mode=reasoning_mode,
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
        lora=LoraConfigSpec(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=normalized_lora_target_modules,
            bias=lora_bias,
            modules_to_save=normalized_lora_modules_to_save,
            use_rslora=use_rslora,
            use_dora=use_dora,
        ),
        extra_args=extra_args_payload,
    )


def build_eval_spec(
    *,
    ctx: click.Context,
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
) -> EvalSpec:
    """校验 evaluate 命令参数并构建 EvalSpec。"""
    validate_run_id_value("--run-id", run_id)
    max_new_tokens_source = ctx.get_parameter_source("max_new_tokens")
    prompt_engine_source = ctx.get_parameter_source("prompt_engine")
    enable_thinking_source = ctx.get_parameter_source("enable_thinking")
    if backend != "llamafactory" and max_new_tokens_source == ParameterSource.COMMANDLINE:
        raise click.UsageError("--max-new-tokens 仅支持 --backend=llamafactory。")
    if backend != "llamafactory" and prompt_engine_source == ParameterSource.COMMANDLINE:
        raise click.UsageError("--prompt-engine 仅支持 --backend=llamafactory。")
    if backend != "llamafactory" and enable_thinking_source == ParameterSource.COMMANDLINE:
        raise click.UsageError("--enable-thinking/--disable-thinking 仅支持 --backend=llamafactory。")

    effective_prompt_engine = prompt_engine
    if backend == "llamafactory":
        if effective_prompt_engine is None and enable_thinking_source == ParameterSource.COMMANDLINE:
            effective_prompt_engine = "native"
        if effective_prompt_engine == "llamafactory" and enable_thinking_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--enable-thinking/--disable-thinking 仅支持 --prompt-engine=native。")

    return EvalSpec(
        backend=backend,
        task_name=task_name,
        run_id=run_id,
        dataset_version=dataset_version,
        artifact_type=artifact_type,
        batch_size=batch_size,
        max_sequence_length=(256 if backend == "bert" and max_sequence_length is None else max_sequence_length),
        max_new_tokens=max_new_tokens,
        prompt_engine=effective_prompt_engine,
        enable_thinking=enable_thinking,
        export_xlsx=not no_export_xlsx,
    )


def build_chat_spec(
    *,
    ctx: click.Context,
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
) -> ChatSpec:
    """校验 chat 命令参数并构建 ChatSpec。"""
    normalized_message = message.strip()
    if not normalized_message:
        raise click.UsageError("--message 不能为空字符串。")

    max_new_tokens_source = ctx.get_parameter_source("max_new_tokens")
    prompt_engine_source = ctx.get_parameter_source("prompt_engine")
    template_name_source = ctx.get_parameter_source("template_name")
    reasoning_mode_source = ctx.get_parameter_source("reasoning_mode")
    reasoning_suffix_style_source = ctx.get_parameter_source("reasoning_suffix_style")
    enable_thinking_source = ctx.get_parameter_source("enable_thinking")
    normalized_external_model_path = external_model_path.strip() if external_model_path is not None else None
    if normalized_external_model_path == "":
        raise click.UsageError("--external-model-path 不能为空字符串。")
    normalized_template_name = template_name.strip() if template_name is not None else None
    if normalized_template_name == "":
        raise click.UsageError("--template 不能为空字符串。")
    effective_prompt_engine = prompt_engine

    if normalized_external_model_path is not None:
        if backend != "llamafactory":
            raise click.UsageError("外部本地模型模式必须使用 --backend=llamafactory。")
        if effective_prompt_engine is None:
            effective_prompt_engine = "native"
        if ctx.get_parameter_source("artifact_type") == ParameterSource.COMMANDLINE:
            raise click.UsageError("传入 --external-model-path 时不允许再传 --artifact-type。")
        if (task_name is None) ^ (run_id is None):
            raise click.UsageError("外部本地模型模式下，--task-name 和 --run-id 要么同时传入，要么同时不传。")
        if effective_prompt_engine == "llamafactory" and normalized_template_name is None and task_name is None:
            raise click.UsageError(
                "external chat 使用 --prompt-engine=llamafactory 时必须提供 --template；若要复用内部已训练 run 的模板，可改为同时传 --task-name 和 --run-id。"
            )
        if effective_prompt_engine == "native" and template_name_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--template 仅支持 --prompt-engine=llamafactory。")
        if effective_prompt_engine == "native" and reasoning_mode_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--reasoning-mode 仅支持 --prompt-engine=llamafactory。")
        if effective_prompt_engine == "native" and reasoning_suffix_style_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--reasoning-suffix-style 仅支持 --prompt-engine=llamafactory。")
        if effective_prompt_engine == "llamafactory" and enable_thinking_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--enable-thinking/--disable-thinking 仅支持 --prompt-engine=native。")
    else:
        if effective_prompt_engine is None:
            effective_prompt_engine = "llamafactory"
        if run_id is None:
            raise click.UsageError("未传 --external-model-path 时必须提供 --run-id。")
        validate_run_id_value("--run-id", run_id)
        if task_name is None:
            default_task_name = "bert-classification" if backend == "bert" else None
            if default_task_name is None:
                raise click.UsageError("未传 --external-model-path 时，llamafactory 后端必须提供 --task-name。")
            task_name = default_task_name
        if effective_prompt_engine != "llamafactory":
            raise click.UsageError("已训练 run chat 当前仅支持 --prompt-engine=llamafactory。")
        if prompt_engine_source == ParameterSource.COMMANDLINE and backend != "llamafactory":
            raise click.UsageError("--prompt-engine 仅支持 --backend=llamafactory。")
        if template_name_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--template 仅支持外部本地模型模式。")
        if reasoning_suffix_style_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--reasoning-suffix-style 仅支持外部本地模型模式。")
        if enable_thinking_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--enable-thinking/--disable-thinking 仅支持外部本地模型模式。")

    if backend != "llamafactory":
        if instruction is not None:
            raise click.UsageError("--instruction 仅支持 --backend=llamafactory。")
        if max_new_tokens_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--max-new-tokens 仅支持 --backend=llamafactory。")
        if prompt_engine_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--prompt-engine 仅支持 --backend=llamafactory。")
        if template_name_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--template 仅支持 --backend=llamafactory。")
        if reasoning_mode_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--reasoning-mode 仅支持 --backend=llamafactory。")
        if reasoning_suffix_style_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--reasoning-suffix-style 仅支持 --backend=llamafactory。")
        if enable_thinking_source == ParameterSource.COMMANDLINE:
            raise click.UsageError("--enable-thinking/--disable-thinking 仅支持 --backend=llamafactory。")
    elif instruction is not None and not instruction.strip():
        raise click.UsageError("--instruction 不能为空字符串。")

    return ChatSpec(
        backend=backend,
        task_name=task_name,
        run_id=run_id,
        artifact_type=artifact_type,
        external_model_path=normalized_external_model_path,
        prompt_engine=effective_prompt_engine,
        message=normalized_message,
        instruction=instruction.strip() if instruction is not None else None,
        template_name=normalized_template_name,
        reasoning_mode=reasoning_mode,
        reasoning_suffix_style=reasoning_suffix_style,
        enable_thinking=enable_thinking,
        max_sequence_length=(256 if backend == "bert" and max_sequence_length is None else max_sequence_length),
        max_new_tokens=max_new_tokens,
    )


__all__ = [
    "build_chat_spec",
    "build_eval_spec",
    "build_train_spec",
    "collect_explicit_lora_overrides",
    "parse_cli_values",
    "parse_sheet_name",
    "validate_new_run_id",
    "validate_run_id_value",
]
