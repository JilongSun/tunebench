"""LlamaFactory 训练运行时文件生成工具。"""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tunebench.artifacts import ModelArtifactLayout
from tunebench.classification import ClassificationDatasetBundle, StructuredTargetDatasetBundle
from tunebench.contracts import TrainSpec

from .generation import apply_qwen_reasoning_suffix
from .models import ResolvedLlamaFactoryModel
from .prompting import build_instruction


_RUNTIME_DIRNAME = "llamafactory"
_DATASET_DIRNAME = "dataset"
_TRAIN_CONFIG_FILENAME = "train.yaml"
_EXPORT_CONFIG_FILENAME = "export.yaml"
_DATASET_INFO_FILENAME = "dataset_info.json"
_TRAIN_DATASET_NAME = "tunebench_train"
_VALIDATION_DATASET_NAME = "tunebench_validation"
_TRAIN_DATASET_FILENAME = "train.jsonl"
_VALIDATION_DATASET_FILENAME = "validation.jsonl"
_COMMANDS_FILENAME = "commands.sh"
_INSTRUCTION_EXTRA_ARG_KEY = "instruction"


@dataclass(frozen=True, slots=True)
class LlamaFactoryWorkspace:
    """描述一次训练所需的运行时工作目录。"""

    runtime_dir: Path
    dataset_dir: Path
    dataset_info_path: Path
    train_dataset_path: Path
    validation_dataset_path: Path | None
    instruction: str
    train_config_path: Path
    export_config_path: Path
    commands_path: Path
    train_command: tuple[str, ...]
    export_command: tuple[str, ...]
    resume_lora_dir: Path | None


def build_runtime_dir(model_layout: ModelArtifactLayout) -> Path:
    """返回 LlamaFactory 运行时目录。"""
    return model_layout.version_dir / _RUNTIME_DIRNAME


def _resolve_llamafactory_cli_command(subcommand: str, config_path: Path) -> tuple[str, ...]:
    """优先绑定当前解释器所在环境中的 llamafactory-cli。"""
    current_python_dir = Path(sys.executable).resolve().parent
    cli_path = current_python_dir / "llamafactory-cli"
    if cli_path.exists():
        return (str(cli_path), subcommand, str(config_path))
    return (sys.executable, "-m", "llamafactory.cli", subcommand, str(config_path))


def _resolve_instruction(spec: TrainSpec, label_names: Sequence[str]) -> str:
    """优先使用手动 instruction，否则自动构建。"""
    custom_instruction = spec.extra_args.get(_INSTRUCTION_EXTRA_ARG_KEY)
    if custom_instruction is None:
        return build_instruction(label_names)
    if not isinstance(custom_instruction, str):
        raise ValueError("llamafactory 的 instruction 配置类型无效，必须是字符串。")

    normalized_instruction = custom_instruction.strip()
    if not normalized_instruction:
        raise ValueError("llamafactory 的 instruction 不能为空字符串。")
    return normalized_instruction


def render_command(command: Sequence[str]) -> str:
    """将命令渲染为便于复制的 shell 字符串。"""
    return shlex.join(command)


def _format_yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _dump_yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, child_value in value.items():
            if isinstance(child_value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(_dump_yaml_lines(child_value, indent + 2))
                continue
            if isinstance(child_value, (list, tuple)):
                if not child_value:
                    lines.append(f"{prefix}{key}: []")
                    continue
                lines.append(f"{prefix}{key}:")
                for item in child_value:
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  -")
                        lines.extend(_dump_yaml_lines(item, indent + 4))
                    else:
                        lines.append(f"{prefix}  - {_format_yaml_scalar(item)}")
                continue
            lines.append(f"{prefix}{key}: {_format_yaml_scalar(child_value)}")
        return lines
    raise TypeError(f"不支持的 YAML 根对象类型: {type(value)!r}")


def write_yaml_file(output_path: Path, payload: dict[str, Any]) -> None:
    """写入简单 YAML 配置。"""
    output_path.write_text("\n".join(_dump_yaml_lines(payload)) + "\n", encoding="utf-8")


def _write_jsonl(output_path: Path, records: Sequence[dict[str, str]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_record_output(record: dict[str, Any]) -> str:
    target_payload = record.get("target")
    if target_payload is None:
        return str(record["label"])
    if not isinstance(target_payload, dict):
        raise ValueError("结构化 target 必须是对象。")
    return json.dumps(target_payload, ensure_ascii=False)


def _build_alpaca_records(
    records: Sequence[dict[str, Any]],
    *,
    instruction: str,
    reasoning_suffix_style: str | None,
    reasoning_mode: str,
) -> list[dict[str, str]]:
    return [
        {
            "instruction": instruction,
            "input": apply_qwen_reasoning_suffix(
                str(record["text"]),
                reasoning_suffix_style=reasoning_suffix_style,
                reasoning_mode=reasoning_mode,
            ),
            "output": _resolve_record_output(record),
        }
        for record in records
    ]


def _build_dataset_info(has_validation_split: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        _TRAIN_DATASET_NAME: {
            "file_name": _TRAIN_DATASET_FILENAME,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }
    if has_validation_split:
        payload[_VALIDATION_DATASET_NAME] = {
            "file_name": _VALIDATION_DATASET_FILENAME,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    return payload


def _build_train_config(
    *,
    spec: TrainSpec,
    model_layout: ModelArtifactLayout,
    resolved_model: ResolvedLlamaFactoryModel,
    workspace: LlamaFactoryWorkspace,
    has_validation_split: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_name_or_path": resolved_model.model_name_or_path,
        "template": resolved_model.template,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": spec.lora.r,
        "lora_alpha": spec.lora.alpha,
        "lora_dropout": spec.lora.dropout,
        "lora_target": ",".join(spec.lora.target_modules) if spec.lora.target_modules else "all",
        "use_rslora": spec.lora.use_rslora,
        "use_dora": spec.lora.use_dora,
        "dataset_dir": str(workspace.dataset_dir),
        "dataset": _TRAIN_DATASET_NAME,
        "cutoff_len": spec.max_sequence_length,
        "output_dir": str(model_layout.checkpoints_dir),
        "logging_steps": 10,
        "save_steps": 200,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "report_to": "none",
        "per_device_train_batch_size": spec.batch_size,
        "gradient_accumulation_steps": 1,
        "learning_rate": spec.learning_rate,
        "num_train_epochs": spec.num_train_epochs,
        "lr_scheduler_type": "linear",
        "warmup_ratio": spec.warmup_ratio,
        "bf16": True,
        "seed": spec.seed,
    }
    if workspace.resume_lora_dir is not None:
        payload["adapter_name_or_path"] = str(workspace.resume_lora_dir)
    if spec.lora.modules_to_save:
        payload["additional_target"] = ",".join(spec.lora.modules_to_save)
    if has_validation_split:
        payload["eval_dataset"] = _VALIDATION_DATASET_NAME
        payload["eval_strategy"] = "epoch"
    return payload


def _build_export_config(
    *,
    model_layout: ModelArtifactLayout,
    resolved_model: ResolvedLlamaFactoryModel,
) -> dict[str, Any]:
    return {
        "model_name_or_path": resolved_model.model_name_or_path,
        "adapter_name_or_path": str(model_layout.lora_dir),
        "template": resolved_model.template,
        "trust_remote_code": True,
        "export_dir": str(model_layout.merged_model_dir),
        "export_size": 5,
        "export_device": "cpu",
        "export_legacy_format": False,
    }


def prepare_training_workspace(
    *,
    spec: TrainSpec,
    model_layout: ModelArtifactLayout,
    resolved_model: ResolvedLlamaFactoryModel,
    dataset_bundle: ClassificationDatasetBundle | StructuredTargetDatasetBundle,
    resume_lora_dir: Path | None = None,
) -> LlamaFactoryWorkspace:
    """写出 LlamaFactory 所需的数据与配置文件。"""
    runtime_dir = build_runtime_dir(model_layout)
    dataset_dir = runtime_dir / _DATASET_DIRNAME
    runtime_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    instruction = _resolve_instruction(spec, tuple(dataset_bundle.label_to_id.keys()))
    train_records = _build_alpaca_records(
        dataset_bundle.train_records,
        instruction=instruction,
        reasoning_suffix_style=resolved_model.reasoning_suffix_style,
        reasoning_mode=resolved_model.reasoning_mode,
    )
    validation_records = (
        _build_alpaca_records(
            dataset_bundle.validation_records,
            instruction=instruction,
            reasoning_suffix_style=resolved_model.reasoning_suffix_style,
            reasoning_mode=resolved_model.reasoning_mode,
        )
        if dataset_bundle.validation_records is not None
        else None
    )

    train_dataset_path = dataset_dir / _TRAIN_DATASET_FILENAME
    validation_dataset_path = dataset_dir / _VALIDATION_DATASET_FILENAME if validation_records is not None else None
    dataset_info_path = dataset_dir / _DATASET_INFO_FILENAME
    train_config_path = runtime_dir / _TRAIN_CONFIG_FILENAME
    export_config_path = runtime_dir / _EXPORT_CONFIG_FILENAME
    commands_path = runtime_dir / _COMMANDS_FILENAME

    _write_jsonl(train_dataset_path, train_records)
    if validation_dataset_path is not None and validation_records is not None:
        _write_jsonl(validation_dataset_path, validation_records)
    dataset_info_path.write_text(
        json.dumps(_build_dataset_info(validation_records is not None), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    workspace = LlamaFactoryWorkspace(
        runtime_dir=runtime_dir,
        dataset_dir=dataset_dir,
        dataset_info_path=dataset_info_path,
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        instruction=instruction,
        train_config_path=train_config_path,
        export_config_path=export_config_path,
        commands_path=commands_path,
        train_command=_resolve_llamafactory_cli_command("train", train_config_path),
        export_command=_resolve_llamafactory_cli_command("export", export_config_path),
        resume_lora_dir=resume_lora_dir,
    )

    write_yaml_file(
        train_config_path,
        _build_train_config(
            spec=spec,
            model_layout=model_layout,
            resolved_model=resolved_model,
            workspace=workspace,
            has_validation_split=validation_records is not None,
        ),
    )
    write_yaml_file(
        export_config_path,
        _build_export_config(model_layout=model_layout, resolved_model=resolved_model),
    )
    commands_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                render_command(workspace.train_command),
                render_command(workspace.export_command),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return workspace