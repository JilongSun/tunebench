"""LlamaFactory 分类训练后端。"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import replace
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

from peft import PeftConfig

from tunebench.artifacts import (
    DatasetPathManager,
    EvalArtifactStore,
    FileSystemEvalArtifactStore,
    METADATA_FILENAME,
    ModelPathManager,
    get_dataset_path_manager,
    get_model_path_manager,
)
from tunebench.classification import StructuredTargetDatasetBundle, load_structured_target_dataset_bundle
from tunebench.contracts import LoraConfigSpec, RunPlan, StageResult, TrainSpec
from tunebench.util import get_logger

from .artifact_sync import sync_trained_adapter_to_lora_dir
from .metadata import load_metadata, resolve_model_key, resolve_reasoning_mode, resolve_template_name
from .models import ResolvedLlamaFactoryModel, resolve_model_variant
from .policies import build_train_reasoning_policy
from .run_lifecycle import build_lifecycle_manifest, export_metadata_copy, write_metadata
from .runtime import build_runtime_dir, prepare_training_workspace, render_command
from .trainer_state_recovery import build_train_result, recover_trainer_history_metrics


logger = get_logger("backends.llamafactory.train_runner")

_LLAMAFACTORY_BACKEND = "llamafactory"
_LLAMAFACTORY_CLI_NAME = "llamafactory-cli"


class ResolvedResumeLora:
    """描述一次继续微调所使用的源 LoRA。"""

    def __init__(
        self,
        *,
        lora_dir: Path,
        source_kind: str,
        source_run_id: str | None,
        metadata_path: Path,
        source_model_key: str,
        source_reasoning_mode: str | None,
        source_template: str | None,
        base_model_name_or_path: str,
        lora: LoraConfigSpec,
    ) -> None:
        self.lora_dir = lora_dir
        self.source_kind = source_kind
        self.source_run_id = source_run_id
        self.metadata_path = metadata_path
        self.source_model_key = source_model_key
        self.source_reasoning_mode = source_reasoning_mode
        self.source_template = source_template
        self.base_model_name_or_path = base_model_name_or_path
        self.lora = lora


class LlamaFactoryClassificationTrainRunner:
    """负责封装 LlamaFactory 分类训练流程。"""

    def __init__(
        self,
        dataset_path_manager: DatasetPathManager | None = None,
        model_path_manager: ModelPathManager | None = None,
        artifact_store: EvalArtifactStore | None = None,
    ) -> None:
        self.dataset_path_manager = dataset_path_manager or get_dataset_path_manager()
        self.model_path_manager = model_path_manager or get_model_path_manager()
        self.artifact_store = artifact_store or FileSystemEvalArtifactStore()

    def _generate_run_id(self) -> str:
        return datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")

    def _require_model_key(self, spec: TrainSpec) -> str:
        if spec.model_key is None or not spec.model_key.strip():
            raise ValueError("llamafactory 后端要求显式提供 model_key。")
        return spec.model_key

    def _resolve_model(self, spec: TrainSpec) -> ResolvedLlamaFactoryModel:
        model_key = self._require_model_key(spec)
        resolved_model = resolve_model_variant(model_key, spec.reasoning_mode)
        if spec.model_name is None:
            return resolved_model

        normalized_model_name = spec.model_name.strip()
        if not normalized_model_name:
            raise ValueError("llamafactory 后端的 --model-name 不能为空字符串。")
        return replace(resolved_model, model_name_or_path=normalized_model_name)

    def _validate_spec(self, spec: TrainSpec) -> None:
        if spec.lora.bias != "none":
            raise ValueError("当前版本的 llamafactory 后端暂不支持 --lora-bias，必须保持 none。")

    def _is_path_like(self, value: str) -> bool:
        return "/" in value or "\\" in value

    def _normalize_resume_lora_value(self, parameter_name: str, value: Any) -> Any:
        if parameter_name in {"lora_target_modules", "lora_modules_to_save"}:
            if not value:
                return ()
            return tuple(value)
        if parameter_name in {"use_rslora", "use_dora"}:
            return bool(value)
        return value

    def _resolve_resume_lora_dir(self, spec: TrainSpec) -> tuple[Path | None, str | None, str | None]:
        if spec.resume_lora is None:
            return None, None, None

        candidate = spec.resume_lora.strip()
        if not candidate:
            raise ValueError("resume_lora 不能为空字符串。")

        if not self._is_path_like(candidate):
            internal_lora_dir = self.model_path_manager.get_lora_dir(_LLAMAFACTORY_BACKEND, spec.task_name, candidate)
            if internal_lora_dir.exists():
                logger.info("resume_lora 已按项目内 run_id 解析: run_id=%s, path=%s", candidate, internal_lora_dir)
                return internal_lora_dir, "internal_run", candidate

        external_lora_dir = Path(candidate).expanduser()
        if external_lora_dir.exists():
            logger.info("resume_lora 已按外部路径解析: input=%s, path=%s", candidate, external_lora_dir)
            return external_lora_dir, "external_path", None

        raise FileNotFoundError(
            "未找到可用于继续训练的 LoRA 目录。"
            f"先按当前 task={spec.task_name} 下的 run_id={candidate} 查找，再按路径 {external_lora_dir} 查找，均不存在。"
        )

    def _validate_resume_model_name(
        self,
        spec: TrainSpec,
        resolved_model: ResolvedLlamaFactoryModel,
        adapter_config: PeftConfig,
    ) -> str:
        base_model_name = str(getattr(adapter_config, "base_model_name_or_path", "") or "").strip()
        if not base_model_name:
            raise ValueError("resume LoRA 缺少 base_model_name_or_path，无法恢复基座模型。")
        if spec.model_name is None:
            if resolved_model.model_name_or_path != base_model_name:
                logger.warning(
                    "resume 模式将以已有 LoRA 记录的基座模型为准；当前模型注册值=%s，LoRA 记录值=%s",
                    resolved_model.model_name_or_path,
                    base_model_name,
                )
            return base_model_name
        if spec.model_name != base_model_name:
            raise ValueError(
                "resume 模式下 --model-name 必须与已有 LoRA 头的基座模型一致。"
                f"当前传入={spec.model_name}；"
                f"LoRA 记录的基座模型={base_model_name}"
            )
        logger.warning(
            "resume 模式检测到显式传入 --model-name，且与已有 LoRA 头的基座模型一致；后续将以 LoRA 记录的基座模型为准: %s",
            base_model_name,
        )
        return base_model_name

    def _validate_resume_lora_overrides(self, spec: TrainSpec, adapter_config: PeftConfig) -> None:
        explicit_overrides_raw = spec.extra_args.get("explicit_lora_overrides", {})
        if not isinstance(explicit_overrides_raw, dict) or not explicit_overrides_raw:
            return

        adapter_values = {
            "lora_r": getattr(adapter_config, "r", None),
            "lora_alpha": getattr(adapter_config, "lora_alpha", None),
            "lora_dropout": getattr(adapter_config, "lora_dropout", None),
            "lora_target_modules": tuple(getattr(adapter_config, "target_modules", ()) or ()),
            "lora_bias": getattr(adapter_config, "bias", None),
            "lora_modules_to_save": tuple(getattr(adapter_config, "modules_to_save", ()) or ()),
            "use_rslora": bool(getattr(adapter_config, "use_rslora", False)),
            "use_dora": bool(getattr(adapter_config, "use_dora", False)),
        }

        mismatches: list[str] = []
        matched_parameters: list[str] = []
        for parameter_name, raw_value in explicit_overrides_raw.items():
            if parameter_name not in adapter_values:
                continue
            expected_value = self._normalize_resume_lora_value(parameter_name, adapter_values[parameter_name])
            actual_value = self._normalize_resume_lora_value(parameter_name, raw_value)
            if actual_value != expected_value:
                mismatches.append(f"{parameter_name}: current={actual_value}, adapter={expected_value}")
            else:
                matched_parameters.append(parameter_name)

        if mismatches:
            mismatch_display = "; ".join(mismatches)
            raise ValueError(
                "resume 模式下显式传入的 LoRA 参数必须与已有 LoRA 头一致。"
                f"不一致项: {mismatch_display}"
            )

        if matched_parameters:
            logger.warning(
                "resume 模式检测到显式传入的 LoRA 参数，且与已有 LoRA 配置一致，将沿用已有 LoRA 头继续训练: %s",
                ",".join(matched_parameters),
            )

    def _load_resume_metadata(self, resume_lora_dir: Path) -> tuple[Path, dict[str, Any]]:
        metadata_path = resume_lora_dir.parent / METADATA_FILENAME
        if not metadata_path.exists():
            raise ValueError(
                "resume 模式要求源 LoRA 目录旁存在 TuneBench metadata。"
                f"未找到: {metadata_path}"
            )

        payload = load_metadata(metadata_path)
        if not isinstance(payload, dict):
            raise ValueError(f"resume LoRA metadata 类型无效: {metadata_path}")
        return metadata_path, payload

    def _validate_resume_task_name(self, spec: TrainSpec, metadata_path: Path, payload: dict[str, Any]) -> None:
        task_name = payload.get("task_name")
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError(f"resume LoRA metadata 缺少 task_name: {metadata_path}")

        if task_name != spec.task_name:
            raise ValueError(
                "resume 模式下仅支持继续微调当前 task 的 LoRA 头。"
                f"当前 task={spec.task_name}；"
                f"源 LoRA task={task_name}；"
                f"metadata={metadata_path}"
            )

    def _validate_resume_model_semantics(
        self,
        *,
        metadata_path: Path,
        payload: dict[str, Any],
        resolved_model: ResolvedLlamaFactoryModel,
    ) -> tuple[str, str | None, str | None]:
        source_model_key = resolve_model_key(payload)
        if source_model_key is None:
            raise ValueError(f"resume LoRA metadata 缺少 model_key，无法校验模型语义: {metadata_path}")
        if source_model_key != resolved_model.variant.model_key:
            raise ValueError(
                "resume 模式下 --model-key 必须与源 LoRA 一致。"
                f"当前 model_key={resolved_model.variant.model_key}；"
                f"源 LoRA model_key={source_model_key}；"
                f"metadata={metadata_path}"
            )

        source_reasoning_mode = resolve_reasoning_mode(payload)
        if source_reasoning_mode != resolved_model.reasoning_mode:
            raise ValueError(
                "resume 模式下 reasoning_mode 必须与源 LoRA 一致。"
                f"当前 reasoning_mode={resolved_model.reasoning_mode}；"
                f"源 LoRA reasoning_mode={source_reasoning_mode}；"
                f"metadata={metadata_path}"
            )

        source_template = resolve_template_name(payload)
        if source_template != resolved_model.template:
            raise ValueError(
                "resume 模式下模板必须与源 LoRA 一致。"
                f"当前 template={resolved_model.template}；"
                f"源 LoRA template={source_template}；"
                f"metadata={metadata_path}"
            )

        return source_model_key, source_reasoning_mode, source_template

    def _require_int_value(self, value: object, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"resume LoRA 配置字段 {field_name} 类型无效，期望 int，实际={value!r}")
        return value

    def _require_float_value(self, value: object, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"resume LoRA 配置字段 {field_name} 类型无效，期望 float，实际={value!r}")
        return float(value)

    def _require_str_tuple(self, value: object, field_name: str) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"resume LoRA 配置字段 {field_name} 类型无效，期望字符串列表，实际={value!r}")
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"resume LoRA 配置字段 {field_name} 含有非字符串项: {value!r}")
        return tuple(value)

    def _require_lora_bias(self, value: object) -> Literal["none", "all", "lora_only"]:
        if value in {"none", "all", "lora_only"}:
            return cast(Literal["none", "all", "lora_only"], value)
        raise ValueError(f"resume LoRA 配置字段 bias 无效: {value!r}")

    def _build_resume_lora_spec(self, adapter_config: PeftConfig) -> LoraConfigSpec:
        return LoraConfigSpec(
            r=self._require_int_value(getattr(adapter_config, "r", None), "r"),
            alpha=self._require_int_value(getattr(adapter_config, "lora_alpha", None), "lora_alpha"),
            dropout=self._require_float_value(getattr(adapter_config, "lora_dropout", None), "lora_dropout"),
            target_modules=self._require_str_tuple(getattr(adapter_config, "target_modules", None), "target_modules"),
            bias=self._require_lora_bias(getattr(adapter_config, "bias", None)),
            modules_to_save=self._require_str_tuple(
                getattr(adapter_config, "modules_to_save", None),
                "modules_to_save",
            ),
            use_rslora=bool(getattr(adapter_config, "use_rslora", False)),
            use_dora=bool(getattr(adapter_config, "use_dora", False)),
        )

    def _resolve_resume_lora(
        self,
        spec: TrainSpec,
        resolved_model: ResolvedLlamaFactoryModel,
    ) -> ResolvedResumeLora | None:
        resume_lora_dir, source_kind, source_run_id = self._resolve_resume_lora_dir(spec)
        if resume_lora_dir is None or source_kind is None:
            return None

        adapter_config = PeftConfig.from_pretrained(str(resume_lora_dir))
        metadata_path, payload = self._load_resume_metadata(resume_lora_dir)
        self._validate_resume_task_name(spec, metadata_path, payload)
        source_model_key, source_reasoning_mode, source_template = self._validate_resume_model_semantics(
            metadata_path=metadata_path,
            payload=payload,
            resolved_model=resolved_model,
        )
        base_model_name = self._validate_resume_model_name(spec, resolved_model, adapter_config)
        self._validate_resume_lora_overrides(spec, adapter_config)
        return ResolvedResumeLora(
            lora_dir=resume_lora_dir,
            source_kind=source_kind,
            source_run_id=source_run_id,
            metadata_path=metadata_path,
            source_model_key=source_model_key,
            source_reasoning_mode=source_reasoning_mode,
            source_template=source_template,
            base_model_name_or_path=base_model_name,
            lora=self._build_resume_lora_spec(adapter_config),
        )

    def _resolve_effective_training_model(
        self,
        spec: TrainSpec,
    ) -> tuple[ResolvedLlamaFactoryModel, ResolvedResumeLora | None]:
        resolved_model = self._resolve_model(spec)
        resume_lora = self._resolve_resume_lora(spec, resolved_model)
        if resume_lora is None:
            return resolved_model, None
        return replace(resolved_model, model_name_or_path=resume_lora.base_model_name_or_path), resume_lora

    def _build_effective_spec(
        self,
        spec: TrainSpec,
        resolved_model: ResolvedLlamaFactoryModel,
        resume_lora: ResolvedResumeLora | None,
    ) -> TrainSpec:
        if resume_lora is None:
            return spec

        return replace(spec, model_name=resolved_model.model_name_or_path, lora=resume_lora.lora)

    def _resolve_dataset_bundle(self, spec: TrainSpec) -> StructuredTargetDatasetBundle:
        return load_structured_target_dataset_bundle(
            self.dataset_path_manager,
            spec.task_name,
            spec.dataset_version,
            spec.num_labels,
        )

    def _build_backend_config(
        self,
        *,
        spec: TrainSpec,
        resolved_model: ResolvedLlamaFactoryModel,
        resume_lora: ResolvedResumeLora | None,
        instruction: str,
        runtime_dir: Path,
        train_config_path: Path,
        export_config_path: Path,
        train_command: tuple[str, ...],
        export_command: tuple[str, ...],
    ) -> dict[str, object]:
        custom_instruction = spec.extra_args.get("instruction")
        manual_instruction = custom_instruction.strip() if isinstance(custom_instruction, str) else None
        if resume_lora is not None:
            model_source = "resume_lora"
            lora_payload = {
                "r": resume_lora.lora.r,
                "alpha": resume_lora.lora.alpha,
                "dropout": resume_lora.lora.dropout,
                "target_modules": list(resume_lora.lora.target_modules),
                "modules_to_save": list(resume_lora.lora.modules_to_save),
                "use_rslora": resume_lora.lora.use_rslora,
                "use_dora": resume_lora.lora.use_dora,
            }
        else:
            model_source = "override" if spec.model_name and spec.model_name.strip() else "registry"
            lora_payload = {
                "r": spec.lora.r,
                "alpha": spec.lora.alpha,
                "dropout": spec.lora.dropout,
                "target_modules": list(spec.lora.target_modules),
                "modules_to_save": list(spec.lora.modules_to_save),
                "use_rslora": spec.lora.use_rslora,
                "use_dora": spec.lora.use_dora,
            }
        reasoning_mode_source = "cli" if spec.reasoning_mode is not None else "model_default"
        reasoning_policy = build_train_reasoning_policy(
            resolved_model,
            reasoning_mode_source=reasoning_mode_source,
        )
        payload = {
            "model_key": resolved_model.variant.model_key,
            "display_name": resolved_model.variant.display_name,
            "loader_family": resolved_model.loader_family,
            "supports_multimodal_wrapper": resolved_model.supports_multimodal_wrapper,
            "reasoning_mode": resolved_model.reasoning_mode,
            "reasoning_suffix_style": resolved_model.reasoning_suffix_style,
            "reasoning_suffix_value": reasoning_policy.reasoning_suffix_value,
            "reasoning_policy": reasoning_policy.to_payload(),
            "model_name_or_path": resolved_model.model_name_or_path,
            "model_source": model_source,
            "template": resolved_model.template,
            "instruction_source": "manual" if manual_instruction else "auto",
            "instruction": instruction,
            "runtime_dir": str(runtime_dir),
            "train_config": str(train_config_path),
            "export_config": str(export_config_path),
            "train_command": render_command(train_command),
            "export_command": render_command(export_command),
            "lora": lora_payload,
        }
        if resume_lora is not None:
            payload["resume"] = {
                "mode": "lora_continue",
                "resume_lora_dir": str(resume_lora.lora_dir),
                "source_kind": resume_lora.source_kind,
                "source_run_id": resume_lora.source_run_id,
                "base_model_name_or_path": resume_lora.base_model_name_or_path,
            }
        return payload

    def _rollback_run_id(self, task_name: str, run_id: str) -> Path | None:
        model_layout = self.model_path_manager.build_layout(_LLAMAFACTORY_BACKEND, task_name, run_id)
        if not model_layout.version_dir.exists():
            logger.warning(
                "LlamaFactory 训练失败回滚跳过：run_id 目录不存在: task=%s, run_id=%s, path=%s",
                task_name,
                run_id,
                model_layout.version_dir,
            )
            return None

        shutil.rmtree(model_layout.version_dir)
        logger.warning(
            "LlamaFactory 训练失败已回滚 run_id 目录: task=%s, run_id=%s, path=%s",
            task_name,
            run_id,
            model_layout.version_dir,
        )
        return model_layout.version_dir

    def _build_subprocess_env(self) -> dict[str, str]:
        return os.environ.copy()

    def _run_command(self, command: tuple[str, ...], *, cwd: Path, stage_name: str) -> None:
        logger.info("开始执行 LlamaFactory %s: %s", stage_name, render_command(command))
        completed = subprocess.run(command, cwd=cwd, check=False, env=self._build_subprocess_env())
        if completed.returncode != 0:
            raise RuntimeError(
                f"LlamaFactory {stage_name} 失败，退出码={completed.returncode}；"
                f"命令={render_command(command)}"
            )

    def build_plan(self, spec: TrainSpec) -> RunPlan:
        self._validate_spec(spec)
        run_id = spec.run_id or "<generated-at-runtime>"
        resolved_model, resume_lora = self._resolve_effective_training_model(spec)
        effective_spec = self._build_effective_spec(spec, resolved_model, resume_lora)
        dataset_bundle = self._resolve_dataset_bundle(spec)
        model_layout = self.model_path_manager.build_layout(_LLAMAFACTORY_BACKEND, spec.task_name, run_id)
        runtime_dir = build_runtime_dir(model_layout)
        custom_instruction = spec.extra_args.get("instruction")
        manual_instruction = custom_instruction.strip() if isinstance(custom_instruction, str) else None
        reasoning_mode_source = "CLI 显式指定" if spec.reasoning_mode is not None else "模型默认值"
        model_source_label = (
            "resume-lora"
            if resume_lora is not None
            else "--model-name"
            if spec.model_name and spec.model_name.strip()
            else "registry"
        )
        reasoning_policy = build_train_reasoning_policy(
            resolved_model,
            reasoning_mode_source="cli" if spec.reasoning_mode is not None else "model_default",
        )
        return RunPlan(
            stage="train",
            summary="执行 LlamaFactory 分类训练与导出。",
            inputs=asdict(effective_spec),
            outputs={
                "output_dir": str(model_layout.version_dir),
                "metadata": str(model_layout.metadata_path),
                "lora_dir": str(model_layout.lora_dir),
                "checkpoints_dir": str(model_layout.checkpoints_dir),
                "merged_model_dir": str(model_layout.merged_model_dir),
                "train_metrics_csv": str(model_layout.train_metrics_csv),
                "runtime_dir": str(runtime_dir),
                "train_config": str(runtime_dir / "train.yaml"),
                "export_config": str(runtime_dir / "export.yaml"),
                "dataset_info": str(runtime_dir / "dataset" / "dataset_info.json"),
                "reasoning_policy": reasoning_policy.to_payload(),
            },
            notes=[
                f"模型键={resolved_model.variant.model_key}，显示名称={resolved_model.variant.display_name}。",
                (
                    f"reasoning_mode={resolved_model.reasoning_mode}（来源={reasoning_mode_source}），"
                    f"template={resolved_model.template}。"
                ),
                (
                    f"no_think 控制策略={'模板切换+Qwen3 后缀' if reasoning_policy.reasoning_suffix_value is not None else '仅模板切换'}，"
                    f"reasoning_suffix_style={reasoning_policy.reasoning_suffix_style or 'none'}，"
                    f"reasoning_suffix_value={reasoning_policy.reasoning_suffix_value or 'none'}。"
                ),
                f"基座模型={resolved_model.model_name_or_path}（来源={model_source_label}）。",
                f"instruction 来源={'手动指定' if manual_instruction else '自动构建'}。",
                f"训练样本数={len(dataset_bundle.train_records)}，验证样本数={len(dataset_bundle.validation_records) if dataset_bundle.validation_records is not None else 0}，标签数={len(dataset_bundle.label_to_id)}。",
                f"训练输出先写入 checkpoints 目录，再同步最终 LoRA adapter 到 lora 目录，随后执行 `{_LLAMAFACTORY_CLI_NAME} export`。",
                "训练完成后仅回收 trainer_state 中的 train/eval loss，不再追加本地 validation 结构化评测。",
                (
                    f"继续微调源 LoRA={resume_lora.lora_dir}，来源={resume_lora.source_kind}，"
                    f"源 run_id={resume_lora.source_run_id or 'none'}。"
                )
                if resume_lora is not None
                else "当前训练为新建 LoRA 头，不继承历史 adapter。",
            ],
        )

    def run(self, spec: TrainSpec) -> StageResult:
        self._validate_spec(spec)
        run_id = spec.run_id or self._generate_run_id()
        run_started_at = perf_counter()
        resolved_model, resume_lora = self._resolve_effective_training_model(spec)
        effective_spec = self._build_effective_spec(spec, resolved_model, resume_lora)
        dataset_bundle = self._resolve_dataset_bundle(spec)
        model_layout = self.model_path_manager.ensure_layout(_LLAMAFACTORY_BACKEND, spec.task_name, run_id)

        logger.info(
            "收到 LlamaFactory 训练请求: task=%s, run_id=%s, model_key=%s, reasoning_mode=%s, model_name_or_path=%s, resume_lora=%s",
            spec.task_name,
            run_id,
            resolved_model.variant.model_key,
            resolved_model.reasoning_mode,
            resolved_model.model_name_or_path,
            str(resume_lora.lora_dir) if resume_lora is not None else "none",
        )

        workspace = prepare_training_workspace(
            spec=effective_spec,
            model_layout=model_layout,
            resolved_model=resolved_model,
            dataset_bundle=dataset_bundle,
            resume_lora_dir=resume_lora.lora_dir if resume_lora is not None else None,
        )
        backend_config = self._build_backend_config(
            spec=effective_spec,
            resolved_model=resolved_model,
            resume_lora=resume_lora,
            instruction=workspace.instruction,
            runtime_dir=workspace.runtime_dir,
            train_config_path=workspace.train_config_path,
            export_config_path=workspace.export_config_path,
            train_command=workspace.train_command,
            export_command=workspace.export_command,
        )

        prepared_manifest = build_lifecycle_manifest(
            spec=effective_spec,
            run_id=run_id,
            dataset_bundle=dataset_bundle,
            output_dir=model_layout.version_dir,
            backend_config=backend_config,
            instruction=workspace.instruction,
            train_result=None,
        )
        write_metadata(model_layout.metadata_path, prepared_manifest)
        current_manifest: dict[str, object] = dict(prepared_manifest)

        try:
            train_command_started_at = perf_counter()
            self._run_command(workspace.train_command, cwd=workspace.runtime_dir, stage_name="train")
            train_command_runtime_seconds = perf_counter() - train_command_started_at

            recovered_train_history = recover_trainer_history_metrics(
                artifact_store=self.artifact_store,
                model_layout=model_layout,
            )
            intermediate_train_result = build_train_result(
                model_layout=model_layout,
                total_runtime_seconds=perf_counter() - run_started_at,
                train_command_runtime_seconds=train_command_runtime_seconds,
                export_command_runtime_seconds=0.0,
                train_history_recovered=recovered_train_history,
            )
            trained_manifest = build_lifecycle_manifest(
                spec=effective_spec,
                run_id=run_id,
                dataset_bundle=dataset_bundle,
                output_dir=model_layout.version_dir,
                backend_config=backend_config,
                instruction=workspace.instruction,
                train_result=intermediate_train_result,
                status="trained",
            )
            write_metadata(model_layout.metadata_path, trained_manifest)
            current_manifest = trained_manifest

            sync_trained_adapter_to_lora_dir(model_layout)

            export_command_started_at = perf_counter()
            self._run_command(workspace.export_command, cwd=workspace.runtime_dir, stage_name="export")
            export_command_runtime_seconds = perf_counter() - export_command_started_at

            train_result = build_train_result(
                model_layout=model_layout,
                total_runtime_seconds=perf_counter() - run_started_at,
                train_command_runtime_seconds=train_command_runtime_seconds,
                export_command_runtime_seconds=export_command_runtime_seconds,
                train_history_recovered=recovered_train_history,
            )

            completed_manifest = build_lifecycle_manifest(
                spec=effective_spec,
                run_id=run_id,
                dataset_bundle=dataset_bundle,
                output_dir=model_layout.version_dir,
                backend_config=backend_config,
                instruction=workspace.instruction,
                train_result=train_result,
            )
            write_metadata(model_layout.metadata_path, completed_manifest)
            current_manifest = completed_manifest

            train_metrics_plots = self.artifact_store.export_train_metrics_plots(model_layout)
            for warning_message in train_metrics_plots.warnings:
                logger.warning("训练指标折线图导出提示: %s", warning_message)

            artifacts = {
                "output_dir": model_layout.version_dir,
                "metadata": model_layout.metadata_path,
                "lora_dir": model_layout.lora_dir,
                "checkpoints_dir": model_layout.checkpoints_dir,
                "merged_model_dir": model_layout.merged_model_dir,
                "train_metrics_csv": model_layout.train_metrics_csv,
                "train_file": dataset_bundle.train_file,
                "runtime_dir": workspace.runtime_dir,
                "train_config": workspace.train_config_path,
                "export_config": workspace.export_config_path,
                "dataset_info": workspace.dataset_info_path,
                "commands": workspace.commands_path,
            }
            if resume_lora is not None:
                artifacts["resume_lora_dir"] = resume_lora.lora_dir
            if train_metrics_plots.loss_plot_path is not None:
                artifacts["train_loss_plot_png"] = train_metrics_plots.loss_plot_path
            if train_metrics_plots.eval_metrics_plot_path is not None:
                artifacts["train_eval_metrics_plot_png"] = train_metrics_plots.eval_metrics_plot_path
            if dataset_bundle.validation_file is not None:
                artifacts["validation_file"] = dataset_bundle.validation_file
            export_metadata_path = export_metadata_copy(
                export_dir=spec.export_dir,
                run_id=run_id,
                manifest=completed_manifest,
            )
            if export_metadata_path is not None:
                artifacts["export_metadata"] = export_metadata_path

            message = "LlamaFactory 分类训练与导出已完成，产物和配置已写入模型资产目录。"

            return StageResult(
                stage="train",
                success=True,
                message=message,
                artifacts=artifacts,
                metrics={
                    "train_examples": float(len(dataset_bundle.train_records)),
                    "validation_examples": float(len(dataset_bundle.validation_records))
                    if dataset_bundle.validation_records is not None
                    else 0.0,
                    "num_labels": float(len(dataset_bundle.label_to_id)),
                    **train_result,
                },
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("LlamaFactory 训练失败")
            failed_manifest = dict(current_manifest)
            failed_manifest["status"] = "failed"
            failed_manifest["failure_message"] = str(exc)
            write_metadata(model_layout.metadata_path, failed_manifest)
            rolled_back_path = self._rollback_run_id(spec.task_name, run_id)
            return StageResult(
                stage="train",
                success=False,
                message=(
                    f"LlamaFactory 训练失败: {exc}；已删除模型目录: {rolled_back_path}"
                    if rolled_back_path is not None
                    else f"LlamaFactory 训练失败: {exc}"
                ),
            )