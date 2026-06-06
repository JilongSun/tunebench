"""workflow 层通用契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from tunebench.cli_support import parse_sheet_name
from tunebench.contracts import (
    DatasetSpec,
    EvalSpec,
    LoraConfigSpec,
    ReasoningGenerationSpec,
    StructuredTargetBuildSpec,
    TrainSpec,
)


def utc_now_iso() -> str:
    """返回 UTC ISO 时间字符串。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StageName(StrEnum):
    """workflow 支持的环节名称。"""

    PREPARE_DATASET = "prepare_dataset"
    GENERATE_REASONING = "generate_reasoning"
    BUILD_STRUCTURED_TARGET = "build_structured_target"
    TRAIN_MODEL = "train_model"
    EVALUATE_MODEL = "evaluate_model"


class WorkflowStatus(StrEnum):
    """workflow 级状态。"""

    DRAFT = "draft"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    READY_NEXT = "ready_next"
    FAILED = "failed"
    REJECTED = "rejected"
    COMPLETED = "completed"


class StageStatus(StrEnum):
    """环节运行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


DEFAULT_STAGE_SEQUENCE: tuple[StageName, ...] = (
    StageName.PREPARE_DATASET,
    StageName.TRAIN_MODEL,
    StageName.EVALUATE_MODEL,
)


def normalize_stage_names(values: tuple[StageName | str, ...]) -> tuple[StageName, ...]:
    """规范化环节名称，并保持顺序去重。"""
    normalized: list[StageName] = []
    seen: set[StageName] = set()
    for raw_value in values:
        value = raw_value if isinstance(raw_value, StageName) else StageName(raw_value)
        if value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return tuple(normalized)


def _normalize_string_dict(payload: dict[str, str] | None) -> dict[str, str]:
    if not payload:
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


@dataclass(slots=True, frozen=True)
class WorkflowRuntimeConfig:
    """描述 workflow 运行时环境。"""

    visible_devices: tuple[str, ...] = ()
    env_overrides: dict[str, str] = field(default_factory=dict)
    working_dir: str | None = None

    def build_env(self, base_env: dict[str, str]) -> dict[str, str]:
        """基于当前进程环境构建子进程环境。"""
        from sys import executable
        env = dict(base_env)
        # 将 Python 解释器所在的 bin 目录放到 PATH 第一位，确保子进程优先使用
        # 该环境的原生扩展库（如 sentencepiece），避免被其他 Python 路径干扰
        python_bin_dir = str(Path(executable).parent)
        existing_path = env.get("PATH", "")
        existing_entries = [entry for entry in existing_path.split(":") if entry and entry != python_bin_dir]
        env["PATH"] = python_bin_dir + (":" + ":".join(existing_entries) if existing_entries else "")
        if self.visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self.visible_devices)
        env.update(_normalize_string_dict(self.env_overrides))
        return env

    def to_payload(self) -> dict[str, Any]:
        return {
            "visible_devices": list(self.visible_devices),
            "env_overrides": dict(self.env_overrides),
            "working_dir": self.working_dir,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "WorkflowRuntimeConfig":
        normalized_payload = payload or {}
        visible_devices = normalized_payload.get("visible_devices") or ()
        return cls(
            visible_devices=tuple(str(value) for value in visible_devices),
            env_overrides=_normalize_string_dict(normalized_payload.get("env_overrides")),
            working_dir=normalized_payload.get("working_dir"),
        )


@dataclass(slots=True, frozen=True)
class WorkflowCreateRequest:
    """创建 workflow 所需的公共配置。"""

    task_name: str
    backend: str
    runtime: WorkflowRuntimeConfig = field(default_factory=WorkflowRuntimeConfig)
    run_id: str | None = None
    enabled_stages: tuple[StageName | str, ...] = DEFAULT_STAGE_SEQUENCE
    review_required_stages: tuple[StageName | str, ...] = ()

    def normalized_enabled_stages(self) -> tuple[StageName, ...]:
        return normalize_stage_names(self.enabled_stages)

    def normalized_review_required_stages(self) -> tuple[StageName, ...]:
        enabled = self.normalized_enabled_stages()
        if not self.review_required_stages:
            return enabled
        review_required = normalize_stage_names(self.review_required_stages)
        enabled_set = set(enabled)
        return tuple(stage for stage in review_required if stage in enabled_set)


@dataclass(slots=True, frozen=True)
class LoraWorkflowConfig:
    """workflow 侧的 LoRA 配置。"""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ()
    bias: str = "none"
    modules_to_save: tuple[str, ...] = ()
    use_rslora: bool = False
    use_dora: bool = False

    def to_contract(self) -> LoraConfigSpec:
        normalized_bias = cast(Literal["none", "all", "lora_only"], self.bias)
        return LoraConfigSpec(
            r=self.r,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=self.target_modules,
            bias=normalized_bias,
            modules_to_save=self.modules_to_save,
            use_rslora=self.use_rslora,
            use_dora=self.use_dora,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "bias": self.bias,
            "modules_to_save": list(self.modules_to_save),
            "use_rslora": self.use_rslora,
            "use_dora": self.use_dora,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "LoraWorkflowConfig":
        normalized_payload = payload or {}
        return cls(
            r=int(normalized_payload.get("r", 8)),
            alpha=int(normalized_payload.get("alpha", 16)),
            dropout=float(normalized_payload.get("dropout", 0.1)),
            target_modules=tuple(str(value) for value in normalized_payload.get("target_modules", ())),
            bias=str(normalized_payload.get("bias", "none")),
            modules_to_save=tuple(str(value) for value in normalized_payload.get("modules_to_save", ())),
            use_rslora=bool(normalized_payload.get("use_rslora", False)),
            use_dora=bool(normalized_payload.get("use_dora", False)),
        )


@dataclass(slots=True, frozen=True)
class PrepareDatasetRequest:
    """数据准备环节请求。"""

    input_path: str
    dataset_version: str
    text_key: str
    label_key: str
    output_path: str | None = None
    output_format: str = "jsonl"
    sheet_name: str | int = 0
    validation_ratio: float = 0.0
    split_seed: int = 42
    is_test: bool = False
    allowed_labels: tuple[str, ...] = ()

    def to_spec(self, *, task_name: str) -> DatasetSpec:
        return DatasetSpec(
            task_name=task_name,
            input_path=Path(self.input_path),
            dataset_version=self.dataset_version,
            text_key=self.text_key,
            label_key=self.label_key,
            output_path=(None if self.output_path is None else Path(self.output_path)),
            output_format=self.output_format,
            sheet_name=self.sheet_name,
            validation_ratio=self.validation_ratio,
            split_seed=self.split_seed,
            is_test=self.is_test,
            allowed_labels=self.allowed_labels,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "dataset_version": self.dataset_version,
            "text_key": self.text_key,
            "label_key": self.label_key,
            "output_path": self.output_path,
            "output_format": self.output_format,
            "sheet_name": self.sheet_name,
            "validation_ratio": self.validation_ratio,
            "split_seed": self.split_seed,
            "is_test": self.is_test,
            "allowed_labels": list(self.allowed_labels),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PrepareDatasetRequest":
        return cls(
            input_path=str(payload["input_path"]),
            dataset_version=str(payload["dataset_version"]),
            text_key=str(payload["text_key"]),
            label_key=str(payload["label_key"]),
            output_path=(None if payload.get("output_path") is None else str(payload.get("output_path"))),
            output_format=str(payload.get("output_format", "jsonl")),
            sheet_name=parse_sheet_name(str(payload.get("sheet_name", "0"))),
            validation_ratio=float(payload.get("validation_ratio", 0.0)),
            split_seed=int(payload.get("split_seed", 42)),
            is_test=bool(payload.get("is_test", False)),
            allowed_labels=tuple(str(value) for value in payload.get("allowed_labels", ())),
        )


@dataclass(slots=True, frozen=True)
class GenerateReasoningRequest:
    """reasoning 生成环节请求。"""

    source_dataset_version: str
    target_dataset_version: str
    teacher_model: str
    endpoint_url: str
    label_profile: str = "l1_5class"
    prompt_version: str = "reasoning_v1"
    api_key_env_var: str = "TUNEBENCH_REASONING_API_KEY"
    max_concurrency: int = 5
    request_timeout_seconds: float = 60.0
    max_attempts: int = 2
    enable_model_verify: bool = False
    resume: bool = False
    sample_limit: int | None = None
    splits: tuple[str, ...] = ("train", "validation")

    def to_spec(self, *, task_name: str) -> ReasoningGenerationSpec:
        return ReasoningGenerationSpec(
            task_name=task_name,
            source_dataset_version=self.source_dataset_version,
            target_dataset_version=self.target_dataset_version,
            teacher_model=self.teacher_model,
            endpoint_url=self.endpoint_url,
            label_profile=self.label_profile,
            prompt_version=self.prompt_version,
            api_key_env_var=self.api_key_env_var,
            max_concurrency=self.max_concurrency,
            request_timeout_seconds=self.request_timeout_seconds,
            max_attempts=self.max_attempts,
            enable_model_verify=self.enable_model_verify,
            resume=self.resume,
            sample_limit=self.sample_limit,
            splits=self.splits,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_dataset_version": self.source_dataset_version,
            "target_dataset_version": self.target_dataset_version,
            "teacher_model": self.teacher_model,
            "endpoint_url": self.endpoint_url,
            "label_profile": self.label_profile,
            "prompt_version": self.prompt_version,
            "api_key_env_var": self.api_key_env_var,
            "max_concurrency": self.max_concurrency,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_attempts": self.max_attempts,
            "enable_model_verify": self.enable_model_verify,
            "resume": self.resume,
            "sample_limit": self.sample_limit,
            "splits": list(self.splits),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GenerateReasoningRequest":
        return cls(
            source_dataset_version=str(payload["source_dataset_version"]),
            target_dataset_version=str(payload["target_dataset_version"]),
            teacher_model=str(payload["teacher_model"]),
            endpoint_url=str(payload["endpoint_url"]),
            label_profile=str(payload.get("label_profile", "l1_5class")),
            prompt_version=str(payload.get("prompt_version", "reasoning_v1")),
            api_key_env_var=str(payload.get("api_key_env_var", "TUNEBENCH_REASONING_API_KEY")),
            max_concurrency=int(payload.get("max_concurrency", 5)),
            request_timeout_seconds=float(payload.get("request_timeout_seconds", 60.0)),
            max_attempts=int(payload.get("max_attempts", 2)),
            enable_model_verify=bool(payload.get("enable_model_verify", False)),
            resume=bool(payload.get("resume", False)),
            sample_limit=_optional_int(payload.get("sample_limit")),
            splits=tuple(str(value) for value in payload.get("splits", ("train", "validation"))),
        )


@dataclass(slots=True, frozen=True)
class BuildStructuredTargetRequest:
    """structured target 构建环节请求。"""

    source_dataset_version: str
    target_dataset_version: str
    confidence: float = 0.9
    splits: tuple[str, ...] = ("train", "validation")

    def to_spec(self, *, task_name: str) -> StructuredTargetBuildSpec:
        return StructuredTargetBuildSpec(
            task_name=task_name,
            source_dataset_version=self.source_dataset_version,
            target_dataset_version=self.target_dataset_version,
            confidence=self.confidence,
            splits=self.splits,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_dataset_version": self.source_dataset_version,
            "target_dataset_version": self.target_dataset_version,
            "confidence": self.confidence,
            "splits": list(self.splits),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BuildStructuredTargetRequest":
        return cls(
            source_dataset_version=str(payload["source_dataset_version"]),
            target_dataset_version=str(payload["target_dataset_version"]),
            confidence=float(payload.get("confidence", 0.9)),
            splits=tuple(str(value) for value in payload.get("splits", ("train", "validation"))),
        )


@dataclass(slots=True, frozen=True)
class TrainModelRequest:
    """训练环节请求。"""

    dataset_version: str
    model_name: str | None = None
    model_key: str | None = None
    instruction: str | None = None
    reasoning_mode: str | None = None
    resume_lora: str | None = None
    export_dir: str | None = None
    num_labels: int | None = None
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_train_epochs: int = 3
    max_sequence_length: int = 256
    warmup_ratio: float = 0.0
    seed: int = 42
    lora: LoraWorkflowConfig = field(default_factory=LoraWorkflowConfig)

    def to_spec(self, *, task_name: str, backend: str, run_id: str) -> TrainSpec:
        normalized_instruction = self.instruction.strip() if isinstance(self.instruction, str) else self.instruction
        normalized_reasoning_mode = cast(Literal["think", "no_think"] | None, self.reasoning_mode)
        if backend == "bert" and self.resume_lora is None and not self.model_name:
            raise ValueError("BERT 训练在 start 模式下必须提供 model_name。")
        if backend == "llamafactory" and not self.model_key:
            raise ValueError("llamafactory 训练必须提供 model_key。")
        if backend != "llamafactory" and normalized_instruction is not None:
            raise ValueError("instruction 仅允许在 llamafactory 训练中使用。")
        if backend == "llamafactory" and normalized_instruction is not None and not normalized_instruction:
            raise ValueError("instruction 不能为空字符串。")
        explicit_lora_overrides = {
            "lora_r": self.lora.r,
            "lora_alpha": self.lora.alpha,
            "lora_dropout": self.lora.dropout,
            "lora_target_modules": self.lora.target_modules,
            "lora_bias": self.lora.bias,
            "lora_modules_to_save": self.lora.modules_to_save,
            "use_rslora": self.lora.use_rslora,
            "use_dora": self.lora.use_dora,
        }
        extra_args_payload: dict[str, Any] = {"explicit_lora_overrides": explicit_lora_overrides}
        if normalized_instruction is not None:
            extra_args_payload["instruction"] = normalized_instruction
        return TrainSpec(
            backend=backend,
            task_name=task_name,
            model_name=self.model_name,
            dataset_version=self.dataset_version,
            model_key=self.model_key,
            reasoning_mode=normalized_reasoning_mode,
            resume_lora=self.resume_lora,
            run_id=run_id,
            export_dir=(None if self.export_dir is None else Path(self.export_dir)),
            num_labels=self.num_labels,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            num_train_epochs=self.num_train_epochs,
            max_sequence_length=self.max_sequence_length,
            warmup_ratio=self.warmup_ratio,
            seed=self.seed,
            lora=self.lora.to_contract(),
            extra_args=extra_args_payload,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "model_name": self.model_name,
            "model_key": self.model_key,
            "instruction": self.instruction,
            "reasoning_mode": self.reasoning_mode,
            "resume_lora": self.resume_lora,
            "export_dir": self.export_dir,
            "num_labels": self.num_labels,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_train_epochs": self.num_train_epochs,
            "max_sequence_length": self.max_sequence_length,
            "warmup_ratio": self.warmup_ratio,
            "seed": self.seed,
            "lora": self.lora.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TrainModelRequest":
        return cls(
            dataset_version=str(payload["dataset_version"]),
            model_name=(None if payload.get("model_name") is None else str(payload.get("model_name"))),
            model_key=(None if payload.get("model_key") is None else str(payload.get("model_key"))),
            instruction=(None if payload.get("instruction") is None else str(payload.get("instruction"))),
            reasoning_mode=(None if payload.get("reasoning_mode") is None else str(payload.get("reasoning_mode"))),
            resume_lora=(None if payload.get("resume_lora") is None else str(payload.get("resume_lora"))),
            export_dir=(None if payload.get("export_dir") is None else str(payload.get("export_dir"))),
            num_labels=_optional_int(payload.get("num_labels")),
            learning_rate=float(payload.get("learning_rate", 2e-5)),
            batch_size=int(payload.get("batch_size", 8)),
            num_train_epochs=int(payload.get("num_train_epochs", 3)),
            max_sequence_length=int(payload.get("max_sequence_length", 256)),
            warmup_ratio=float(payload.get("warmup_ratio", 0.0)),
            seed=int(payload.get("seed", 42)),
            lora=LoraWorkflowConfig.from_payload(payload.get("lora")),
        )


@dataclass(slots=True, frozen=True)
class EvaluateModelRequest:
    """评测环节请求。"""

    dataset_version: str
    artifact_type: str = "merged"
    batch_size: int = 8
    max_sequence_length: int | None = None
    max_new_tokens: int | None = None
    prompt_engine: str | None = None
    enable_thinking: bool | None = None
    export_xlsx: bool = True

    def to_spec(self, *, task_name: str, backend: str, run_id: str) -> EvalSpec:
        normalized_prompt_engine = cast(Literal["llamafactory", "native"] | None, self.prompt_engine)
        if backend != "llamafactory" and self.max_new_tokens is not None:
            raise ValueError("max_new_tokens 仅允许在 llamafactory 评测中使用。")
        if backend != "llamafactory" and normalized_prompt_engine is not None:
            raise ValueError("prompt_engine 仅允许在 llamafactory 评测中使用。")
        if backend != "llamafactory" and self.enable_thinking is not None:
            raise ValueError("enable_thinking 仅允许在 llamafactory 评测中使用。")

        effective_prompt_engine = normalized_prompt_engine
        if backend == "llamafactory":
            if effective_prompt_engine is None and self.enable_thinking is not None:
                effective_prompt_engine = "native"
            if effective_prompt_engine == "llamafactory" and self.enable_thinking is not None:
                raise ValueError("enable_thinking 仅允许在 native prompt-engine 中使用。")

        return EvalSpec(
            backend=backend,
            task_name=task_name,
            run_id=run_id,
            dataset_version=self.dataset_version,
            artifact_type=self.artifact_type,
            batch_size=self.batch_size,
            max_sequence_length=(256 if backend == "bert" and self.max_sequence_length is None else self.max_sequence_length),
            max_new_tokens=self.max_new_tokens,
            prompt_engine=effective_prompt_engine,
            enable_thinking=self.enable_thinking,
            export_xlsx=self.export_xlsx,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "artifact_type": self.artifact_type,
            "batch_size": self.batch_size,
            "max_sequence_length": self.max_sequence_length,
            "max_new_tokens": self.max_new_tokens,
            "prompt_engine": self.prompt_engine,
            "enable_thinking": self.enable_thinking,
            "export_xlsx": self.export_xlsx,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EvaluateModelRequest":
        return cls(
            dataset_version=str(payload["dataset_version"]),
            artifact_type=str(payload.get("artifact_type", "merged")),
            batch_size=int(payload.get("batch_size", 8)),
            max_sequence_length=_optional_int(payload.get("max_sequence_length")),
            max_new_tokens=_optional_int(payload.get("max_new_tokens")),
            prompt_engine=(None if payload.get("prompt_engine") is None else str(payload.get("prompt_engine"))),
            enable_thinking=payload.get("enable_thinking"),
            export_xlsx=bool(payload.get("export_xlsx", True)),
        )


@dataclass(slots=True, frozen=True)
class WorkflowRecord:
    """workflow 主记录。"""

    workflow_id: str
    task_name: str
    backend: str
    run_id: str
    runtime: WorkflowRuntimeConfig
    enabled_stages: tuple[StageName, ...]
    review_required_stages: tuple[StageName, ...]
    status: WorkflowStatus = WorkflowStatus.DRAFT
    current_stage: StageName | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    version: int = 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "task_name": self.task_name,
            "backend": self.backend,
            "run_id": self.run_id,
            "runtime": self.runtime.to_payload(),
            "enabled_stages": [stage.value for stage in self.enabled_stages],
            "review_required_stages": [stage.value for stage in self.review_required_stages],
            "status": self.status.value,
            "current_stage": (None if self.current_stage is None else self.current_stage.value),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "WorkflowRecord":
        current_stage_raw = payload.get("current_stage")
        return cls(
            workflow_id=str(payload["workflow_id"]),
            task_name=str(payload["task_name"]),
            backend=str(payload["backend"]),
            run_id=str(payload["run_id"]),
            runtime=WorkflowRuntimeConfig.from_payload(payload.get("runtime")),
            enabled_stages=normalize_stage_names(tuple(payload.get("enabled_stages", ()))),
            review_required_stages=normalize_stage_names(tuple(payload.get("review_required_stages", ()))),
            status=WorkflowStatus(payload.get("status", WorkflowStatus.DRAFT.value)),
            current_stage=(None if current_stage_raw is None else StageName(current_stage_raw)),
            created_at=str(payload.get("created_at", utc_now_iso())),
            updated_at=str(payload.get("updated_at", utc_now_iso())),
            version=int(payload.get("version", 1)),
        )


@dataclass(slots=True, frozen=True)
class StageRunRecord:
    """单次环节运行记录。"""

    stage_run_id: str
    workflow_id: str
    stage_name: StageName
    status: StageStatus
    request_payload: dict[str, Any]
    log_path: str
    request_path: str
    result_path: str
    requires_review: bool
    plan_payload: dict[str, Any] | None = None
    result_payload: dict[str, Any] | None = None
    pid: int | None = None
    exit_code: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    version: int = 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "stage_run_id": self.stage_run_id,
            "workflow_id": self.workflow_id,
            "stage_name": self.stage_name.value,
            "status": self.status.value,
            "request_payload": self.request_payload,
            "plan_payload": self.plan_payload,
            "result_payload": self.result_payload,
            "log_path": self.log_path,
            "request_path": self.request_path,
            "result_path": self.result_path,
            "requires_review": self.requires_review,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "StageRunRecord":
        return cls(
            stage_run_id=str(payload["stage_run_id"]),
            workflow_id=str(payload["workflow_id"]),
            stage_name=StageName(payload["stage_name"]),
            status=StageStatus(payload["status"]),
            request_payload=dict(payload.get("request_payload") or {}),
            plan_payload=payload.get("plan_payload"),
            result_payload=payload.get("result_payload"),
            log_path=str(payload["log_path"]),
            request_path=str(payload["request_path"]),
            result_path=str(payload["result_path"]),
            requires_review=bool(payload.get("requires_review", True)),
            pid=_optional_int(payload.get("pid")),
            exit_code=_optional_int(payload.get("exit_code")),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            created_at=str(payload.get("created_at", utc_now_iso())),
            updated_at=str(payload.get("updated_at", utc_now_iso())),
            version=int(payload.get("version", 1)),
        )


@dataclass(slots=True, frozen=True)
class WorkflowEventRecord:
    """workflow 事件记录。"""

    event_id: str
    workflow_id: str
    event_type: str
    payload: dict[str, Any]
    stage_run_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "workflow_id": self.workflow_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "stage_run_id": self.stage_run_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "WorkflowEventRecord":
        return cls(
            event_id=str(payload["event_id"]),
            workflow_id=str(payload["workflow_id"]),
            event_type=str(payload["event_type"]),
            payload=dict(payload.get("payload") or {}),
            stage_run_id=(None if payload.get("stage_run_id") is None else str(payload.get("stage_run_id"))),
            created_at=str(payload.get("created_at", utc_now_iso())),
        )


@dataclass(slots=True, frozen=True)
class WorkflowStagePlan:
    """描述 workflow 中单个环节的执行计划。"""

    stage_name: StageName
    depends_on: tuple[StageName, ...] = ()
    requires_review: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name.value,
            "depends_on": [stage.value for stage in self.depends_on],
            "requires_review": self.requires_review,
        }


@dataclass(slots=True, frozen=True)
class WorkflowPreview:
    """描述 workflow 预览结果。"""

    task_name: str
    backend: str
    run_id: str
    stages: tuple[WorkflowStagePlan, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "backend": self.backend,
            "run_id": self.run_id,
            "stages": [stage.to_payload() for stage in self.stages],
        }


@dataclass(slots=True, frozen=True)
class WorkflowSnapshot:
    """描述 workflow 查询视图。"""

    workflow: WorkflowRecord
    stage_runs: tuple[StageRunRecord, ...]
    events: tuple[WorkflowEventRecord, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow.to_payload(),
            "stage_runs": [record.to_payload() for record in self.stage_runs],
            "events": [event.to_payload() for event in self.events],
        }
