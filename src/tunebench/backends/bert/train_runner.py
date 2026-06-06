"""BERT 分类训练后端。"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from tunebench.artifacts import (
	CHECKPOINTS_DIRNAME,
    DatasetPathManager,
    EvalArtifactStore,
    FileSystemEvalArtifactStore,
	LORA_DIRNAME,
	MERGED_MODEL_DIRNAME,
	METADATA_FILENAME,
    ModelPathManager,
	TRAIN_METRICS_FILENAME,
    build_classification_train_manifest,
    get_dataset_path_manager,
    get_model_path_manager,
)
from tunebench.classification import (
    ClassificationTrainEvalCallback,
    compute_classification_metrics_bundle,
    flatten_label_metrics,
    load_training_dataset_bundle,
)
from tunebench.contracts import RunPlan, StageResult, TrainSpec
from tunebench.util import get_logger


_VISIBLE_TRAIN_DEVICE_INDEX = 0
_GPU_OOM_WARNING_RATIO_THRESHOLD = 0.80
_BERT_BACKEND = "bert"

logger = get_logger("backends.bert.train_runner")


class LoraTargetModuleResolver:
    """统一管理不同模型的 LoRA 注入模块配置。"""

    _TARGET_MODULES_BY_MODEL_TYPE: dict[str, tuple[str, ...]] = {
        "modernbert": ("Wqkv", "Wo"),
        "bert": ("query", "value"),
        "roberta": ("query", "value"),
        "ernie": ("query", "value"),
    }
    _TARGET_MODULES_BY_NAME_KEYWORD: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("modernbert", ("Wqkv", "Wo")),
        ("bert", ("query", "value")),
        ("roberta", ("query", "value")),
        ("deberta", ("query", "value")),
        ("ernie", ("query", "value")),
    )
    _DEFAULT_TARGET_MODULES: tuple[str, ...] = ("q_proj", "v_proj")

    def resolve(self, model_name: str) -> list[str]:
        """优先按 model_type 解析，失败时回退到名称关键字。"""
        model_type = self._resolve_model_type(model_name)
        if model_type is not None:
            target_modules = self._TARGET_MODULES_BY_MODEL_TYPE.get(model_type)
            if target_modules is not None:
                return list(target_modules)

        lowered = model_name.lower()
        for keyword, target_modules in self._TARGET_MODULES_BY_NAME_KEYWORD:
            if keyword in lowered:
                return list(target_modules)
        return list(self._DEFAULT_TARGET_MODULES)

    def _resolve_model_type(self, model_name: str) -> str | None:
        """从模型配置中读取 model_type，避免仅凭名称猜测。"""
        try:
            config = AutoConfig.from_pretrained(model_name)
        except Exception as exc:
            logger.debug("读取模型配置失败，回退到名称关键字匹配: model_name=%s, error=%s", model_name, exc)
            return None

        model_type = getattr(config, "model_type", None)
        if not isinstance(model_type, str) or not model_type:
            return None
        return model_type.lower()


class BertClassificationTrainRunner:
    """负责封装当前 BERT 分类训练流程。"""

    def __init__(
        self,
        dataset_path_manager: DatasetPathManager | None = None,
        model_path_manager: ModelPathManager | None = None,
        lora_target_module_resolver: LoraTargetModuleResolver | None = None,
        eval_artifact_store: EvalArtifactStore | None = None,
    ) -> None:
        self.dataset_path_manager = dataset_path_manager or get_dataset_path_manager()
        self.model_path_manager = model_path_manager or get_model_path_manager()
        self.lora_target_module_resolver = lora_target_module_resolver or LoraTargetModuleResolver()
        self.eval_artifact_store = eval_artifact_store or FileSystemEvalArtifactStore()

    def _generate_run_id(self) -> str:
        return datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")

    def _resolve_device(self) -> str:
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境未检测到 CUDA，训练必须使用 GPU。")

        device_count = torch.cuda.device_count()
        if device_count != 1:
            raise ValueError(
                "训练进程必须且只能暴露 1 张 GPU；"
                "请在启动进程前通过 CUDA_VISIBLE_DEVICES 绑定物理 4-7 号卡。"
            )

        gpu_index = _VISIBLE_TRAIN_DEVICE_INDEX
        device_name = torch.cuda.get_device_name(gpu_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_index)
        used_bytes = total_bytes - free_bytes
        used_ratio = (used_bytes / total_bytes) if total_bytes > 0 else 1.0
        if used_ratio > _GPU_OOM_WARNING_RATIO_THRESHOLD:
            used_gb = used_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            logger.warning(
                "当前可见训练卡显存占用较高，可能触发 OOM: gpu_index=%s, device=%s, used=%.2fGB/%.2fGB, used_ratio=%.1f%%",
                gpu_index,
                device_name,
                used_gb,
                total_gb,
                used_ratio * 100,
            )

        torch.cuda.set_device(gpu_index)
        return f"cuda:{gpu_index}"

    def _require_model_name(self, spec: TrainSpec) -> str:
        if spec.model_name is None or not spec.model_name.strip():
            raise ValueError("当前训练路径需要显式提供 model_name。")
        return spec.model_name

    def _resolve_lora_target_modules(self, spec: TrainSpec) -> list[str]:
        if spec.lora.target_modules:
            return list(spec.lora.target_modules)
        return self.lora_target_module_resolver.resolve(self._require_model_name(spec))

    def _is_path_like(self, value: str) -> bool:
        return "/" in value or "\\" in value

    def _resolve_tokenizer_source(self, model_dir: Path | None, model_name: str) -> str:
        if model_dir is not None and (model_dir / "tokenizer_config.json").exists():
            return str(model_dir)
        return model_name

    def _resolve_resume_lora_dir(self, spec: TrainSpec) -> Path | None:
        if spec.resume_lora is None:
            return None

        candidate = spec.resume_lora.strip()
        if not candidate:
            raise ValueError("resume_lora 不能为空字符串。")

        if not self._is_path_like(candidate):
            internal_lora_dir = self.model_path_manager.get_lora_dir(_BERT_BACKEND, spec.task_name, candidate)
            if internal_lora_dir.exists():
                logger.info("resume_lora 已按项目内 run_id 解析: run_id=%s, path=%s", candidate, internal_lora_dir)
                return internal_lora_dir

        external_lora_dir = Path(candidate).expanduser()
        if external_lora_dir.exists():
            logger.info("resume_lora 已按外部路径解析: input=%s, path=%s", candidate, external_lora_dir)
            return external_lora_dir

        raise FileNotFoundError(
            "未找到可用于继续训练的 LoRA 目录。"
            f"先按当前 task={spec.task_name} 下的 run_id={candidate} 查找，再按路径 {external_lora_dir} 查找，均不存在。"
        )

    def _validate_resume_model_name(self, spec: TrainSpec, adapter_config: PeftConfig) -> str:
        base_model_name = str(getattr(adapter_config, "base_model_name_or_path", "") or "").strip()
        if not base_model_name:
            raise ValueError("resume LoRA 缺少 base_model_name_or_path，无法恢复基座模型。")
        if spec.model_name is None:
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

    def _normalize_resume_lora_value(self, parameter_name: str, value: Any) -> Any:
        if parameter_name in {"lora_target_modules", "lora_modules_to_save"}:
            if not value:
                return ()
            return tuple(value)
        if parameter_name in {"use_rslora", "use_dora"}:
            return bool(value)
        return value

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

    def _build_model_and_tokenizer(
        self,
        spec: TrainSpec,
        label_to_id: dict[str, int],
    ) -> tuple[Any, PreTrainedTokenizerBase, dict[str, Any], Path | None]:
        resume_lora_dir = self._resolve_resume_lora_dir(spec)

        if resume_lora_dir is None:
            model_name = self._require_model_name(spec)
            tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_name, use_fast=True))
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(label_to_id),
                id2label={index: label for label, index in label_to_id.items()},
                label2id=label_to_id,
            )
            target_modules = self._resolve_lora_target_modules(spec)
            modules_to_save = list(spec.lora.modules_to_save) if spec.lora.modules_to_save else None
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=spec.lora.r,
                lora_alpha=spec.lora.alpha,
                lora_dropout=spec.lora.dropout,
                target_modules=target_modules,
                bias=spec.lora.bias,
                modules_to_save=modules_to_save,
                use_rslora=spec.lora.use_rslora,
                use_dora=spec.lora.use_dora,
            )
            model = get_peft_model(base_model, lora_config)
            return (
                model,
                tokenizer,
                {
                    "r": lora_config.r,
                    "lora_alpha": lora_config.lora_alpha,
                    "lora_dropout": lora_config.lora_dropout,
                    "target_modules": list(lora_config.target_modules or []),
                    "bias": lora_config.bias,
                    "modules_to_save": list(lora_config.modules_to_save or []),
                    "use_rslora": lora_config.use_rslora,
                    "use_dora": lora_config.use_dora,
                    "resume_lora_dir": None,
                },
                None,
            )

        adapter_config = PeftConfig.from_pretrained(str(resume_lora_dir))
        base_model_name = self._validate_resume_model_name(spec, adapter_config)
        self._validate_resume_lora_overrides(spec, adapter_config)
        tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(
                self._resolve_tokenizer_source(resume_lora_dir, base_model_name),
                use_fast=True,
            ),
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(label_to_id),
            id2label={index: label for label, index in label_to_id.items()},
            label2id=label_to_id,
        )
        model = PeftModel.from_pretrained(base_model, str(resume_lora_dir), is_trainable=True)
        return (
            model,
            tokenizer,
            {
                "r": getattr(adapter_config, "r", None),
                "lora_alpha": getattr(adapter_config, "lora_alpha", None),
                "lora_dropout": getattr(adapter_config, "lora_dropout", None),
                "target_modules": list(getattr(adapter_config, "target_modules", []) or []),
                "bias": getattr(adapter_config, "bias", None),
                "modules_to_save": list(getattr(adapter_config, "modules_to_save", []) or []),
                "use_rslora": bool(getattr(adapter_config, "use_rslora", False)),
                "use_dora": bool(getattr(adapter_config, "use_dora", False)),
                "resume_lora_dir": str(resume_lora_dir),
            },
            resume_lora_dir,
        )

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict[str, float]:
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids
        if isinstance(logits, tuple):
            logits = logits[0]
        metrics, label_metrics = compute_classification_metrics_bundle(np.asarray(logits), np.asarray(labels))
        metrics.update(flatten_label_metrics(label_metrics))
        return metrics

    def _build_dataset(self, records: list[dict[str, str]], label_to_id: dict[str, int]) -> Dataset:
        return Dataset.from_list(
            [{"text": record["text"], "label": label_to_id[record["label"]]} for record in records]
        )

    def _tokenize_dataset(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_sequence_length: int,
    ) -> Dataset:
        return dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_sequence_length),
            batched=True,
            desc="tokenizing",
        )

    def _build_training_arguments(
        self,
        spec: TrainSpec,
        output_dir: Path,
        has_validation: bool,
    ) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=spec.learning_rate,
            per_device_train_batch_size=spec.batch_size,
            per_device_eval_batch_size=spec.batch_size,
            num_train_epochs=spec.num_train_epochs,
            warmup_ratio=spec.warmup_ratio,
            seed=spec.seed,
            eval_strategy="epoch" if has_validation else "no",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=has_validation,
            metric_for_best_model="f1_macro" if has_validation else None,
            greater_is_better=True if has_validation else None,
            report_to=[],
            remove_unused_columns=True,
        )

    def _save_initial_model_manifest(self, output_dir: Path, spec: TrainSpec) -> Path:
        manifest_path = output_dir / "base_model.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "model_name": spec.model_name,
                    "initialized_at": datetime.now().isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return manifest_path

    def _save_merged_model(
        self,
        model: Any,
        output_dir: Path,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Path:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        return output_dir

    def _build_metadata(
        self,
        spec: TrainSpec,
        run_id: str,
        train_file: Path,
        validation_file: Path | None,
        train_records: list[dict[str, str]],
        validation_records: list[dict[str, str]] | None,
        label_to_id: dict[str, int],
        output_dir: Path,
        device: str,
        lora_config: dict[str, Any],
        train_result: dict[str, float] | None = None,
        eval_result: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return build_classification_train_manifest(
            spec=spec,
            run_id=run_id,
            train_file=train_file,
            validation_file=validation_file,
            train_records=train_records,
            validation_records=validation_records,
            label_to_id=label_to_id,
            output_dir=output_dir,
            device=device,
            backend_config={"lora": lora_config},
            train_result=train_result,
            eval_result=eval_result,
        )

    def _write_metadata(self, output_path: Path, payload: dict[str, Any]) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def _rollback_run_id(self, task_name: str, run_id: str) -> Path | None:
        model_layout = self.model_path_manager.build_layout(_BERT_BACKEND, task_name, run_id)
        if not model_layout.version_dir.exists():
            logger.warning(
                "训练失败回滚跳过：run_id 目录不存在: task=%s, run_id=%s, path=%s",
                task_name,
                run_id,
                model_layout.version_dir,
            )
            return None

        shutil.rmtree(model_layout.version_dir)
        logger.warning(
            "训练失败已回滚 run_id 目录: task=%s, run_id=%s, path=%s",
            task_name,
            run_id,
            model_layout.version_dir,
        )
        return model_layout.version_dir

    def build_plan(self, spec: TrainSpec) -> RunPlan:
        run_id = spec.run_id or "<generated-at-runtime>"
        dataset_layout = self.dataset_path_manager.build_layout(spec.task_name, spec.dataset_version)
        model_layout = self.model_path_manager.build_layout(_BERT_BACKEND, spec.task_name, run_id)
        return RunPlan(
            stage="train",
            summary="执行 BERT 微调训练。",
            inputs=asdict(spec),
            outputs={
                "train_file": str(dataset_layout.final_dir / "train.jsonl|json"),
                "validation_file": str(dataset_layout.final_dir / "validation.jsonl|json"),
                "output_dir": str(model_layout.version_dir),
                "metadata": str(model_layout.metadata_path),
                "train_metrics_csv": str(model_layout.train_metrics_csv),
                "validation_label_metrics_csv": str(model_layout.validation_label_metrics_csv),
                "train_loss_plot_png": str(model_layout.train_loss_plot_png),
                "train_eval_metrics_plot_png": str(model_layout.train_eval_metrics_plot_png),
            },
            notes=[
                "训练输入固定来自 assets/data/.../final 下的标准化数据。",
                "训练固定读取 train，若 validation 文件存在则自动启用训练期验证。",
                "默认使用 Transformers + PEFT LoRA 进行序列分类训练。",
                "训练必须使用 CUDA，且当前进程必须仅暴露 1 张 GPU；物理 4-7 号卡需由外层启动器通过 CUDA_VISIBLE_DEVICES 绑定。",
                f"训练产物会分别写入 {CHECKPOINTS_DIRNAME}、{LORA_DIRNAME}、{MERGED_MODEL_DIRNAME} 与 {METADATA_FILENAME}。",
                f"训练完成后会读取 {TRAIN_METRICS_FILENAME}，并将 loss 与其他验证指标分别导出为两张折线图。",
            ],
        )

    def run(self, spec: TrainSpec) -> StageResult:
        run_id = spec.run_id or self._generate_run_id()
        try:
            logger.info(
                "开始训练: task=%s, run_id=%s, dataset_version=%s, model=%s",
                spec.task_name,
                run_id,
                spec.dataset_version,
                spec.model_name,
            )
            dataset_bundle = load_training_dataset_bundle(
                self.dataset_path_manager,
                spec.task_name,
                spec.dataset_version,
                num_labels=spec.num_labels,
            )
            train_file = dataset_bundle.train_file
            validation_file = dataset_bundle.validation_file
            train_records = dataset_bundle.train_records
            validation_records = dataset_bundle.validation_records
            label_to_id = dataset_bundle.label_to_id
            model_layout = self.model_path_manager.ensure_layout(_BERT_BACKEND, spec.task_name, run_id)
            device = self._resolve_device()
            set_seed(spec.seed)

            model, tokenizer, lora_config_payload, resume_lora_dir = self._build_model_and_tokenizer(spec, label_to_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

            train_dataset = self._build_dataset(train_records, label_to_id)
            eval_dataset = self._build_dataset(validation_records, label_to_id) if validation_records is not None else None
            train_dataset = self._tokenize_dataset(train_dataset, tokenizer, spec.max_sequence_length)
            if eval_dataset is not None:
                eval_dataset = self._tokenize_dataset(eval_dataset, tokenizer, spec.max_sequence_length)

            training_arguments = self._build_training_arguments(
                spec,
                model_layout.checkpoints_dir,
                has_validation=eval_dataset is not None,
            )

            eval_callback = ClassificationTrainEvalCallback(
                model_layout,
                run_id=run_id,
                id_to_label=dataset_bundle.id_to_label,
                artifact_store=self.eval_artifact_store,
            )
            trainer = Trainer(
                model=model,
                args=training_arguments,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=self._compute_metrics if eval_dataset is not None else None,
                callbacks=[eval_callback],
            )

            logger.info("开始执行 Trainer.train")
            train_output = trainer.train()
            if eval_dataset is not None:
                eval_callback.suppress_next_evaluate_artifact()
                eval_metrics = trainer.evaluate()
                eval_callback.append_final_evaluate_summary(eval_metrics)
            else:
                eval_metrics = None

            trainer.save_model(str(model_layout.lora_dir))
            tokenizer.save_pretrained(model_layout.lora_dir)
            self._save_merged_model(model, model_layout.merged_model_dir, tokenizer)
            initial_model_manifest = self._save_initial_model_manifest(model_layout.initial_model_dir, spec)

            metadata = self._build_metadata(
                spec=spec,
                run_id=run_id,
                train_file=train_file,
                validation_file=validation_file,
                train_records=train_records,
                validation_records=validation_records,
                label_to_id=label_to_id,
                output_dir=model_layout.version_dir,
                device=device,
                lora_config=lora_config_payload,
                train_result={key: float(value) for key, value in train_output.metrics.items() if isinstance(value, (int, float))},
                eval_result={key: float(value) for key, value in eval_metrics.items() if isinstance(value, (int, float))} if eval_metrics is not None else None,
            )
            self._write_metadata(model_layout.metadata_path, metadata)

            train_metrics_plots = self.eval_artifact_store.export_train_metrics_plots(model_layout)
            for warning_message in train_metrics_plots.warnings:
                logger.warning("训练指标折线图导出提示: %s", warning_message)

            artifacts = {
                "output_dir": model_layout.version_dir,
                "metadata": model_layout.metadata_path,
                "train_file": train_file,
                "train_metrics_csv": model_layout.train_metrics_csv,
                "lora_dir": model_layout.lora_dir,
                "checkpoints_dir": model_layout.checkpoints_dir,
                "merged_model_dir": model_layout.merged_model_dir,
                "initial_model_manifest": initial_model_manifest,
            }
            if resume_lora_dir is not None:
                artifacts["resume_lora_dir"] = resume_lora_dir
            if train_metrics_plots.loss_plot_path is not None:
                artifacts["train_loss_plot_png"] = train_metrics_plots.loss_plot_path
            if train_metrics_plots.eval_metrics_plot_path is not None:
                artifacts["train_eval_metrics_plot_png"] = train_metrics_plots.eval_metrics_plot_path
            if validation_file is not None:
                artifacts["validation_file"] = validation_file
                artifacts["validation_label_metrics_csv"] = model_layout.validation_label_metrics_csv
            if spec.export_dir is not None:
                export_metadata_path = spec.export_dir / f"{run_id}_{METADATA_FILENAME}"
                self._write_metadata(export_metadata_path, metadata)
                artifacts["export_metadata"] = export_metadata_path

            return StageResult(
                stage="train",
                success=True,
                message="BERT LoRA 训练已完成，训练产物与元数据已写入模型资产目录。",
                artifacts=artifacts,
                metrics={
                    "train_examples": float(len(train_records)),
                    "validation_examples": float(len(validation_records)) if validation_records is not None else 0.0,
                    "num_labels": float(len(label_to_id)),
                    **({key: float(value) for key, value in train_output.metrics.items() if isinstance(value, (int, float))}),
                    **({key: float(value) for key, value in eval_metrics.items() if isinstance(value, (int, float))} if eval_metrics is not None else {}),
                },
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("训练失败")
            rolled_back_path = self._rollback_run_id(spec.task_name, run_id)
            return StageResult(
                stage="train",
                success=False,
                message=(
                    f"训练失败: {exc}；已删除模型目录: {rolled_back_path}"
                    if rolled_back_path is not None
                    else f"训练失败: {exc}"
                ),
            )