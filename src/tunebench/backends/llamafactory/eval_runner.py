"""LlamaFactory 分类评测后端。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from tqdm import tqdm

from tunebench.artifacts import (
    DEFAULT_EVAL_REPORT_ARTIFACT_NAMES,
    DatasetPathManager,
    ModelPathManager,
    TEST_LABEL_METRICS_ARTIFACT_NAME,
    TEST_METRICS_ARTIFACT_NAME,
    TEST_PREDICTIONS_ARTIFACT_NAME,
    get_dataset_path_manager,
    get_model_path_manager,
)
from tunebench.artifacts.eval_store import EvalArtifactStore, FileSystemEvalArtifactStore
from tunebench.classification import TEST_SPLIT_NAME, load_classification_records, resolve_split_file, validate_classification_records
from tunebench.classification.structured_output_metrics import (
    assess_structured_output,
    compute_intent_metrics_bundle,
    sanitize_structured_output_text,
)
from tunebench.contracts import EvalSpec, RunPlan, StageResult
from tunebench.util import get_logger

from .generation import (
    build_messages,
    encode_prompt_ids_with_chat_template,
    encode_prompt_ids_with_llamafactory,
    generate_outputs_from_prompt_ids,
)
from .loaders import load_model_and_tokenizer
from .metadata import load_metadata, resolve_default_instruction, resolve_model_key, resolve_reasoning_mode, resolve_reasoning_suffix_style
from .policies import build_chat_reasoning_policy, build_reasoning_policy_from_metadata


logger = get_logger("backends.llamafactory.eval_runner")

_LLAMAFACTORY_BACKEND = "llamafactory"
_TEST_METRICS_FIELDNAMES = [
    "run_id",
    "split",
    "prompt_engine",
    "enable_thinking",
    "sample_count",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "avg_confidence",
    "json_valid_rate",
    "reasoning_length_pass_rate",
    "confidence_enum_pass_rate",
    "confidence_range_pass_rate",
    "exact_match_rate",
    "total_latency_ms",
    "avg_latency_ms",
]
_LABEL_METRICS_FIELDNAMES = [
    "run_id",
    "stage",
    "split",
    "epoch",
    "step",
    "label",
    "support",
    "precision",
    "recall",
    "f1",
]
_TEST_PREDICTIONS_FIELDNAMES = [
    "run_id",
    "split",
    "prompt_engine",
    "enable_thinking",
    "index",
    "text",
    "true_label",
    "pred_label",
    "is_correct",
    "confidence",
    "json_valid",
    "reasoning_char_count",
    "reasoning_length_valid",
    "confidence_enum_valid",
    "confidence_range_valid",
    "pred_intents",
    "errors",
    "cleaned_output",
    "raw_output",
    "finish_reason",
    "prompt_token_count",
    "generated_token_count",
]


@dataclass(frozen=True, slots=True)
class StructuredOutputEvalResult:
    """描述一次结构化输出评测的聚合结果。"""

    metrics_row: dict[str, Any]
    label_metrics: dict[str, dict[str, float | int]]
    prediction_rows: list[dict[str, Any]]


class _EvaluationProgressReporter:
    """使用 tqdm 在终端输出评测生成进度。"""

    def __init__(self, *, total_samples: int, batch_size: int) -> None:
        self.total_batches = (total_samples + batch_size - 1) // batch_size if total_samples else 0
        self._current_samples = 0
        self._progress_bar = tqdm(
            total=total_samples,
            desc="评测生成",
            unit="sample",
            dynamic_ncols=True,
            leave=True,
            disable=(total_samples <= 0),
        )

    def update(
        self,
        completed_samples: int,
        total_samples: int,
        completed_batches: int,
        total_batches: int,
        batch_latency_ms: float,
    ) -> None:
        if total_samples <= 0:
            return

        delta_samples = completed_samples - self._current_samples
        if delta_samples > 0:
            self._progress_bar.update(delta_samples)
            self._current_samples = completed_samples

        if completed_batches > 0:
            self._progress_bar.set_postfix_str(
                f"batch {completed_batches}/{total_batches}, 最近批次 {batch_latency_ms / 1000.0:.1f}s"
            )

    def close(self) -> None:
        self._progress_bar.close()


def build_label_metrics_rows(
    *,
    run_id: str,
    stage: str,
    split: str,
    label_metrics: dict[str, dict[str, float | int]],
    epoch: float | None = None,
    step: int | None = None,
) -> list[dict[str, float | int | str | None]]:
    """构建逐标签指标行。"""
    rows: list[dict[str, float | int | str | None]] = []
    for label_name in sorted(label_metrics):
        metrics = label_metrics[label_name]
        rows.append(
            {
                "run_id": run_id,
                "stage": stage,
                "split": split,
                "epoch": epoch,
                "step": step,
                "label": label_name,
                "support": int(metrics.get("support", 0)),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
            }
        )
    return rows


def evaluate_structured_records(
    *,
    run_id: str,
    split_name: str,
    records: Sequence[dict[str, Any]],
    label_names: Sequence[str],
    runtime: Any,
    instruction: str,
    batch_size: int,
    max_sequence_length: int | None,
    max_new_tokens: int | None,
    prompt_engine: str,
    enable_thinking: bool | None,
    reasoning_suffix_style: str | None,
    reasoning_mode: str | None,
    progress_callback: Any | None = None,
) -> StructuredOutputEvalResult:
    """对指定 split 记录执行结构化生成评测。"""
    if prompt_engine == "native":
        messages_list = [
            build_messages(
                instruction=instruction,
                text=str(record["text"]),
            )
            for record in records
        ]
        chat_template_kwargs: dict[str, object] = {}
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = enable_thinking
        prompt_id_batches = encode_prompt_ids_with_chat_template(
            runtime=runtime,
            messages_list=messages_list,
            max_sequence_length=max_sequence_length,
            chat_template_kwargs=chat_template_kwargs,
        )
    else:
        messages_list = [
            build_messages(
                instruction=instruction,
                text=str(record["text"]),
                reasoning_suffix_style=reasoning_suffix_style,
                reasoning_mode=reasoning_mode,
                apply_reasoning_suffix=True,
            )
            for record in records
        ]
        prompt_id_batches = encode_prompt_ids_with_llamafactory(
            runtime=runtime,
            messages_list=messages_list,
            max_sequence_length=max_sequence_length,
        )
    generated_outputs, total_latency_ms = generate_outputs_from_prompt_ids(
        runtime=runtime,
        prompt_id_batches=prompt_id_batches,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
    generated_outputs = list(generated_outputs)
    raw_outputs = [generated_output.text for generated_output in generated_outputs]
    cleaned_outputs = [sanitize_structured_output_text(raw_output) for raw_output in raw_outputs]

    assessments = [assess_structured_output(raw_output, label_names) for raw_output in cleaned_outputs]
    gold_intents = [{str(record["label"])} for record in records]
    predicted_intents = [set(assessment.predicted_intents) for assessment in assessments]
    intent_metrics, label_metrics = compute_intent_metrics_bundle(
        gold_intents=gold_intents,
        predicted_intents=predicted_intents,
        label_names=label_names,
    )

    valid_max_confidences = [assessment.max_confidence for assessment in assessments if assessment.max_confidence is not None]
    sample_count = len(records)
    metrics_row = {
        "run_id": run_id,
        "split": split_name,
        "prompt_engine": prompt_engine,
        "enable_thinking": enable_thinking if prompt_engine == "native" else None,
        "sample_count": sample_count,
        "precision_macro": intent_metrics["precision_macro"],
        "recall_macro": intent_metrics["recall_macro"],
        "f1_macro": intent_metrics["f1_macro"],
        "avg_confidence": sum(valid_max_confidences) / len(valid_max_confidences) if valid_max_confidences else 0.0,
        "json_valid_rate": sum(1 for assessment in assessments if assessment.json_valid) / sample_count if sample_count else 0.0,
        "reasoning_length_pass_rate": sum(1 for assessment in assessments if assessment.reasoning_length_valid) / sample_count
        if sample_count
        else 0.0,
        "confidence_enum_pass_rate": sum(1 for assessment in assessments if assessment.confidence_enum_valid) / sample_count
        if sample_count
        else 0.0,
        "confidence_range_pass_rate": sum(1 for assessment in assessments if assessment.confidence_range_valid) / sample_count
        if sample_count
        else 0.0,
        "exact_match_rate": sum(1 for gold, pred in zip(gold_intents, predicted_intents) if gold == pred) / sample_count
        if sample_count
        else 0.0,
        "total_latency_ms": total_latency_ms,
        "avg_latency_ms": total_latency_ms / sample_count if sample_count else 0.0,
    }
    prediction_rows = [
        {
            "run_id": run_id,
            "split": split_name,
            "prompt_engine": prompt_engine,
            "enable_thinking": enable_thinking if prompt_engine == "native" else None,
            "index": index,
            "text": str(record["text"]),
            "true_label": str(record["label"]),
            "pred_label": assessment.primary_intent or "",
            "is_correct": str(record["label"]) in set(assessment.predicted_intents),
            "confidence": assessment.max_confidence if assessment.max_confidence is not None else 0.0,
            "json_valid": assessment.json_valid,
            "reasoning_char_count": assessment.reasoning_char_count,
            "reasoning_length_valid": assessment.reasoning_length_valid,
            "confidence_enum_valid": assessment.confidence_enum_valid,
            "confidence_range_valid": assessment.confidence_range_valid,
            "pred_intents": json.dumps(list(assessment.predicted_intents), ensure_ascii=False),
            "errors": json.dumps(list(assessment.errors), ensure_ascii=False),
            "cleaned_output": cleaned_output,
            "raw_output": generated_output.text,
            "finish_reason": generated_output.finish_reason,
            "prompt_token_count": generated_output.prompt_token_count,
            "generated_token_count": generated_output.generated_token_count,
        }
        for index, (record, assessment, cleaned_output, generated_output) in enumerate(
            zip(records, assessments, cleaned_outputs, generated_outputs, strict=False)
        )
    ]
    return StructuredOutputEvalResult(
        metrics_row=metrics_row,
        label_metrics=label_metrics,
        prediction_rows=prediction_rows,
    )


class LlamaFactoryClassificationEvalRunner:
    """负责 LlamaFactory 分类独立评测与结果导出。"""

    def __init__(
        self,
        dataset_path_manager: DatasetPathManager | None = None,
        model_path_manager: ModelPathManager | None = None,
        artifact_store: EvalArtifactStore | None = None,
    ) -> None:
        self.dataset_path_manager = dataset_path_manager or get_dataset_path_manager()
        self.model_path_manager = model_path_manager or get_model_path_manager()
        self.artifact_store = artifact_store or FileSystemEvalArtifactStore()

    def _resolve_instruction(self, metadata: dict[str, Any]) -> str:
        return resolve_default_instruction(metadata)

    def _resolve_prompt_engine(self, spec: EvalSpec, metadata: dict[str, Any]) -> str:
        if spec.prompt_engine is not None:
            return spec.prompt_engine
        if resolve_model_key(metadata) == "qwen3_5_4b":
            return "native"
        return "llamafactory"

    def _resolve_enable_thinking(self, spec: EvalSpec, prompt_engine: str) -> bool | None:
        if prompt_engine != "native":
            return None
        return False if spec.enable_thinking is None else spec.enable_thinking

    def build_plan(self, spec: EvalSpec) -> RunPlan:
        model_layout = self.model_path_manager.build_layout(_LLAMAFACTORY_BACKEND, spec.task_name, spec.run_id)
        metadata = load_metadata(model_layout.metadata_path)
        prompt_engine = self._resolve_prompt_engine(spec, metadata)
        enable_thinking = self._resolve_enable_thinking(spec, prompt_engine)
        if prompt_engine == "native":
            reasoning_policy = build_chat_reasoning_policy(
                prompt_engine=prompt_engine,
                template=None,
                reasoning_mode=resolve_reasoning_mode(metadata),
                reasoning_suffix_style=resolve_reasoning_suffix_style(metadata),
                enable_thinking=enable_thinking,
            )
        else:
            reasoning_policy = build_reasoning_policy_from_metadata(metadata)
        return RunPlan(
            stage="evaluate",
            summary="执行 LlamaFactory 结构化输出评测并生成 JSON 与 intent 指标。",
            inputs=asdict(spec),
            outputs={
                "eval_dir": str(model_layout.eval_dir),
                "metadata": str(model_layout.metadata_path),
                "test_label_metrics_csv": str(model_layout.test_label_metrics_csv),
                "test_predictions_csv": str(model_layout.test_predictions_csv),
                "eval_report_xlsx": str(model_layout.eval_report_xlsx) if spec.export_xlsx else None,
                "test_metrics_csv": str(model_layout.test_metrics_csv),
                "merged_model_dir": str(model_layout.merged_model_dir),
                "lora_dir": str(model_layout.lora_dir),
                "reasoning_policy": reasoning_policy.to_payload(),
            },
            notes=[
                "会加载训练期 metadata 里的 model_name_or_path、template、reasoning_mode 与 label_names。",
                (
                    f"当前评测使用 prompt_engine={prompt_engine}，template={reasoning_policy.template}，"
                    f"reasoning_mode={reasoning_policy.effective_reasoning_mode}，"
                    f"reasoning_control={reasoning_policy.reasoning_control}。"
                ),
                (
                    "当 run 对应 qwen3.5 且未显式指定 --prompt-engine 时，默认改走 native 链，"
                    "避免继续依赖不稳定的 no_think 模板输出。"
                    if resolve_model_key(metadata) == "qwen3_5_4b"
                    else "默认沿用训练 metadata 对应的 LlamaFactory 模板链；如需切换可显式传 --prompt-engine。"
                ),
                "测试集样本会执行批量生成，并校验 JSON 合法性、reasoning 长度、confidence 枚举和值域。",
                f"默认会将 {'/'.join(DEFAULT_EVAL_REPORT_ARTIFACT_NAMES)} 汇总为 {model_layout.eval_report_xlsx.name}。",
            ],
        )

    def run(self, spec: EvalSpec) -> StageResult:
        try:
            logger.info(
                "开始 LlamaFactory 评测: task=%s, run_id=%s, dataset_version=%s, split=%s, artifact_type=%s",
                spec.task_name,
                spec.run_id,
                spec.dataset_version,
                TEST_SPLIT_NAME,
                spec.artifact_type,
            )
            model_layout = self.model_path_manager.ensure_layout(_LLAMAFACTORY_BACKEND, spec.task_name, spec.run_id)
            dataset_layout = self.dataset_path_manager.build_layout(spec.task_name, spec.dataset_version)
            metadata = load_metadata(model_layout.metadata_path)
            label_names_raw = metadata.get("label_names")
            if not isinstance(label_names_raw, list) or not all(isinstance(label_name, str) for label_name in label_names_raw):
                raise ValueError("metadata.label_names 缺失或类型无效。")
            label_names = tuple(label_names_raw)

            split_file = resolve_split_file(dataset_layout.final_dir, TEST_SPLIT_NAME)
            records = validate_classification_records(load_classification_records(split_file), TEST_SPLIT_NAME)
            logger.info("评测数据读取完成: split=%s, sample_count=%s", TEST_SPLIT_NAME, len(records))
            logger.info("开始加载评测模型与 tokenizer。")
            runtime = load_model_and_tokenizer(
                artifact_type=spec.artifact_type,
                metadata=metadata,
                model_layout=model_layout,
            )
            logger.info("评测模型加载完成。")
            instruction = self._resolve_instruction(metadata)
            prompt_engine = self._resolve_prompt_engine(spec, metadata)
            enable_thinking = self._resolve_enable_thinking(spec, prompt_engine)
            reasoning_suffix_style = resolve_reasoning_suffix_style(metadata)
            reasoning_mode = resolve_reasoning_mode(metadata)
            progress_reporter = _EvaluationProgressReporter(total_samples=len(records), batch_size=spec.batch_size)
            logger.info(
                "开始批量生成: sample_count=%s, batch_size=%s, total_batches=%s, prompt_engine=%s, enable_thinking=%s",
                len(records),
                spec.batch_size,
                progress_reporter.total_batches,
                prompt_engine,
                enable_thinking,
            )
            try:
                evaluation_result = evaluate_structured_records(
                    run_id=spec.run_id,
                    split_name=TEST_SPLIT_NAME,
                    records=records,
                    label_names=label_names,
                    runtime=runtime,
                    instruction=instruction,
                    batch_size=spec.batch_size,
                    max_sequence_length=spec.max_sequence_length,
                    max_new_tokens=spec.max_new_tokens,
                    prompt_engine=prompt_engine,
                    enable_thinking=enable_thinking,
                    reasoning_suffix_style=reasoning_suffix_style,
                    reasoning_mode=reasoning_mode,
                    progress_callback=progress_reporter.update,
                )
            finally:
                progress_reporter.close()
            logger.info("批量生成完成，开始汇总结构化输出指标。")

            self.artifact_store.append_artifact_rows(
                model_layout=model_layout,
                artifact_name=TEST_METRICS_ARTIFACT_NAME,
                fieldnames=_TEST_METRICS_FIELDNAMES,
                rows=[evaluation_result.metrics_row],
            )
            self.artifact_store.append_artifact_rows(
                model_layout=model_layout,
                artifact_name=TEST_LABEL_METRICS_ARTIFACT_NAME,
                fieldnames=_LABEL_METRICS_FIELDNAMES,
                rows=build_label_metrics_rows(
                    run_id=spec.run_id,
                    stage="evaluate",
                    split=TEST_SPLIT_NAME,
                    label_metrics=evaluation_result.label_metrics,
                ),
            )
            self.artifact_store.append_artifact_rows(
                model_layout=model_layout,
                artifact_name=TEST_PREDICTIONS_ARTIFACT_NAME,
                fieldnames=_TEST_PREDICTIONS_FIELDNAMES,
                rows=evaluation_result.prediction_rows,
            )

            artifacts = {
                "eval_dir": model_layout.eval_dir,
                "test_metrics_csv": model_layout.test_metrics_csv,
                "test_label_metrics_csv": model_layout.test_label_metrics_csv,
                "test_predictions_csv": model_layout.test_predictions_csv,
            }
            message = "LlamaFactory 结构化输出评测完成，结果已写入 CSV 文件。"
            if spec.export_xlsx:
                try:
                    export_result = self.artifact_store.export_eval_report(model_layout)
                    if export_result.output_path is not None:
                        artifacts["eval_report_xlsx"] = export_result.output_path
                        message = "LlamaFactory 结构化输出评测完成，结果已写入 CSV 并汇总为 XLSX。"
                    for warning_message in export_result.warnings:
                        logger.warning("XLSX 汇总导出提示: %s", warning_message)
                except Exception as export_exc:
                    logger.warning("XLSX 汇总导出失败: %s", export_exc)
                    message = f"LlamaFactory 结构化输出评测完成，CSV 已生成，但 XLSX 汇总导出失败: {export_exc}"

            return StageResult(
                stage="evaluate",
                success=True,
                message=message,
                artifacts=artifacts,
                metrics={
                    key: float(value)
                    for key, value in evaluation_result.metrics_row.items()
                    if isinstance(value, (int, float))
                },
            )
        except Exception as exc:
            logger.exception("LlamaFactory 评测失败")
            return StageResult(
                stage="evaluate",
                success=False,
                message=f"LlamaFactory 评测失败: {exc}",
            )


__all__ = [
    "LlamaFactoryClassificationEvalRunner",
    "StructuredOutputEvalResult",
    "build_label_metrics_rows",
    "evaluate_structured_records",
]
