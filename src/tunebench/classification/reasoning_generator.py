"""分类 reasoning 数据增强执行器。"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from tunebench.artifacts import DatasetPathManager, get_dataset_path_manager
from tunebench.backends.llamafactory.prompting import resolve_label_guidance
from tunebench.contracts import ReasoningGenerationSpec, RunPlan, StageResult
from tunebench.util import get_logger

from .dataset_loader import (
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
    VALIDATION_SPLIT_NAME,
    load_classification_records,
    resolve_optional_split_file,
    resolve_split_file,
    validate_classification_records,
)


logger = get_logger("classification.reasoning_generator")

_STAGE_NAME = "generate-reasoning"
_STAGE_FILE_SUFFIX = ".jsonl"
_DEFAULT_LABEL_PROFILE = "l1_5class"
_DEFAULT_PROMPT_VERSION = "reasoning_v1"
_DEFAULT_ENDPOINT_URL = "http://192.168.75.109:30114/v1/chat/completions"
_DEFAULT_REASONING_GUIDANCE = resolve_label_guidance(("ICT巡检", "态势查询", "ICT排障", "配置管理", "知识问答"))


@dataclass(frozen=True, slots=True)
class _ReasoningLabelProfile:
    key: str
    labels: tuple[str, ...]
    label_definitions: dict[str, str]
    boundary_rules: tuple[str, ...]
    generation_examples: tuple[tuple[str, str, str], ...]

    def render_label_definitions(self) -> str:
        lines = [f"- {label}: {self.label_definitions[label]}" for label in self.labels]
        return "\n".join(lines)

    def render_boundary_rules(self) -> str:
        return "\n".join(f"- {rule}" for rule in self.boundary_rules)

    def render_generation_examples(self) -> str:
        lines: list[str] = []
        for index, (query, label, reasoning) in enumerate(self.generation_examples, start=1):
            lines.extend(
                [
                    f"示例{index}输入: {query}",
                    f"示例{index}金标标签: {label}",
                    f"示例{index}输出: {json.dumps({'reasoning': reasoning}, ensure_ascii=False)}",
                ]
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _ChatCompletionUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class _ChatCompletionResult:
    content: str
    reasoning_field: str | None
    usage: _ChatCompletionUsage | None


@dataclass(frozen=True, slots=True)
class _VerificationResult:
    passed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    source_index: int
    split_name: str
    text: str
    label: str


_LABEL_PROFILES: dict[str, _ReasoningLabelProfile] = {
    _DEFAULT_LABEL_PROFILE: _ReasoningLabelProfile(
        key=_DEFAULT_LABEL_PROFILE,
        labels=("ICT巡检", "态势查询", "ICT排障", "配置管理", "知识问答"),
        label_definitions=_DEFAULT_REASONING_GUIDANCE.label_definitions,
        boundary_rules=(
            "不要重新分类，必须严格围绕给定金标标签生成依据。",
            *_DEFAULT_REASONING_GUIDANCE.rules,
            "reasoning 必须是可展示的分类依据，不要展开长链路思考，不要输出候选标签比较过程。",
        ),
        generation_examples=tuple(
            (example.query, example.label, example.reasoning)
            for example in _DEFAULT_REASONING_GUIDANCE.examples
        ),
    )
}


class _OpenAICompatibleReasoningClient:
    """基于 OpenAI 兼容协议的异步客户端。"""

    def __init__(self, spec: ReasoningGenerationSpec, semaphore: asyncio.Semaphore) -> None:
        self._spec = spec
        self._semaphore = semaphore
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(spec.api_key_env_var)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(timeout=spec.request_timeout_seconds, headers=headers)

    async def __aenter__(self) -> _OpenAICompatibleReasoningClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
        await self._client.aclose()

    async def create_chat_completion(self, messages: list[dict[str, str]]) -> _ChatCompletionResult:
        payload = {
            "model": self._spec.teacher_model,
            "messages": messages,
            "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with self._semaphore:
            response = await self._client.post(self._spec.endpoint_url, json=payload)
        response.raise_for_status()
        payload = response.json()

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("大模型返回缺少 choices。")

        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("大模型返回缺少 message。")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("大模型返回 content 为空。")

        usage_payload = payload.get("usage")
        usage: _ChatCompletionUsage | None = None
        if isinstance(usage_payload, dict):
            usage = _ChatCompletionUsage(
                prompt_tokens=_safe_int(usage_payload.get("prompt_tokens")),
                completion_tokens=_safe_int(usage_payload.get("completion_tokens")),
                total_tokens=_safe_int(usage_payload.get("total_tokens")),
            )

        reasoning_field = message.get("reasoning")
        return _ChatCompletionResult(
            content=content,
            reasoning_field=reasoning_field if isinstance(reasoning_field, str) else None,
            usage=usage,
        )


class ClassificationReasoningGenerator:
    """为标准分类数据生成 reasoning。"""

    def __init__(self, dataset_path_manager: DatasetPathManager | None = None) -> None:
        self.dataset_path_manager = dataset_path_manager or get_dataset_path_manager()

    def build_plan(self, spec: ReasoningGenerationSpec) -> RunPlan:
        """生成 reasoning 数据增强计划。"""
        source_layout = self.dataset_path_manager.build_layout(spec.task_name, spec.source_dataset_version)
        target_layout = self.dataset_path_manager.build_layout(spec.task_name, spec.target_dataset_version)
        outputs: dict[str, str] = {
            "source_dataset_version_dir": str(source_layout.version_dir),
            "target_dataset_version_dir": str(target_layout.version_dir),
            "metadata": str(target_layout.metadata_path),
        }
        for split_name in spec.splits:
            outputs[f"stage_{split_name}"] = str(target_layout.stage_dir / f"{split_name}{_STAGE_FILE_SUFFIX}")
            outputs[f"final_{split_name}"] = str(target_layout.final_dir / f"{split_name}.jsonl")
        return RunPlan(
            stage=_STAGE_NAME,
            summary="基于标准分类数据生成 reasoning 增强版本。",
            inputs=asdict(spec),
            outputs=outputs,
            notes=[
                "worker 会逐个 split 读取 final 层标准数据，并输出新的 stage/final 数据版本。",
                "resume=True 时，会尝试跳过已存在的 source_index 记录。",
                "teacher_model、endpoint_url 和 api_key_env_var 共同决定远端推理请求行为。",
            ],
        )

    def run(self, spec: ReasoningGenerationSpec) -> StageResult:
        """执行 reasoning 生成逻辑。"""
        try:
            self._validate_spec(spec)
            return asyncio.run(self._run_async(spec))
        except Exception as exc:  # pragma: no cover - CLI 防护分支
            logger.exception("reasoning 生成失败")
            return StageResult(
                stage=_STAGE_NAME,
                success=False,
                message=f"reasoning 生成失败: {exc}",
            )

    async def _run_async(self, spec: ReasoningGenerationSpec) -> StageResult:
        profile = _resolve_label_profile(spec.label_profile)
        source_layout = self.dataset_path_manager.build_layout(spec.task_name, spec.source_dataset_version)
        target_layout = self.dataset_path_manager.ensure_layout(spec.task_name, spec.target_dataset_version)

        artifacts: dict[str, Path] = {
            "dataset_version_dir": target_layout.version_dir,
            "metadata": target_layout.metadata_path,
        }
        metrics: dict[str, float] = {}
        metadata = {
            "task_name": spec.task_name,
            "dataset_version": spec.target_dataset_version,
            "source_dataset_version": spec.source_dataset_version,
            "stage": _STAGE_NAME,
            "label_profile": spec.label_profile,
            "prompt_version": spec.prompt_version,
            "teacher_model": spec.teacher_model,
            "endpoint_url": spec.endpoint_url,
            "api_key_env_var": spec.api_key_env_var,
            "max_concurrency": spec.max_concurrency,
            "max_attempts": spec.max_attempts,
            "enable_model_verify": spec.enable_model_verify,
            "request_timeout_seconds": spec.request_timeout_seconds,
            "resume": spec.resume,
            "sample_limit": spec.sample_limit,
            "final_schema": {
                "source_index": "int",
                "text": "string",
                "label": "string",
                "reasoning": "string | null",
                "status": "accepted | rejected",
                "errors": "list[string]",
            },
            "splits": {},
        }

        semaphore = asyncio.Semaphore(spec.max_concurrency)
        async with _OpenAICompatibleReasoningClient(spec, semaphore) as client:
            for split_name in spec.splits:
                source_file = _resolve_source_split_file(source_layout.final_dir, split_name)
                if source_file is None:
                    logger.info("跳过缺失 split: %s", split_name)
                    continue

                split_result = await self._process_split(
                    spec=spec,
                    profile=profile,
                    client=client,
                    split_name=split_name,
                    source_file=source_file,
                    target_layout=target_layout,
                )
                metadata["splits"][split_name] = split_result["metadata"]
                metrics.update(split_result["metrics"])
                artifacts[f"stage_{split_name}"] = split_result["stage_path"]
                artifacts[f"final_{split_name}"] = split_result["final_path"]

        await asyncio.to_thread(
            target_layout.metadata_path.write_text,
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return StageResult(
            stage=_STAGE_NAME,
            success=True,
            message="reasoning 数据增强已完成，已生成新的 stage/final 数据版本。",
            artifacts=artifacts,
            metrics=metrics,
        )

    async def _process_split(
        self,
        *,
        spec: ReasoningGenerationSpec,
        profile: _ReasoningLabelProfile,
        client: _OpenAICompatibleReasoningClient,
        split_name: str,
        source_file: Path,
        target_layout: Any,
    ) -> dict[str, Any]:
        raw_records = await asyncio.to_thread(load_classification_records, source_file)
        source_records = validate_classification_records(raw_records, split_name)
        if spec.sample_limit is not None:
            source_records = source_records[: spec.sample_limit]

        stage_path = target_layout.stage_dir / f"{split_name}{_STAGE_FILE_SUFFIX}"
        final_path = target_layout.final_dir / source_file.name
        output_format = "jsonl" if source_file.suffix == ".jsonl" else "json"

        processed_indexes: set[int] = set()
        if spec.resume:
            existing_stage_records = await _load_jsonl_file(stage_path)
            for record in existing_stage_records:
                if not isinstance(record, dict):
                    continue
                source_index = _safe_int(record.get("source_index"))
                if source_index is not None:
                    processed_indexes.add(source_index)
        else:
            if stage_path.exists():
                stage_path.unlink()

        pending_queue: asyncio.Queue[_SourceRecord | None] = asyncio.Queue()
        result_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        writer_task = asyncio.create_task(_stage_writer(result_queue, stage_path))
        worker_count = max(1, spec.max_concurrency)
        workers = [
            asyncio.create_task(self._worker(spec=spec, profile=profile, client=client, pending_queue=pending_queue, result_queue=result_queue))
            for _ in range(worker_count)
        ]

        for source_index, record in enumerate(source_records, start=1):
            if source_index in processed_indexes:
                continue
            await pending_queue.put(
                _SourceRecord(
                    source_index=source_index,
                    split_name=split_name,
                    text=record["text"],
                    label=record["label"],
                )
            )

        for _ in workers:
            await pending_queue.put(None)

        await pending_queue.join()
        await asyncio.gather(*workers)
        await result_queue.put(None)
        await result_queue.join()
        await writer_task

        stage_records = await _load_jsonl_file(stage_path)
        final_records = _build_final_records_from_stage(stage_records)
        await _write_records(final_path, final_records, output_format)

        accepted_count = sum(1 for record in stage_records if record.get("status") == "accepted")
        rejected_count = sum(1 for record in stage_records if record.get("status") == "rejected")
        total_count = len(stage_records)
        skipped_count = sum(1 for record in stage_records if record.get("status") == "skipped")
        accept_rate = accepted_count / total_count if total_count else 0.0

        return {
            "stage_path": stage_path,
            "final_path": final_path,
            "metadata": {
                "source_dataset_version": spec.source_dataset_version,
                "source_input_path": str(source_file),
                "output_format": output_format,
                "row_count": len(source_records),
                "processed_row_count": total_count,
                "accepted_row_count": accepted_count,
                "rejected_row_count": rejected_count,
                "skipped_row_count": skipped_count,
                "stage_output_path": str(stage_path),
                "final_output_path": str(final_path),
            },
            "metrics": {
                f"{split_name}_row_count": float(len(source_records)),
                f"{split_name}_accepted_count": float(accepted_count),
                f"{split_name}_rejected_count": float(rejected_count),
                f"{split_name}_accept_rate": accept_rate,
            },
        }

    async def _worker(
        self,
        *,
        spec: ReasoningGenerationSpec,
        profile: _ReasoningLabelProfile,
        client: _OpenAICompatibleReasoningClient,
        pending_queue: asyncio.Queue[_SourceRecord | None],
        result_queue: asyncio.Queue[dict[str, Any] | None],
    ) -> None:
        while True:
            source_record = await pending_queue.get()
            if source_record is None:
                pending_queue.task_done()
                return

            try:
                stage_record = await self._generate_reasoning_record(
                    spec=spec,
                    profile=profile,
                    client=client,
                    source_record=source_record,
                )
            except Exception as exc:  # pragma: no cover - 防护分支
                logger.exception(
                    "reasoning worker 未捕获异常: split=%s, source_index=%s",
                    source_record.split_name,
                    source_record.source_index,
                )
                stage_record = {
                    "source_index": source_record.source_index,
                    "split": source_record.split_name,
                    "text": source_record.text,
                    "label": source_record.label,
                    "teacher_model": spec.teacher_model,
                    "prompt_version": spec.prompt_version,
                    "generation_attempts": spec.max_attempts,
                    "verification_attempts": 0,
                    "raw_generation_content": None,
                    "api_message_reasoning": None,
                    "parsed_reasoning": None,
                    "verification_passed": False,
                    "status": "rejected",
                    "errors": [f"未捕获异常: {exc}"],
                    "generation_usage": None,
                    "verification_usage": None,
                    "generation_latency_ms": None,
                    "verification_latency_ms": None,
                }
            await result_queue.put(stage_record)
            pending_queue.task_done()

    async def _generate_reasoning_record(
        self,
        *,
        spec: ReasoningGenerationSpec,
        profile: _ReasoningLabelProfile,
        client: _OpenAICompatibleReasoningClient,
        source_record: _SourceRecord,
    ) -> dict[str, Any]:
        errors: list[str] = []
        last_content: str | None = None
        last_reasoning_field: str | None = None
        last_generation_usage: dict[str, int | None] | None = None
        last_generation_latency_ms: float | None = None
        last_verification_usage: dict[str, int | None] | None = None
        last_verification_latency_ms: float | None = None
        verification_attempts = 0

        for attempt in range(1, spec.max_attempts + 1):
            generation_messages = _build_generation_messages(profile, source_record)
            generation_started_at = perf_counter()
            try:
                generation_result = await client.create_chat_completion(generation_messages)
                last_generation_latency_ms = round((perf_counter() - generation_started_at) * 1000, 2)
                last_generation_usage = generation_result.usage.to_dict() if generation_result.usage else None
                last_content = generation_result.content
                last_reasoning_field = generation_result.reasoning_field

                reasoning_payload = _extract_json_payload(generation_result.content)
                reasoning_text = _normalize_reasoning_payload(reasoning_payload)

                if not spec.enable_model_verify:
                    return {
                        "source_index": source_record.source_index,
                        "split": source_record.split_name,
                        "text": source_record.text,
                        "label": source_record.label,
                        "teacher_model": spec.teacher_model,
                        "prompt_version": spec.prompt_version,
                        "generation_attempts": attempt,
                        "verification_attempts": 0,
                        "raw_generation_content": last_content,
                        "api_message_reasoning": last_reasoning_field,
                        "parsed_reasoning": reasoning_text,
                        "verification_passed": None,
                        "status": "accepted",
                        "errors": [],
                        "generation_usage": last_generation_usage,
                        "verification_usage": None,
                        "generation_latency_ms": last_generation_latency_ms,
                        "verification_latency_ms": None,
                    }

                verification_started_at = perf_counter()
                verification_result, verification_usage = await self._verify_reasoning(
                    client=client,
                    profile=profile,
                    source_record=source_record,
                    reasoning_text=reasoning_text,
                )
                verification_attempts += 1
                last_verification_latency_ms = round((perf_counter() - verification_started_at) * 1000, 2)
                last_verification_usage = verification_usage.to_dict() if verification_usage else None
                if verification_result.passed:
                    return {
                        "source_index": source_record.source_index,
                        "split": source_record.split_name,
                        "text": source_record.text,
                        "label": source_record.label,
                        "teacher_model": spec.teacher_model,
                        "prompt_version": spec.prompt_version,
                        "generation_attempts": attempt,
                        "verification_attempts": verification_attempts,
                        "raw_generation_content": last_content,
                        "api_message_reasoning": last_reasoning_field,
                        "parsed_reasoning": reasoning_text,
                        "verification_passed": True,
                        "status": "accepted",
                        "errors": [],
                        "generation_usage": last_generation_usage,
                        "verification_usage": last_verification_usage,
                        "generation_latency_ms": last_generation_latency_ms,
                        "verification_latency_ms": last_verification_latency_ms,
                    }

                errors.append(f"第 {attempt} 次校验未通过: {verification_result.reason}")
            except Exception as exc:
                logger.warning(
                    "reasoning 生成失败: split=%s, source_index=%s, attempt=%s, error=%s",
                    source_record.split_name,
                    source_record.source_index,
                    attempt,
                    exc,
                )
                last_generation_latency_ms = round((perf_counter() - generation_started_at) * 1000, 2)
                errors.append(f"第 {attempt} 次生成失败: {exc}")

        return {
            "source_index": source_record.source_index,
            "split": source_record.split_name,
            "text": source_record.text,
            "label": source_record.label,
            "teacher_model": spec.teacher_model,
            "prompt_version": spec.prompt_version,
            "generation_attempts": spec.max_attempts,
            "verification_attempts": verification_attempts,
            "raw_generation_content": last_content,
            "api_message_reasoning": last_reasoning_field,
            "parsed_reasoning": None,
            "verification_passed": False,
            "status": "rejected",
            "errors": errors,
            "generation_usage": last_generation_usage,
            "verification_usage": last_verification_usage,
            "generation_latency_ms": last_generation_latency_ms,
            "verification_latency_ms": last_verification_latency_ms,
        }

    async def _verify_reasoning(
        self,
        *,
        client: _OpenAICompatibleReasoningClient,
        profile: _ReasoningLabelProfile,
        source_record: _SourceRecord,
        reasoning_text: str,
    ) -> tuple[_VerificationResult, _ChatCompletionUsage | None]:
        verification_messages = _build_verification_messages(profile, source_record, reasoning_text)
        verification_response = await client.create_chat_completion(verification_messages)
        verification_payload = _extract_json_payload(verification_response.content)
        verification_result = _normalize_verification_payload(verification_payload)
        return verification_result, verification_response.usage

    def _validate_spec(self, spec: ReasoningGenerationSpec) -> None:
        if spec.source_dataset_version == spec.target_dataset_version:
            raise ValueError("source_dataset_version 与 target_dataset_version 不能相同。")
        if not spec.teacher_model.strip():
            raise ValueError("teacher_model 不能为空。")
        if not spec.endpoint_url.strip():
            raise ValueError("endpoint_url 不能为空。")
        if spec.max_concurrency <= 0:
            raise ValueError("max_concurrency 必须大于 0。")
        if spec.max_attempts <= 0:
            raise ValueError("max_attempts 必须大于 0。")
        if spec.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds 必须大于 0。")
        unsupported_splits = [split for split in spec.splits if split not in {TRAIN_SPLIT_NAME, VALIDATION_SPLIT_NAME, TEST_SPLIT_NAME}]
        if unsupported_splits:
            raise ValueError(f"存在不支持的 split: {unsupported_splits}")
        if not spec.splits:
            raise ValueError("至少需要指定一个 split。")
        _resolve_label_profile(spec.label_profile)


def _resolve_label_profile(profile_key: str) -> _ReasoningLabelProfile:
    try:
        return _LABEL_PROFILES[profile_key]
    except KeyError as exc:
        supported_profiles = ", ".join(sorted(_LABEL_PROFILES))
        raise ValueError(f"未注册的 label_profile: {profile_key}；当前支持: {supported_profiles}") from exc


def _resolve_source_split_file(final_dir: Path, split_name: str) -> Path | None:
    if split_name == TRAIN_SPLIT_NAME:
        return resolve_split_file(final_dir, split_name)
    return resolve_optional_split_file(final_dir, split_name)


def _build_generation_messages(profile: _ReasoningLabelProfile, source_record: _SourceRecord) -> list[dict[str, str]]:
    system_prompt = (
        "你是 ICT 一级意图分类数据增强助手。"
        "你的任务不是重新分类，而是基于给定的金标标签生成一段可展示的分类依据。"
        "必须只输出 JSON 对象，且仅允许一个 reasoning 字段。"
    )
    user_prompt = "\n".join(
        [
            "请严格按以下要求生成结果：",
            "1. 只依据给定金标标签生成 supporting reasoning，不要改标签。",
            "2. 仅输出 JSON：{\"reasoning\":\"...\"}。",
            "3. reasoning 必须是中文，120 字以内。",
            "4. 不要输出 Markdown、代码块、解释、额外字段。",
            "5. 不要使用‘可能’‘猜测’等弱判断措辞。",
            "",
            "标签集合与定义：",
            profile.render_label_definitions(),
            "",
            "关键边界：",
            profile.render_boundary_rules(),
            "",
            "参考示例：",
            profile.render_generation_examples(),
            "",
            f"用户输入：{source_record.text}",
            f"金标标签：{source_record.label}",
            "现在只输出 JSON 结果。",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_verification_messages(
    profile: _ReasoningLabelProfile,
    source_record: _SourceRecord,
    reasoning_text: str,
) -> list[dict[str, str]]:
    system_prompt = (
        "你是分类依据审核器。"
        "请判断给定 reasoning 是否支持指定金标标签。"
        "必须只输出 JSON：{\"pass\": true/false, \"reason\": \"...\"}。"
    )
    user_prompt = "\n".join(
        [
            "请根据以下标签定义与边界进行审核。",
            profile.render_label_definitions(),
            "",
            "关键边界：",
            profile.render_boundary_rules(),
            "",
            f"用户输入：{source_record.text}",
            f"金标标签：{source_record.label}",
            f"待审核 reasoning：{reasoning_text}",
            "若 reasoning 准确支持该金标标签，则 pass=true；否则 pass=false 并给出简短原因。",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_json_payload(content: str) -> dict[str, Any]:
    normalized_content = content.strip()
    if normalized_content.startswith("```"):
        lines = normalized_content.splitlines()
        if len(lines) >= 3 and lines[-1].startswith("```"):
            normalized_content = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(normalized_content)
    except json.JSONDecodeError:
        start = normalized_content.find("{")
        end = normalized_content.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise ValueError("返回内容无法解析为 JSON。")
        payload = json.loads(normalized_content[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("返回 JSON 顶层必须是对象。")
    return payload


def _normalize_reasoning_payload(payload: dict[str, Any]) -> str:
    extra_keys = set(payload) - {"reasoning"}
    if extra_keys:
        raise ValueError(f"reasoning 返回存在非法字段: {sorted(extra_keys)}")

    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, str):
        raise ValueError("reasoning 字段必须是字符串。")

    normalized_reasoning = reasoning.strip()
    if not normalized_reasoning:
        raise ValueError("reasoning 不能为空。")
    if len(normalized_reasoning) > 120:
        raise ValueError("reasoning 长度不能超过 120 字。")
    return normalized_reasoning


def _normalize_verification_payload(payload: dict[str, Any]) -> _VerificationResult:
    extra_keys = set(payload) - {"pass", "reason"}
    if extra_keys:
        raise ValueError(f"verification 返回存在非法字段: {sorted(extra_keys)}")

    passed = payload.get("pass")
    if not isinstance(passed, bool):
        raise ValueError("verification pass 字段必须是布尔值。")

    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("verification reason 字段必须是字符串。")
    return _VerificationResult(passed=passed, reason=reason.strip())


def _build_final_records_from_stage(stage_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_records: dict[int, dict[str, Any]] = {}
    for record in stage_records:
        source_index = _safe_int(record.get("source_index"))
        text = record.get("text")
        label = record.get("label")
        status = record.get("status")
        reasoning = record.get("parsed_reasoning")
        errors = record.get("errors")
        if source_index is None or not isinstance(text, str) or not isinstance(label, str):
            continue

        normalized_status = str(status).strip() if isinstance(status, str) else ""
        if normalized_status not in {"accepted", "rejected"}:
            continue

        normalized_reasoning: str | None = None
        if reasoning is not None:
            if not isinstance(reasoning, str):
                continue
            stripped_reasoning = reasoning.strip()
            if stripped_reasoning:
                normalized_reasoning = stripped_reasoning

        normalized_errors: list[str] = []
        if isinstance(errors, list):
            normalized_errors = [str(item).strip() for item in errors if str(item).strip()]

        final_records[source_index] = {
            "source_index": source_index,
            "text": text,
            "label": label,
            "reasoning": normalized_reasoning,
            "status": normalized_status,
            "errors": normalized_errors,
        }
    return [final_records[index] for index in sorted(final_records)]


async def _stage_writer(result_queue: asyncio.Queue[dict[str, Any] | None], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        output_path.write_text("", encoding="utf-8")

    while True:
        record = await result_queue.get()
        try:
            if record is None:
                return
            await asyncio.to_thread(_append_jsonl_record, output_path, record)
        finally:
            result_queue.task_done()


def _append_jsonl_record(output_path: Path, record: dict[str, Any]) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _load_jsonl_file(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        return []

    def _read_jsonl() -> list[dict[str, Any]]:
        lines = [line for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [json.loads(line) for line in lines]

    return await asyncio.to_thread(_read_jsonl)


async def _write_records(output_path: Path, records: list[dict[str, Any]], output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        await asyncio.to_thread(
            output_path.write_text,
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    await asyncio.to_thread(
        output_path.write_text,
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def _safe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


__all__ = [
    "ClassificationReasoningGenerator",
    "ReasoningGenerationSpec",
    "_DEFAULT_ENDPOINT_URL",
    "_DEFAULT_LABEL_PROFILE",
    "_DEFAULT_PROMPT_VERSION",
]