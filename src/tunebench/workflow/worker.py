"""workflow 环节 worker 入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import aiofiles

from tunebench.util import get_logger, setup_logging
from tunebench.workflow.models import (
    BuildStructuredTargetRequest,
    EvaluateModelRequest,
    GenerateReasoningRequest,
    PrepareDatasetRequest,
    StageName,
    TrainModelRequest,
)


logger = get_logger("workflow.worker")


def _normalize_stage_result_payload(result: Any) -> dict[str, Any]:
    artifacts = {}
    for key, value in result.artifacts.items():
        artifacts[key] = str(value)
    return {
        "stage": result.stage,
        "success": result.success,
        "message": result.message,
        "artifacts": artifacts,
        "metrics": dict(result.metrics),
    }


async def _read_json_file(input_path: Path) -> dict[str, Any]:
    async with aiofiles.open(input_path, "r", encoding="utf-8") as file_obj:
        return json.loads(await file_obj.read())


async def _write_json_file(output_path: Path, payload: dict[str, Any]) -> None:
    async with aiofiles.open(output_path, "w", encoding="utf-8") as file_obj:
        await file_obj.write(json.dumps(payload, ensure_ascii=False, indent=2))


async def _run_prepare_dataset_stage(payload: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path

    from tunebench.classification import ClassificationDataPreparer
    from tunebench.contracts import DatasetSpec

    request = PrepareDatasetRequest.from_payload(payload["request"])
    spec = request.to_spec(task_name=str(payload["task_name"]))
    spec = DatasetSpec(
        task_name=spec.task_name,
        input_path=Path(str(spec.input_path)),
        dataset_version=spec.dataset_version,
        text_key=spec.text_key,
        label_key=spec.label_key,
        output_path=(None if spec.output_path is None else Path(str(spec.output_path))),
        output_format=spec.output_format,
        sheet_name=spec.sheet_name,
        validation_ratio=spec.validation_ratio,
        split_seed=spec.split_seed,
        is_test=spec.is_test,
        allowed_labels=spec.allowed_labels,
    )
    result = await asyncio.to_thread(ClassificationDataPreparer().run, spec)
    return _normalize_stage_result_payload(result)


async def _run_train_stage(payload: dict[str, Any]) -> dict[str, Any]:
    from tunebench.backends import get_classification_backend

    request = TrainModelRequest.from_payload(payload["request"])
    spec = request.to_spec(
        task_name=str(payload["task_name"]),
        backend=str(payload["backend"]),
    )
    backend_runner = get_classification_backend(spec.backend)
    result = await asyncio.to_thread(backend_runner.run_train, spec)
    return _normalize_stage_result_payload(result)


async def _run_generate_reasoning_stage(payload: dict[str, Any]) -> dict[str, Any]:
    from tunebench.classification.reasoning_generator import ClassificationReasoningGenerator

    request = GenerateReasoningRequest.from_payload(payload["request"])
    spec = request.to_spec(task_name=str(payload["task_name"]))
    result = await asyncio.to_thread(ClassificationReasoningGenerator().run, spec)
    return _normalize_stage_result_payload(result)


async def _run_build_structured_target_stage(payload: dict[str, Any]) -> dict[str, Any]:
    from tunebench.classification import ClassificationStructuredTargetBuilder

    request = BuildStructuredTargetRequest.from_payload(payload["request"])
    spec = request.to_spec(task_name=str(payload["task_name"]))
    result = await asyncio.to_thread(ClassificationStructuredTargetBuilder().run, spec)
    return _normalize_stage_result_payload(result)


async def _run_evaluate_stage(payload: dict[str, Any]) -> dict[str, Any]:
    from tunebench.backends import get_classification_backend

    request = EvaluateModelRequest.from_payload(payload["request"])
    spec = request.to_spec(
        task_name=str(payload["task_name"]),
        backend=str(payload["backend"]),
    )
    backend_runner = get_classification_backend(spec.backend)
    result = await asyncio.to_thread(backend_runner.run_evaluate, spec)
    return _normalize_stage_result_payload(result)


async def _dispatch_stage(stage_name: StageName, payload: dict[str, Any]) -> dict[str, Any]:
    if stage_name == StageName.PREPARE_DATASET:
        return await _run_prepare_dataset_stage(payload)
    if stage_name == StageName.GENERATE_REASONING:
        return await _run_generate_reasoning_stage(payload)
    if stage_name == StageName.BUILD_STRUCTURED_TARGET:
        return await _run_build_structured_target_stage(payload)
    if stage_name == StageName.TRAIN_MODEL:
        return await _run_train_stage(payload)
    if stage_name == StageName.EVALUATE_MODEL:
        return await _run_evaluate_stage(payload)
    raise ValueError(f"worker 尚未支持该阶段: {stage_name.value}")


async def _main_async() -> int:
    parser = argparse.ArgumentParser(description="TuneBench workflow stage worker")
    parser.add_argument("--stage", required=True, help="阶段名称")
    parser.add_argument("--request-file", required=True, help="请求文件路径")
    parser.add_argument("--result-file", required=True, help="结果文件路径")
    args = parser.parse_args()

    setup_logging()
    stage_name = StageName(args.stage)
    request_file = Path(args.request_file)
    result_file = Path(args.result_file)

    try:
        payload = await _read_json_file(request_file)
        result_payload = await _dispatch_stage(stage_name, payload)
    except Exception as exc:  # pragma: no cover - worker 防护分支
        logger.exception("workflow worker 执行失败: stage=%s", stage_name.value)
        result_payload = {
            "stage": stage_name.value,
            "success": False,
            "message": f"workflow worker 执行失败: {exc}",
            "artifacts": {},
            "metrics": {},
        }

    await _write_json_file(result_file, result_payload)
    return 0 if result_payload.get("success") else 1


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
