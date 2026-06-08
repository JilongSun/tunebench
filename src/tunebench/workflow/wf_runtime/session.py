"""单个 workflow 的运行时会话。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import aiofiles

from tunebench.artifacts import get_dataset_path_manager, get_model_path_manager
from tunebench.workflow.models import (
    BuildStructuredTargetRequest,
    EvaluateModelRequest,
    GenerateReasoningRequest,
    PrepareDatasetRequest,
    StageName,
    StageRunRecord,
    StageStatus,
    TrainModelRequest,
    WorkflowEventRecord,
    WorkflowRecord,
    WorkflowSnapshot,
    WorkflowStatus,
    utc_now_iso,
)
from tunebench.workflow.paths import WorkflowPathManager
from tunebench.workflow.store import WorkflowStateStore


@dataclass(slots=True)
class WorkflowStageLaunch:
    """描述一次环节启动结果。"""

    stage_run: StageRunRecord
    process: asyncio.subprocess.Process
    log_handle: Any


class WorkflowRuntimeSession:
    """封装单个 workflow 的异步运行时上下文。"""

    def __init__(
        self,
        *,
        workflow_id: str,
        store: WorkflowStateStore,
        path_manager: WorkflowPathManager,
        watch_tasks: dict[str, asyncio.Task[None]],
        stage_run_id_factory: Callable[[StageName], str],
        event_id_factory: Callable[[], str],
    ) -> None:
        self.workflow_id = workflow_id
        self.store = store
        self.path_manager = path_manager
        self.watch_tasks = watch_tasks
        self.dataset_path_manager = get_dataset_path_manager()
        self.model_path_manager = get_model_path_manager()
        self._stage_run_id_factory = stage_run_id_factory
        self._event_id_factory = event_id_factory
        self._workflow: WorkflowRecord | None = None

    async def __aenter__(self) -> "WorkflowRuntimeSession":
        self._workflow = await self._require_workflow(self.workflow_id)
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._workflow = None

    @property
    def workflow(self) -> WorkflowRecord:
        if self._workflow is None:
            raise RuntimeError("workflow 运行时会话尚未进入上下文。")
        return self._workflow

    async def start_stage(self, *, stage_name: StageName, stage_payload: dict[str, Any]) -> WorkflowStageLaunch:
        workflow = self.workflow
        stage_runs = await self.store.list_stage_runs(workflow.workflow_id)
        self._ensure_stage_can_start(workflow, stage_name, stage_payload, stage_runs)

        stage_run_id = self._stage_run_id_factory(stage_name)
        stage_paths = self.path_manager.ensure_stage_run_paths(workflow.workflow_id, stage_run_id)
        plan_payload = self._build_stage_plan_payload(workflow, stage_name, stage_payload)
        worker_payload = {
            "workflow_id": workflow.workflow_id,
            "stage_run_id": stage_run_id,
            "stage_name": stage_name.value,
            "task_name": workflow.task_name,
            "backend": workflow.backend,
            "request": stage_payload,
        }
        await self._write_json_file(stage_paths.request_path, worker_payload)

        stage_run = StageRunRecord(
            stage_run_id=stage_run_id,
            workflow_id=workflow.workflow_id,
            stage_name=stage_name,
            status=StageStatus.PENDING,
            request_payload=stage_payload,
            plan_payload=plan_payload,
            log_path=str(stage_paths.log_path),
            request_path=str(stage_paths.request_path),
            result_path=str(stage_paths.result_path),
        )
        await self.store.create_stage_run(stage_run)
        await self.store.append_event(
            WorkflowEventRecord(
                event_id=self._event_id_factory(),
                workflow_id=workflow.workflow_id,
                stage_run_id=stage_run_id,
                event_type="stage_queued",
                payload={
                    "stage_name": stage_name.value,
                    "inputs": plan_payload.get("inputs", {}),
                    "outputs": plan_payload.get("outputs", {}),
                },
            )
        )

        log_handle = await asyncio.to_thread(stage_paths.log_path.open, "ab")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tunebench.workflow.worker",
            "--stage",
            stage_name.value,
            "--request-file",
            str(stage_paths.request_path),
            "--result-file",
            str(stage_paths.result_path),
            cwd=(workflow.runtime.working_dir or None),
            env=workflow.runtime.build_env(dict(os.environ)),
            stdout=log_handle,
            stderr=asyncio.subprocess.STDOUT,
        )

        now = utc_now_iso()
        running_stage_run = replace(
            stage_run,
            status=StageStatus.RUNNING,
            pid=process.pid,
            started_at=now,
            updated_at=now,
            version=stage_run.version + 1,
        )
        running_workflow = replace(
            workflow,
            status=WorkflowStatus.RUNNING,
            updated_at=now,
            version=workflow.version + 1,
        )
        await self.store.update_stage_run(running_stage_run)
        await self.store.update_workflow(running_workflow)
        await self.store.append_event(
            WorkflowEventRecord(
                event_id=self._event_id_factory(),
                workflow_id=workflow.workflow_id,
                stage_run_id=stage_run_id,
                event_type="stage_started",
                payload={"stage_name": stage_name.value, "pid": process.pid},
            )
        )
        self._workflow = running_workflow
        return WorkflowStageLaunch(stage_run=running_stage_run, process=process, log_handle=log_handle)

    async def get_state(self, *, event_limit: int = 50) -> WorkflowSnapshot:
        workflow = await self._require_workflow(self.workflow_id)
        self._workflow = workflow
        stage_runs = await self.store.list_stage_runs(self.workflow_id)
        for stage_run in stage_runs:
            await self.refresh_stage_run(stage_run)
        workflow = await self._require_workflow(self.workflow_id)
        self._workflow = workflow
        stage_runs = await self.store.list_stage_runs(self.workflow_id)
        events = await self.store.list_events(self.workflow_id, limit=event_limit)

        # 分离最近一次失败和已完成的历史产物
        last_failure = self._find_last_failure(stage_runs)
        completed_artifacts = tuple(
            run for run in stage_runs if run.status == StageStatus.SUCCEEDED
        )

        return WorkflowSnapshot(
            workflow=workflow,
            stage_runs=tuple(stage_runs),
            events=tuple(events),
            last_failure=last_failure,
            completed_artifacts=completed_artifacts,
        )

    @staticmethod
    def _find_last_failure(stage_runs: list[StageRunRecord]) -> dict[str, Any] | None:
        """查找最近一次失败的环节，返回简化的失败上下文。"""
        for run in reversed(stage_runs):
            if run.status != StageStatus.FAILED:
                continue
            failure_info: dict[str, Any] = {
                "stage_run_id": run.stage_run_id,
                "stage_name": run.stage_name.value,
                "finished_at": run.finished_at,
                "exit_code": run.exit_code,
            }
            if run.result_payload:
                failure_info["message"] = run.result_payload.get("message", "")
                failure_info["error_detail"] = run.result_payload.get("error", "")
            return failure_info
        return None

    async def tail_stage_log(self, stage_run_id: str, *, max_bytes: int = 8192) -> str:
        stage_run = await self._require_stage_run(stage_run_id)
        log_path = Path(stage_run.log_path)
        if not log_path.exists():
            return ""
        return await asyncio.to_thread(self._read_log_tail, log_path, max_bytes)

    async def finalize_stage(self, stage_run_id: str, *, exit_code: int | None) -> None:
        stage_run = await self._require_stage_run(stage_run_id)
        workflow = self.workflow
        result_payload = await self._try_load_json_file(Path(stage_run.result_path))
        now = utc_now_iso()

        if result_payload is None:
            result_payload = {
                "stage": stage_run.stage_name.value,
                "success": False,
                "message": "worker 进程结束，但未生成 result.json。",
            }

        if result_payload.get("success"):
            next_stage_status = StageStatus.SUCCEEDED
        else:
            next_stage_status = StageStatus.FAILED

        updated_stage_run = replace(
            stage_run,
            status=next_stage_status,
            result_payload=result_payload,
            exit_code=exit_code,
            finished_at=now,
            updated_at=now,
            version=stage_run.version + 1,
        )
        stage_runs = await self.store.list_stage_runs(workflow.workflow_id)
        updated_stage_runs = self._replace_stage_run(stage_runs, updated_stage_run)
        updated_workflow = replace(
            workflow,
            status=self._derive_workflow_status(updated_stage_runs),
            updated_at=now,
            version=workflow.version + 1,
        )
        await self.store.update_stage_run(updated_stage_run)
        await self.store.update_workflow(updated_workflow)
        await self.store.append_event(
            WorkflowEventRecord(
                event_id=self._event_id_factory(),
                workflow_id=workflow.workflow_id,
                stage_run_id=stage_run.stage_run_id,
                event_type="stage_finished",
                payload={
                    "stage_name": stage_run.stage_name.value,
                    "status": updated_stage_run.status.value,
                    "exit_code": exit_code,
                },
            )
        )
        self._workflow = updated_workflow

    async def refresh_stage_run(self, stage_run: StageRunRecord) -> None:
        if stage_run.status != StageStatus.RUNNING:
            return
        if stage_run.stage_run_id in self.watch_tasks:
            return
        result_payload = await self._try_load_json_file(Path(stage_run.result_path))
        if result_payload is not None:
            await self.finalize_stage(stage_run.stage_run_id, exit_code=stage_run.exit_code)
            return
        if stage_run.pid is None:
            return
        is_alive = await asyncio.to_thread(self._pid_exists, stage_run.pid)
        if not is_alive:
            await self.finalize_stage(stage_run.stage_run_id, exit_code=stage_run.exit_code)

    async def _require_workflow(self, workflow_id: str) -> WorkflowRecord:
        workflow = await self.store.get_workflow(workflow_id)
        if workflow is None:
            raise KeyError(f"workflow 不存在: {workflow_id}")
        return workflow

    async def _require_stage_run(self, stage_run_id: str) -> StageRunRecord:
        stage_run = await self.store.get_stage_run(stage_run_id)
        if stage_run is None:
            raise KeyError(f"stage_run 不存在: {stage_run_id}")
        if stage_run.workflow_id != self.workflow_id:
            raise ValueError(f"stage_run 不属于当前 workflow: {stage_run_id}")
        return stage_run

    def _ensure_stage_can_start(
        self,
        workflow: WorkflowRecord,
        stage_name: StageName,
        stage_payload: dict[str, Any],
        stage_runs: list[StageRunRecord],
    ) -> None:
        if stage_name not in set(workflow.enabled_stages):
            raise ValueError(f"workflow 未启用环节: {stage_name.value}")
        active_stage_run = next(
            (record for record in stage_runs if record.status in {StageStatus.RUNNING, StageStatus.PENDING}),
            None,
        )
        if active_stage_run is not None:
            raise ValueError(
                "workflow 当前仍有执行中的 operation: "
                f"{active_stage_run.stage_name.value} ({active_stage_run.stage_run_id})"
            )
        self._validate_stage_resources(workflow, stage_name, stage_payload)

    def _build_stage_plan_payload(
        self,
        workflow: WorkflowRecord,
        stage_name: StageName,
        stage_payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "operation": stage_name.value,
            "workflow_id": workflow.workflow_id,
            "inputs": self._describe_stage_inputs(stage_name, stage_payload),
            "outputs": self._describe_stage_outputs(workflow, stage_name, stage_payload),
        }

    def _validate_stage_resources(
        self,
        workflow: WorkflowRecord,
        stage_name: StageName,
        stage_payload: dict[str, Any],
    ) -> None:
        if stage_name == StageName.PREPARE_DATASET:
            request = PrepareDatasetRequest.from_payload(stage_payload)
            self._ensure_dataset_absent(workflow.task_name, request.dataset_version)
            return
        if stage_name == StageName.GENERATE_REASONING:
            request = GenerateReasoningRequest.from_payload(stage_payload)
            self._ensure_dataset_exists(workflow.task_name, request.source_dataset_version)
            self._ensure_dataset_absent(workflow.task_name, request.target_dataset_version)
            return
        if stage_name == StageName.BUILD_STRUCTURED_TARGET:
            request = BuildStructuredTargetRequest.from_payload(stage_payload)
            self._ensure_dataset_exists(workflow.task_name, request.source_dataset_version)
            self._ensure_dataset_absent(workflow.task_name, request.target_dataset_version)
            return
        if stage_name == StageName.TRAIN_MODEL:
            request = TrainModelRequest.from_payload(stage_payload)
            self._ensure_identifier("run_id", request.run_id)
            self._ensure_dataset_exists(workflow.task_name, request.dataset_version)
            self._ensure_model_absent(workflow.backend, workflow.task_name, request.run_id)
            return
        if stage_name == StageName.EVALUATE_MODEL:
            request = EvaluateModelRequest.from_payload(stage_payload)
            self._ensure_identifier("run_id", request.run_id)
            self._ensure_dataset_exists(workflow.task_name, request.dataset_version)
            model_layout = self._ensure_model_exists(workflow.backend, workflow.task_name, request.run_id)
            self._ensure_evaluate_outputs_absent(model_layout, export_xlsx=request.export_xlsx)
            return
        raise ValueError(f"未知 operation: {stage_name.value}")

    def _describe_stage_inputs(self, stage_name: StageName, stage_payload: dict[str, Any]) -> dict[str, Any]:
        if stage_name == StageName.PREPARE_DATASET:
            request = PrepareDatasetRequest.from_payload(stage_payload)
            return {"input_path": request.input_path}
        if stage_name == StageName.GENERATE_REASONING:
            request = GenerateReasoningRequest.from_payload(stage_payload)
            return {
                "source_dataset_version": request.source_dataset_version,
                "teacher_model": request.teacher_model,
            }
        if stage_name == StageName.BUILD_STRUCTURED_TARGET:
            request = BuildStructuredTargetRequest.from_payload(stage_payload)
            return {"source_dataset_version": request.source_dataset_version}
        if stage_name == StageName.TRAIN_MODEL:
            request = TrainModelRequest.from_payload(stage_payload)
            return {
                "dataset_version": request.dataset_version,
                "resume_lora": request.resume_lora,
            }
        request = EvaluateModelRequest.from_payload(stage_payload)
        return {
            "dataset_version": request.dataset_version,
            "run_id": request.run_id,
            "artifact_type": request.artifact_type,
        }

    def _describe_stage_outputs(
        self,
        workflow: WorkflowRecord,
        stage_name: StageName,
        stage_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if stage_name == StageName.PREPARE_DATASET:
            request = PrepareDatasetRequest.from_payload(stage_payload)
            layout = self.dataset_path_manager.build_layout(workflow.task_name, request.dataset_version)
            return {"dataset_version": request.dataset_version, "dataset_dir": str(layout.version_dir)}
        if stage_name == StageName.GENERATE_REASONING:
            request = GenerateReasoningRequest.from_payload(stage_payload)
            layout = self.dataset_path_manager.build_layout(workflow.task_name, request.target_dataset_version)
            return {"dataset_version": request.target_dataset_version, "dataset_dir": str(layout.version_dir)}
        if stage_name == StageName.BUILD_STRUCTURED_TARGET:
            request = BuildStructuredTargetRequest.from_payload(stage_payload)
            layout = self.dataset_path_manager.build_layout(workflow.task_name, request.target_dataset_version)
            return {"dataset_version": request.target_dataset_version, "dataset_dir": str(layout.version_dir)}
        if stage_name == StageName.TRAIN_MODEL:
            request = TrainModelRequest.from_payload(stage_payload)
            layout = self.model_path_manager.build_layout(workflow.backend, workflow.task_name, request.run_id)
            return {"run_id": request.run_id, "model_dir": str(layout.version_dir)}
        request = EvaluateModelRequest.from_payload(stage_payload)
        layout = self.model_path_manager.build_layout(workflow.backend, workflow.task_name, request.run_id)
        return {
            "run_id": request.run_id,
            "dataset_version": request.dataset_version,
            "eval_dir": str(layout.eval_dir),
        }

    def _derive_workflow_status(self, stage_runs: list[StageRunRecord]) -> WorkflowStatus:
        if any(stage_run.status in {StageStatus.PENDING, StageStatus.RUNNING} for stage_run in stage_runs):
            return WorkflowStatus.RUNNING
        if stage_runs and stage_runs[-1].status == StageStatus.FAILED:
            return WorkflowStatus.FAILED
        return WorkflowStatus.IDLE

    def _replace_stage_run(
        self,
        stage_runs: list[StageRunRecord],
        updated_stage_run: StageRunRecord,
    ) -> list[StageRunRecord]:
        return [
            (updated_stage_run if stage_run.stage_run_id == updated_stage_run.stage_run_id else stage_run)
            for stage_run in stage_runs
        ]

    def _ensure_identifier(self, name: str, value: str) -> None:
        if not value.strip():
            raise ValueError(f"{name} 不能为空字符串。")
        if "/" in value or "\\" in value:
            raise ValueError(f"{name} 不能包含路径分隔符。")

    def _ensure_dataset_exists(self, task_name: str, dataset_version: str) -> Path:
        layout = self.dataset_path_manager.build_layout(task_name, dataset_version)
        if not layout.version_dir.exists():
            raise ValueError(f"dataset_version 不存在，无法作为输入: {layout.version_dir}")
        return layout.version_dir

    def _ensure_dataset_absent(self, task_name: str, dataset_version: str) -> Path:
        layout = self.dataset_path_manager.build_layout(task_name, dataset_version)
        if layout.version_dir.exists():
            raise ValueError(f"dataset_version 已存在，可能覆盖已有产物: {layout.version_dir}")
        return layout.version_dir

    def _ensure_model_exists(self, backend: str, task_name: str, run_id: str):
        layout = self.model_path_manager.build_layout(backend, task_name, run_id)
        if not layout.version_dir.exists():
            raise ValueError(f"run_id 不存在，无法作为输入: {layout.version_dir}")
        return layout

    def _ensure_model_absent(self, backend: str, task_name: str, run_id: str) -> None:
        layout = self.model_path_manager.build_layout(backend, task_name, run_id)
        if layout.version_dir.exists():
            raise ValueError(f"run_id 已存在，可能覆盖已有产物: {layout.version_dir}")

    def _ensure_evaluate_outputs_absent(self, model_layout: Any, *, export_xlsx: bool) -> None:
        blocking_paths = [
            model_layout.test_metrics_csv,
            model_layout.test_label_metrics_csv,
            model_layout.test_predictions_csv,
        ]
        if export_xlsx:
            blocking_paths.append(model_layout.eval_report_xlsx)
        existing_paths = [str(path) for path in blocking_paths if path.exists()]
        if existing_paths:
            raise ValueError("评测输出已存在，继续执行可能覆盖已有产物: " + ", ".join(existing_paths))

    async def _write_json_file(self, output_path: Path, payload: dict[str, Any]) -> None:
        async with aiofiles.open(output_path, "w", encoding="utf-8") as file_obj:
            await file_obj.write(json.dumps(payload, ensure_ascii=False, indent=2))

    async def _try_load_json_file(self, input_path: Path) -> dict[str, Any] | None:
        if not input_path.exists():
            return None
        try:
            async with aiofiles.open(input_path, "r", encoding="utf-8") as file_obj:
                return json.loads(await file_obj.read())
        except json.JSONDecodeError:
            return None

    def _read_log_tail(self, log_path: Path, max_bytes: int) -> str:
        with log_path.open("rb") as file_obj:
            file_obj.seek(0, os.SEEK_END)
            file_size = file_obj.tell()
            file_obj.seek(max(file_size - max_bytes, 0), os.SEEK_SET)
            return file_obj.read().decode("utf-8", errors="replace")

    def _pid_exists(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True