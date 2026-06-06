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

from tunebench.workflow.models import (
    StageName,
    StageRunRecord,
    StageStatus,
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
        self._ensure_stage_can_start(workflow, stage_name, stage_runs)

        stage_run_id = self._stage_run_id_factory(stage_name)
        stage_paths = self.path_manager.ensure_stage_run_paths(workflow.workflow_id, stage_run_id)
        plan_payload = self._build_stage_plan_payload(workflow, stage_name, stage_payload)
        worker_payload = {
            "workflow_id": workflow.workflow_id,
            "stage_run_id": stage_run_id,
            "stage_name": stage_name.value,
            "task_name": workflow.task_name,
            "backend": workflow.backend,
            "run_id": workflow.run_id,
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
            requires_review=stage_name in set(workflow.review_required_stages),
        )
        await self.store.create_stage_run(stage_run)
        await self.store.append_event(
            WorkflowEventRecord(
                event_id=self._event_id_factory(),
                workflow_id=workflow.workflow_id,
                stage_run_id=stage_run_id,
                event_type="stage_queued",
                payload={"stage_name": stage_name.value},
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
            current_stage=stage_name,
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

    async def approve_stage(self, stage_run_id: str) -> StageRunRecord:
        stage_run = await self._require_stage_run(stage_run_id)
        if stage_run.status != StageStatus.AWAITING_REVIEW:
            raise ValueError(f"环节未处于 awaiting_review: {stage_run_id}")

        workflow = self.workflow
        now = utc_now_iso()
        updated_stage_run = replace(
            stage_run,
            status=StageStatus.APPROVED,
            updated_at=now,
            version=stage_run.version + 1,
        )
        updated_workflow = replace(
            workflow,
            status=(WorkflowStatus.COMPLETED if self._is_last_stage(workflow, stage_run.stage_name) else WorkflowStatus.READY_NEXT),
            current_stage=None,
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
                event_type="stage_approved",
                payload={"stage_name": stage_run.stage_name.value},
            )
        )
        self._workflow = updated_workflow
        return updated_stage_run

    async def reject_stage(self, stage_run_id: str, *, reason: str) -> StageRunRecord:
        stage_run = await self._require_stage_run(stage_run_id)
        if stage_run.status != StageStatus.AWAITING_REVIEW:
            raise ValueError(f"环节未处于 awaiting_review: {stage_run_id}")

        workflow = self.workflow
        now = utc_now_iso()
        updated_stage_run = replace(
            stage_run,
            status=StageStatus.REJECTED,
            result_payload={**(stage_run.result_payload or {}), "review_rejection_reason": reason},
            updated_at=now,
            version=stage_run.version + 1,
        )
        updated_workflow = replace(
            workflow,
            status=WorkflowStatus.REJECTED,
            current_stage=stage_run.stage_name,
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
                event_type="stage_rejected",
                payload={"stage_name": stage_run.stage_name.value, "reason": reason},
            )
        )
        self._workflow = updated_workflow
        return updated_stage_run

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
        return WorkflowSnapshot(workflow=workflow, stage_runs=tuple(stage_runs), events=tuple(events))

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
            next_stage_status = StageStatus.AWAITING_REVIEW if stage_run.requires_review else StageStatus.APPROVED
            next_workflow_status = (
                WorkflowStatus.AWAITING_REVIEW
                if stage_run.requires_review
                else (WorkflowStatus.COMPLETED if self._is_last_stage(workflow, stage_run.stage_name) else WorkflowStatus.READY_NEXT)
            )
        else:
            next_stage_status = StageStatus.FAILED
            next_workflow_status = WorkflowStatus.FAILED

        updated_stage_run = replace(
            stage_run,
            status=next_stage_status,
            result_payload=result_payload,
            exit_code=exit_code,
            finished_at=now,
            updated_at=now,
            version=stage_run.version + 1,
        )
        updated_workflow = replace(
            workflow,
            status=next_workflow_status,
            current_stage=(stage_run.stage_name if next_workflow_status in {WorkflowStatus.AWAITING_REVIEW, WorkflowStatus.FAILED} else None),
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
        stage_runs: list[StageRunRecord],
    ) -> None:
        if stage_name not in set(workflow.enabled_stages):
            raise ValueError(f"workflow 未启用环节: {stage_name.value}")
        latest_runs = self._get_latest_stage_runs(stage_runs)
        expected_stage = self._get_next_startable_stage(workflow, latest_runs)
        if expected_stage is None:
            raise ValueError(f"workflow 已无可执行环节: {workflow.workflow_id}")
        if expected_stage != stage_name:
            raise ValueError(
                f"当前仅允许执行下一环节 {expected_stage.value}，不能直接执行 {stage_name.value}。"
            )
        blocking_stage = latest_runs.get(stage_name)
        if blocking_stage and blocking_stage.status in {StageStatus.RUNNING, StageStatus.PENDING, StageStatus.AWAITING_REVIEW}:
            raise ValueError(f"环节仍未结束或待审核: {stage_name.value}")

    def _get_next_startable_stage(
        self,
        workflow: WorkflowRecord,
        latest_runs: dict[StageName, StageRunRecord],
    ) -> StageName | None:
        for stage_name in workflow.enabled_stages:
            latest_run = latest_runs.get(stage_name)
            if latest_run is None:
                return stage_name
            if latest_run.status in {StageStatus.APPROVED, StageStatus.SKIPPED}:
                continue
            if latest_run.status in {StageStatus.FAILED, StageStatus.REJECTED}:
                return stage_name
            return None
        return None

    def _get_latest_stage_runs(self, stage_runs: list[StageRunRecord]) -> dict[StageName, StageRunRecord]:
        latest_runs: dict[StageName, StageRunRecord] = {}
        for record in stage_runs:
            latest_runs[record.stage_name] = record
        return latest_runs

    def _build_stage_plan_payload(
        self,
        workflow: WorkflowRecord,
        stage_name: StageName,
        stage_payload: dict[str, Any],
    ) -> dict[str, Any]:
        stage_index = workflow.enabled_stages.index(stage_name)
        return {
            "stage_name": stage_name.value,
            "depends_on": [item.value for item in workflow.enabled_stages[:stage_index]],
            "requires_review": stage_name in set(workflow.review_required_stages),
            "inputs": stage_payload,
        }

    def _is_last_stage(self, workflow: WorkflowRecord, stage_name: StageName) -> bool:
        return bool(workflow.enabled_stages) and workflow.enabled_stages[-1] == stage_name

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