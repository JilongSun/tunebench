"""workflow 应用层服务。"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

from tunebench.util import get_logger
from tunebench.workflow.models import (
    BuildStructuredTargetRequest,
    DEFAULT_STAGE_SEQUENCE,
    EvaluateModelRequest,
    GenerateReasoningRequest,
    PrepareDatasetRequest,
    StageName,
    StageRunRecord,
    TrainModelRequest,
    WorkflowCreateRequest,
    WorkflowEventRecord,
    WorkflowPreview,
    WorkflowRecord,
    WorkflowSnapshot,
    WorkflowStagePlan,
    WorkflowStatus,
)
from tunebench.workflow.paths import WorkflowPathManager
from tunebench.workflow.wf_runtime import WorkflowRuntimeSession
from tunebench.workflow.store import SqliteWorkflowStateStore, WorkflowStateStore


logger = get_logger("workflow.service")


class WorkflowService:
    """负责协调 workflow 生命周期与后台 watcher。"""

    def __init__(
        self,
        *,
        store: WorkflowStateStore | None = None,
        path_manager: WorkflowPathManager | None = None,
    ) -> None:
        self.path_manager = path_manager or WorkflowPathManager()
        self.store = store or SqliteWorkflowStateStore(self.path_manager.get_sqlite_path())
        self._store_initialized = False
        self._watch_tasks: dict[str, asyncio.Task[None]] = {}

    async def initialize(self) -> None:
        if self._store_initialized:
            return
        self.path_manager.ensure_root_dirs()
        await self.store.initialize()
        self._store_initialized = True

    async def preview_workflow(self, request: WorkflowCreateRequest) -> WorkflowPreview:
        enabled_stages = request.normalized_enabled_stages()
        review_required_stages = set(request.normalized_review_required_stages())
        run_id = request.run_id or self._generate_run_id()
        stages: list[WorkflowStagePlan] = []
        for index, stage_name in enumerate(enabled_stages):
            depends_on = enabled_stages[:index]
            stages.append(
                WorkflowStagePlan(
                    stage_name=stage_name,
                    depends_on=depends_on,
                    requires_review=stage_name in review_required_stages,
                )
            )
        return WorkflowPreview(
            task_name=request.task_name,
            backend=request.backend,
            run_id=run_id,
            stages=tuple(stages),
        )

    async def create_workflow(self, request: WorkflowCreateRequest) -> WorkflowSnapshot:
        await self.initialize()
        self._validate_backend(request.backend)

        enabled_stages = request.normalized_enabled_stages() or DEFAULT_STAGE_SEQUENCE
        review_required_stages = request.normalized_review_required_stages()
        run_id = request.run_id or self._generate_run_id()
        self._validate_identifier("run_id", run_id)
        self._validate_new_run_id(request.backend, request.task_name, run_id)

        workflow = WorkflowRecord(
            workflow_id=self._generate_workflow_id(),
            task_name=request.task_name,
            backend=request.backend,
            run_id=run_id,
            runtime=request.runtime,
            enabled_stages=enabled_stages,
            review_required_stages=review_required_stages,
            status=WorkflowStatus.DRAFT,
        )
        await self.store.create_workflow(workflow)
        await self.store.append_event(
            WorkflowEventRecord(
                event_id=self._generate_event_id(),
                workflow_id=workflow.workflow_id,
                event_type="workflow_created",
                payload={
                    "backend": workflow.backend,
                    "run_id": workflow.run_id,
                    "enabled_stages": [stage.value for stage in workflow.enabled_stages],
                },
            )
        )
        return await self.get_workflow_state(workflow.workflow_id)

    async def run_prepare_dataset(self, workflow_id: str, request: PrepareDatasetRequest) -> StageRunRecord:
        return await self._run_stage(
            workflow_id=workflow_id,
            stage_name=StageName.PREPARE_DATASET,
            stage_payload=request.to_payload(),
        )

    async def run_train_model(self, workflow_id: str, request: TrainModelRequest) -> StageRunRecord:
        return await self._run_stage(
            workflow_id=workflow_id,
            stage_name=StageName.TRAIN_MODEL,
            stage_payload=request.to_payload(),
        )

    async def run_generate_reasoning(self, workflow_id: str, request: GenerateReasoningRequest) -> StageRunRecord:
        return await self._run_stage(
            workflow_id=workflow_id,
            stage_name=StageName.GENERATE_REASONING,
            stage_payload=request.to_payload(),
        )

    async def run_build_structured_target(self, workflow_id: str, request: BuildStructuredTargetRequest) -> StageRunRecord:
        return await self._run_stage(
            workflow_id=workflow_id,
            stage_name=StageName.BUILD_STRUCTURED_TARGET,
            stage_payload=request.to_payload(),
        )

    async def run_evaluate_model(self, workflow_id: str, request: EvaluateModelRequest) -> StageRunRecord:
        return await self._run_stage(
            workflow_id=workflow_id,
            stage_name=StageName.EVALUATE_MODEL,
            stage_payload=request.to_payload(),
        )

    async def approve_stage(self, stage_run_id: str) -> StageRunRecord:
        await self.initialize()
        stage_run = await self._require_stage_run(stage_run_id)
        async with self._open_runtime(stage_run.workflow_id) as runtime:
            return await runtime.approve_stage(stage_run_id)

    async def reject_stage(self, stage_run_id: str, *, reason: str) -> StageRunRecord:
        await self.initialize()
        stage_run = await self._require_stage_run(stage_run_id)
        async with self._open_runtime(stage_run.workflow_id) as runtime:
            return await runtime.reject_stage(stage_run_id, reason=reason)

    async def get_workflow_state(self, workflow_id: str, *, event_limit: int = 50) -> WorkflowSnapshot:
        await self.initialize()
        async with self._open_runtime(workflow_id) as runtime:
            return await runtime.get_state(event_limit=event_limit)

    async def tail_stage_log(self, stage_run_id: str, *, max_bytes: int = 8192) -> str:
        await self.initialize()
        stage_run = await self._require_stage_run(stage_run_id)
        async with self._open_runtime(stage_run.workflow_id) as runtime:
            return await runtime.tail_stage_log(stage_run_id, max_bytes=max_bytes)

    async def _run_stage(self, *, workflow_id: str, stage_name: StageName, stage_payload: dict[str, Any]) -> StageRunRecord:
        await self.initialize()
        async with self._open_runtime(workflow_id) as runtime:
            launch = await runtime.start_stage(stage_name=stage_name, stage_payload=stage_payload)
        self._watch_tasks[launch.stage_run.stage_run_id] = asyncio.create_task(
            self._watch_stage_process(
                stage_run_id=launch.stage_run.stage_run_id,
                process=launch.process,
                log_handle=launch.log_handle,
            )
        )
        return launch.stage_run

    async def _watch_stage_process(
        self,
        *,
        stage_run_id: str,
        process: asyncio.subprocess.Process,
        log_handle: Any,
    ) -> None:
        try:
            exit_code = await process.wait()
            await self._finalize_stage(stage_run_id, exit_code=exit_code)
        finally:
            await asyncio.to_thread(log_handle.close)
            self._watch_tasks.pop(stage_run_id, None)

    async def _finalize_stage(self, stage_run_id: str, *, exit_code: int | None) -> None:
        stage_run = await self._require_stage_run(stage_run_id)
        async with self._open_runtime(stage_run.workflow_id) as runtime:
            await runtime.finalize_stage(stage_run_id, exit_code=exit_code)

    async def _require_stage_run(self, stage_run_id: str) -> StageRunRecord:
        stage_run = await self.store.get_stage_run(stage_run_id)
        if stage_run is None:
            raise KeyError(f"stage_run 不存在: {stage_run_id}")
        return stage_run

    def _open_runtime(self, workflow_id: str) -> WorkflowRuntimeSession:
        return WorkflowRuntimeSession(
            workflow_id=workflow_id,
            store=self.store,
            path_manager=self.path_manager,
            watch_tasks=self._watch_tasks,
            stage_run_id_factory=self._generate_stage_run_id,
            event_id_factory=self._generate_event_id,
        )

    def _generate_workflow_id(self) -> str:
        return f"wf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

    def _generate_run_id(self) -> str:
        return f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

    def _generate_stage_run_id(self, stage_name: StageName) -> str:
        return f"{stage_name.value}_{uuid.uuid4().hex[:12]}"

    def _generate_event_id(self) -> str:
        return f"event_{uuid.uuid4().hex[:12]}"

    def _validate_backend(self, backend: str) -> None:
        if backend not in {"bert", "llamafactory"}:
            raise ValueError(f"不支持的 backend: {backend}")

    def _validate_identifier(self, name: str, value: str) -> None:
        if not value.strip():
            raise ValueError(f"{name} 不能为空字符串。")
        if "/" in value or "\\" in value:
            raise ValueError(f"{name} 不能包含路径分隔符。")

    def _validate_new_run_id(self, backend: str, task_name: str, run_id: str) -> None:
        from tunebench.artifacts import get_model_path_manager

        version_dir = get_model_path_manager().build_layout(backend, task_name, run_id).version_dir
        if version_dir.exists():
            raise ValueError(f"run_id 已存在，请更换标识符: {version_dir}")
