"""内存版 workflow 状态存储。"""

from __future__ import annotations

from tunebench.workflow.models import StageRunRecord, WorkflowEventRecord, WorkflowRecord
from tunebench.workflow.store.base import WorkflowStateStore


class InMemoryWorkflowStateStore(WorkflowStateStore):
    """用于测试和本地调试的内存存储实现。"""

    def __init__(self) -> None:
        self._workflows: dict[str, WorkflowRecord] = {}
        self._stage_runs: dict[str, StageRunRecord] = {}
        self._events: list[WorkflowEventRecord] = []

    async def initialize(self) -> None:
        return None

    async def create_workflow(self, record: WorkflowRecord) -> None:
        self._workflows[record.workflow_id] = record

    async def get_workflow(self, workflow_id: str) -> WorkflowRecord | None:
        return self._workflows.get(workflow_id)

    async def update_workflow(self, record: WorkflowRecord) -> None:
        self._workflows[record.workflow_id] = record

    async def create_stage_run(self, record: StageRunRecord) -> None:
        self._stage_runs[record.stage_run_id] = record

    async def get_stage_run(self, stage_run_id: str) -> StageRunRecord | None:
        return self._stage_runs.get(stage_run_id)

    async def update_stage_run(self, record: StageRunRecord) -> None:
        self._stage_runs[record.stage_run_id] = record

    async def list_stage_runs(self, workflow_id: str) -> list[StageRunRecord]:
        return sorted(
            [record for record in self._stage_runs.values() if record.workflow_id == workflow_id],
            key=lambda record: record.created_at,
        )

    async def append_event(self, record: WorkflowEventRecord) -> None:
        self._events.append(record)

    async def list_events(self, workflow_id: str, *, limit: int = 50) -> list[WorkflowEventRecord]:
        records = [record for record in self._events if record.workflow_id == workflow_id]
        return records[-limit:]
