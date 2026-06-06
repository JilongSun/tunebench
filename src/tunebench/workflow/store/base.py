"""workflow 状态存储抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from tunebench.workflow.models import StageRunRecord, WorkflowEventRecord, WorkflowRecord


class WorkflowStateStore(ABC):
    """workflow 状态存储的统一异步接口。"""

    @abstractmethod
    async def initialize(self) -> None:
        """初始化底层存储。"""

    @abstractmethod
    async def create_workflow(self, record: WorkflowRecord) -> None:
        """写入新的 workflow 记录。"""

    @abstractmethod
    async def get_workflow(self, workflow_id: str) -> WorkflowRecord | None:
        """按主键读取 workflow。"""

    @abstractmethod
    async def update_workflow(self, record: WorkflowRecord) -> None:
        """更新 workflow 记录。"""

    @abstractmethod
    async def create_stage_run(self, record: StageRunRecord) -> None:
        """写入新的阶段运行记录。"""

    @abstractmethod
    async def get_stage_run(self, stage_run_id: str) -> StageRunRecord | None:
        """按主键读取阶段运行记录。"""

    @abstractmethod
    async def update_stage_run(self, record: StageRunRecord) -> None:
        """更新阶段运行记录。"""

    @abstractmethod
    async def list_stage_runs(self, workflow_id: str) -> list[StageRunRecord]:
        """读取 workflow 下全部阶段运行记录。"""

    @abstractmethod
    async def append_event(self, record: WorkflowEventRecord) -> None:
        """追加 workflow 事件。"""

    @abstractmethod
    async def list_events(self, workflow_id: str, *, limit: int = 50) -> list[WorkflowEventRecord]:
        """读取 workflow 事件。"""
