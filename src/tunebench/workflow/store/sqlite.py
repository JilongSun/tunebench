"""SQLite 版 workflow 状态存储。"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import JSON, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from tunebench.workflow.models import StageRunRecord, WorkflowEventRecord, WorkflowRecord
from tunebench.workflow.store.base import WorkflowStateStore


class _Base(DeclarativeBase):
    pass


class _WorkflowRow(_Base):
    __tablename__ = "workflow_records"

    workflow_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_name: Mapped[str] = mapped_column(String(255), nullable=False)
    backend: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    current_stage: Mapped[str | None] = mapped_column(String(128), nullable=True)
    runtime_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    enabled_stages: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    review_required_stages: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class _StageRunRow(_Base):
    __tablename__ = "stage_run_records"

    stage_run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    stage_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    request_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    plan_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    log_path: Mapped[str] = mapped_column(Text, nullable=False)
    request_path: Mapped[str] = mapped_column(Text, nullable=False)
    result_path: Mapped[str] = mapped_column(Text, nullable=False)
    requires_review: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class _WorkflowEventRow(_Base):
    __tablename__ = "workflow_event_records"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    stage_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)


class SqliteWorkflowStateStore(WorkflowStateStore):
    """基于 SQLite 的异步状态存储。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.engine: AsyncEngine = create_async_engine(
            f"sqlite+aiosqlite:///{self.db_path}",
            future=True,
        )
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self.engine.begin() as connection:
            await connection.run_sync(_Base.metadata.create_all)

    async def create_workflow(self, record: WorkflowRecord) -> None:
        async with self.session_factory() as session:
            session.add(self._to_workflow_row(record))
            await session.commit()

    async def get_workflow(self, workflow_id: str) -> WorkflowRecord | None:
        async with self.session_factory() as session:
            row = await session.get(_WorkflowRow, workflow_id)
            return None if row is None else self._from_workflow_row(row)

    async def update_workflow(self, record: WorkflowRecord) -> None:
        async with self.session_factory() as session:
            row = await session.get(_WorkflowRow, record.workflow_id)
            if row is None:
                raise KeyError(f"workflow 不存在: {record.workflow_id}")
            row.task_name = record.task_name
            row.backend = record.backend
            row.run_id = record.run_id
            row.status = record.status.value
            row.current_stage = None if record.current_stage is None else record.current_stage.value
            row.runtime_payload = record.runtime.to_payload()
            row.enabled_stages = [stage.value for stage in record.enabled_stages]
            row.review_required_stages = [stage.value for stage in record.review_required_stages]
            row.created_at = record.created_at
            row.updated_at = record.updated_at
            row.version = record.version
            await session.commit()

    async def create_stage_run(self, record: StageRunRecord) -> None:
        async with self.session_factory() as session:
            session.add(self._to_stage_run_row(record))
            await session.commit()

    async def get_stage_run(self, stage_run_id: str) -> StageRunRecord | None:
        async with self.session_factory() as session:
            row = await session.get(_StageRunRow, stage_run_id)
            return None if row is None else self._from_stage_run_row(row)

    async def update_stage_run(self, record: StageRunRecord) -> None:
        async with self.session_factory() as session:
            row = await session.get(_StageRunRow, record.stage_run_id)
            if row is None:
                raise KeyError(f"stage_run 不存在: {record.stage_run_id}")
            row.workflow_id = record.workflow_id
            row.stage_name = record.stage_name.value
            row.status = record.status.value
            row.request_payload = record.request_payload
            row.plan_payload = record.plan_payload
            row.result_payload = record.result_payload
            row.log_path = record.log_path
            row.request_path = record.request_path
            row.result_path = record.result_path
            row.requires_review = 1 if record.requires_review else 0
            row.pid = record.pid
            row.exit_code = record.exit_code
            row.started_at = record.started_at
            row.finished_at = record.finished_at
            row.created_at = record.created_at
            row.updated_at = record.updated_at
            row.version = record.version
            await session.commit()

    async def list_stage_runs(self, workflow_id: str) -> list[StageRunRecord]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(_StageRunRow).where(_StageRunRow.workflow_id == workflow_id).order_by(_StageRunRow.created_at.asc())
            )
            return [self._from_stage_run_row(row) for row in result.scalars().all()]

    async def append_event(self, record: WorkflowEventRecord) -> None:
        async with self.session_factory() as session:
            session.add(self._to_event_row(record))
            await session.commit()

    async def list_events(self, workflow_id: str, *, limit: int = 50) -> list[WorkflowEventRecord]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(_WorkflowEventRow)
                .where(_WorkflowEventRow.workflow_id == workflow_id)
                .order_by(_WorkflowEventRow.created_at.asc())
            )
            rows = result.scalars().all()
            if limit > 0:
                rows = rows[-limit:]
            return [self._from_event_row(row) for row in rows]

    def _to_workflow_row(self, record: WorkflowRecord) -> _WorkflowRow:
        return _WorkflowRow(
            workflow_id=record.workflow_id,
            task_name=record.task_name,
            backend=record.backend,
            run_id=record.run_id,
            status=record.status.value,
            current_stage=(None if record.current_stage is None else record.current_stage.value),
            runtime_payload=record.runtime.to_payload(),
            enabled_stages=[stage.value for stage in record.enabled_stages],
            review_required_stages=[stage.value for stage in record.review_required_stages],
            created_at=record.created_at,
            updated_at=record.updated_at,
            version=record.version,
        )

    def _from_workflow_row(self, row: _WorkflowRow) -> WorkflowRecord:
        return WorkflowRecord.from_payload(
            {
                "workflow_id": row.workflow_id,
                "task_name": row.task_name,
                "backend": row.backend,
                "run_id": row.run_id,
                "status": row.status,
                "current_stage": row.current_stage,
                "runtime": row.runtime_payload,
                "enabled_stages": row.enabled_stages,
                "review_required_stages": row.review_required_stages,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "version": row.version,
            }
        )

    def _to_stage_run_row(self, record: StageRunRecord) -> _StageRunRow:
        return _StageRunRow(
            stage_run_id=record.stage_run_id,
            workflow_id=record.workflow_id,
            stage_name=record.stage_name.value,
            status=record.status.value,
            request_payload=record.request_payload,
            plan_payload=record.plan_payload,
            result_payload=record.result_payload,
            log_path=record.log_path,
            request_path=record.request_path,
            result_path=record.result_path,
            requires_review=1 if record.requires_review else 0,
            pid=record.pid,
            exit_code=record.exit_code,
            started_at=record.started_at,
            finished_at=record.finished_at,
            created_at=record.created_at,
            updated_at=record.updated_at,
            version=record.version,
        )

    def _from_stage_run_row(self, row: _StageRunRow) -> StageRunRecord:
        return StageRunRecord.from_payload(
            {
                "stage_run_id": row.stage_run_id,
                "workflow_id": row.workflow_id,
                "stage_name": row.stage_name,
                "status": row.status,
                "request_payload": row.request_payload,
                "plan_payload": row.plan_payload,
                "result_payload": row.result_payload,
                "log_path": row.log_path,
                "request_path": row.request_path,
                "result_path": row.result_path,
                "requires_review": bool(row.requires_review),
                "pid": row.pid,
                "exit_code": row.exit_code,
                "started_at": row.started_at,
                "finished_at": row.finished_at,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "version": row.version,
            }
        )

    def _to_event_row(self, record: WorkflowEventRecord) -> _WorkflowEventRow:
        return _WorkflowEventRow(
            event_id=record.event_id,
            workflow_id=record.workflow_id,
            stage_run_id=record.stage_run_id,
            event_type=record.event_type,
            payload=record.payload,
            created_at=record.created_at,
        )

    def _from_event_row(self, row: _WorkflowEventRow) -> WorkflowEventRecord:
        return WorkflowEventRecord.from_payload(
            {
                "event_id": row.event_id,
                "workflow_id": row.workflow_id,
                "stage_run_id": row.stage_run_id,
                "event_type": row.event_type,
                "payload": row.payload,
                "created_at": row.created_at,
            }
        )
