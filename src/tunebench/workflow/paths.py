"""workflow 目录与文件路径管理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_WORKFLOW_ROOT_DIRNAME = "workflows"
_STATE_DIRNAME = "state"
_STAGE_RUNS_DIRNAME = "stage_runs"
_REQUEST_FILENAME = "request.json"
_RESULT_FILENAME = "result.json"
_LOG_FILENAME = "output.log"
_SQLITE_FILENAME = "workflow_state.sqlite3"


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _get_default_workflow_root_dir() -> Path:
    return _get_project_root() / "assets" / _WORKFLOW_ROOT_DIRNAME


def _resolve_root_dir(root_dir: str | Path | None) -> Path:
    if root_dir is None:
        return _get_default_workflow_root_dir()
    candidate = Path(root_dir)
    return candidate if candidate.is_absolute() else _get_project_root() / candidate


@dataclass(slots=True)
class StageRunPaths:
    """单个环节运行的文件路径。"""

    workflow_dir: Path
    stage_run_dir: Path
    request_path: Path
    result_path: Path
    log_path: Path


class WorkflowPathManager:
    """统一管理 workflow 状态目录与运行文件。"""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = _resolve_root_dir(root_dir)

    def get_state_dir(self) -> Path:
        return self.root_dir / _STATE_DIRNAME

    def get_sqlite_path(self) -> Path:
        return self.get_state_dir() / _SQLITE_FILENAME

    def get_workflow_dir(self, workflow_id: str) -> Path:
        return self.root_dir / workflow_id

    def get_stage_runs_dir(self, workflow_id: str) -> Path:
        return self.get_workflow_dir(workflow_id) / _STAGE_RUNS_DIRNAME

    def get_stage_run_dir(self, workflow_id: str, stage_run_id: str) -> Path:
        return self.get_stage_runs_dir(workflow_id) / stage_run_id

    def build_stage_run_paths(self, workflow_id: str, stage_run_id: str) -> StageRunPaths:
        stage_run_dir = self.get_stage_run_dir(workflow_id, stage_run_id)
        return StageRunPaths(
            workflow_dir=self.get_workflow_dir(workflow_id),
            stage_run_dir=stage_run_dir,
            request_path=stage_run_dir / _REQUEST_FILENAME,
            result_path=stage_run_dir / _RESULT_FILENAME,
            log_path=stage_run_dir / _LOG_FILENAME,
        )

    def ensure_root_dirs(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.get_state_dir().mkdir(parents=True, exist_ok=True)

    def ensure_stage_run_paths(self, workflow_id: str, stage_run_id: str) -> StageRunPaths:
        self.ensure_root_dirs()
        paths = self.build_stage_run_paths(workflow_id, stage_run_id)
        paths.workflow_dir.mkdir(parents=True, exist_ok=True)
        self.get_stage_runs_dir(workflow_id).mkdir(parents=True, exist_ok=True)
        paths.stage_run_dir.mkdir(parents=True, exist_ok=True)
        return paths
