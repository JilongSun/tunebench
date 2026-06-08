"""资产路径与目录结构管理工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .names import (
    build_eval_report_filename,
    CHECKPOINTS_DIRNAME,
    EVAL_DIRNAME,
    FINAL_DIRNAME,
    INITIAL_MODEL_DIRNAME,
    LORA_DIRNAME,
    MERGED_MODEL_DIRNAME,
    METADATA_FILENAME,
    RAW_DIRNAME,
    STAGE_DIRNAME,
    TEST_LABEL_METRICS_FILENAME,
    TEST_METRICS_FILENAME,
    TEST_PREDICTIONS_FILENAME,
    TRAIN_EVAL_METRICS_PLOT_FILENAME,
    TRAIN_LOSS_PLOT_FILENAME,
    TRAIN_METRICS_FILENAME,
    # VALIDATION_LABEL_METRICS_FILENAME,  # 暂时禁用
)


def _get_project_root() -> Path:
    """获取项目根目录。"""
    return Path(__file__).resolve().parents[3]


def _get_default_assets_root_dir() -> Path:
    """获取默认资产根目录（项目根目录下的 assets）。"""
    return _get_project_root() / "assets"


def _resolve_root_dir(root_dir: str | Path | None, default_root_dir: Path) -> Path:
    """解析根目录参数。"""
    if root_dir is None:
        return default_root_dir
    candidate = Path(root_dir)
    return candidate if candidate.is_absolute() else _get_project_root() / candidate


def _get_default_model_root_dir() -> Path:
    """获取默认模型根目录（assets/models）。"""
    return _get_default_assets_root_dir() / "models"


def _get_default_dataset_root_dir() -> Path:
    """获取默认数据根目录（assets/data/classification）。"""
    return _get_default_assets_root_dir() / "data" / "classification"


@dataclass(slots=True)
class DatasetArtifactLayout:
    """描述一个 dataset_version 对应的数据资产目录布局。"""

    task_name: str
    dataset_version: str
    root_dir: Path
    task_dir: Path
    version_dir: Path
    raw_dir: Path
    stage_dir: Path
    final_dir: Path
    metadata_path: Path


@dataclass(slots=True)
class ModelArtifactLayout:
    """描述一个 run_id 对应的模型产物目录布局。"""

    backend: str
    task_name: str
    run_id: str
    root_dir: Path
    backend_dir: Path
    classification_root_dir: Path
    task_dir: Path
    version_dir: Path
    initial_model_dir: Path
    lora_dir: Path
    checkpoints_dir: Path
    merged_model_dir: Path
    eval_dir: Path
    train_metrics_csv: Path
    # validation_label_metrics_csv: Path  # 暂时禁用
    train_loss_plot_png: Path
    train_eval_metrics_plot_png: Path
    test_metrics_csv: Path
    test_label_metrics_csv: Path
    test_predictions_csv: Path
    eval_report_xlsx: Path
    metadata_path: Path


class ModelPathManager:
    """统一管理模型、checkpoint、LoRA 头和版本目录。"""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = _resolve_root_dir(root_dir, _get_default_model_root_dir())

    def get_backend_dir(self, backend: str) -> Path:
        """获取某个后端的根目录。"""
        return self.root_dir / backend

    def get_classification_root_dir(self, backend: str) -> Path:
        """获取某个后端下分类任务的根目录。"""
        return self.get_backend_dir(backend) / "classification"

    def get_task_dir(self, backend: str, task_name: str) -> Path:
        """获取某个任务的根目录。"""
        return self.get_classification_root_dir(backend) / task_name

    def get_version_dir(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取某个 run_id 对应的版本目录。"""
        return self.get_task_dir(backend, task_name) / run_id

    def get_initial_model_dir(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取初始模型目录。"""
        return self.get_version_dir(backend, task_name, run_id) / INITIAL_MODEL_DIRNAME

    def get_lora_dir(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取 LoRA 头目录。"""
        return self.get_version_dir(backend, task_name, run_id) / LORA_DIRNAME

    def get_checkpoints_dir(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取 checkpoint 根目录。"""
        return self.get_version_dir(backend, task_name, run_id) / CHECKPOINTS_DIRNAME

    def get_merged_model_dir(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取合并模型目录。"""
        return self.get_version_dir(backend, task_name, run_id) / MERGED_MODEL_DIRNAME

    def get_eval_dir(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取评测产物目录。"""
        return self.get_version_dir(backend, task_name, run_id) / EVAL_DIRNAME

    def get_train_metrics_csv(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取训练过程指标 CSV。"""
        return self.get_eval_dir(backend, task_name, run_id) / TRAIN_METRICS_FILENAME

    # def get_validation_label_metrics_csv(self, backend: str, task_name: str, run_id: str) -> Path:
    #     """获取训练期 validation 按标签指标 CSV。"""
    #     return self.get_eval_dir(backend, task_name, run_id) / VALIDATION_LABEL_METRICS_FILENAME

    def get_train_loss_plot_png(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取训练 loss 折线图。"""
        return self.get_eval_dir(backend, task_name, run_id) / TRAIN_LOSS_PLOT_FILENAME

    def get_train_eval_metrics_plot_png(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取训练期验证指标折线图。"""
        return self.get_eval_dir(backend, task_name, run_id) / TRAIN_EVAL_METRICS_PLOT_FILENAME

    def get_test_metrics_csv(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取测试指标 CSV。"""
        return self.get_eval_dir(backend, task_name, run_id) / TEST_METRICS_FILENAME

    def get_test_label_metrics_csv(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取测试集按标签指标 CSV。"""
        return self.get_eval_dir(backend, task_name, run_id) / TEST_LABEL_METRICS_FILENAME

    def get_test_predictions_csv(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取测试预测明细 CSV。"""
        return self.get_eval_dir(backend, task_name, run_id) / TEST_PREDICTIONS_FILENAME

    def get_eval_report_xlsx(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取评测汇总 Excel 文件。"""
        return self.get_eval_dir(backend, task_name, run_id) / build_eval_report_filename(run_id)

    def get_metadata_path(self, backend: str, task_name: str, run_id: str) -> Path:
        """获取模型产物元数据文件路径。"""
        return self.get_version_dir(backend, task_name, run_id) / METADATA_FILENAME

    def build_layout(self, backend: str, task_name: str, run_id: str) -> ModelArtifactLayout:
        """构建完整的模型产物目录布局。"""
        backend_dir = self.get_backend_dir(backend)
        classification_root_dir = self.get_classification_root_dir(backend)
        task_dir = self.get_task_dir(backend, task_name)
        version_dir = self.get_version_dir(backend, task_name, run_id)
        return ModelArtifactLayout(
            backend=backend,
            task_name=task_name,
            run_id=run_id,
            root_dir=self.root_dir,
            backend_dir=backend_dir,
            classification_root_dir=classification_root_dir,
            task_dir=task_dir,
            version_dir=version_dir,
            initial_model_dir=version_dir / INITIAL_MODEL_DIRNAME,
            lora_dir=version_dir / LORA_DIRNAME,
            checkpoints_dir=version_dir / CHECKPOINTS_DIRNAME,
            merged_model_dir=version_dir / MERGED_MODEL_DIRNAME,
            eval_dir=version_dir / EVAL_DIRNAME,
            train_metrics_csv=version_dir / EVAL_DIRNAME / TRAIN_METRICS_FILENAME,
            # validation_label_metrics_csv=version_dir / EVAL_DIRNAME / VALIDATION_LABEL_METRICS_FILENAME,  # 暂时禁用
            train_loss_plot_png=version_dir / EVAL_DIRNAME / TRAIN_LOSS_PLOT_FILENAME,
            train_eval_metrics_plot_png=version_dir / EVAL_DIRNAME / TRAIN_EVAL_METRICS_PLOT_FILENAME,
            test_metrics_csv=version_dir / EVAL_DIRNAME / TEST_METRICS_FILENAME,
            test_label_metrics_csv=version_dir / EVAL_DIRNAME / TEST_LABEL_METRICS_FILENAME,
            test_predictions_csv=version_dir / EVAL_DIRNAME / TEST_PREDICTIONS_FILENAME,
            eval_report_xlsx=version_dir / EVAL_DIRNAME / build_eval_report_filename(run_id),
            metadata_path=version_dir / METADATA_FILENAME,
        )

    def ensure_layout(self, backend: str, task_name: str, run_id: str) -> ModelArtifactLayout:
        """创建并返回完整目录布局。"""
        layout = self.build_layout(backend, task_name, run_id)
        layout.backend_dir.mkdir(parents=True, exist_ok=True)
        layout.classification_root_dir.mkdir(parents=True, exist_ok=True)
        layout.task_dir.mkdir(parents=True, exist_ok=True)
        layout.version_dir.mkdir(parents=True, exist_ok=True)
        layout.initial_model_dir.mkdir(parents=True, exist_ok=True)
        layout.lora_dir.mkdir(parents=True, exist_ok=True)
        layout.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        layout.merged_model_dir.mkdir(parents=True, exist_ok=True)
        layout.eval_dir.mkdir(parents=True, exist_ok=True)
        return layout

    def get_checkpoint_dir(self, backend: str, task_name: str, run_id: str, checkpoint_name: str) -> Path:
        """获取某个具体 checkpoint 目录。"""
        return self.get_checkpoints_dir(backend, task_name, run_id) / checkpoint_name


class DatasetPathManager:
    """统一管理微调数据资产目录与版本。"""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = _resolve_root_dir(root_dir, _get_default_dataset_root_dir())

    def get_task_dir(self, task_name: str) -> Path:
        """获取某个任务的数据目录。"""
        return self.root_dir / task_name

    def get_version_dir(self, task_name: str, dataset_version: str) -> Path:
        """获取某个数据版本目录。"""
        return self.get_task_dir(task_name) / dataset_version

    def get_raw_dir(self, task_name: str, dataset_version: str) -> Path:
        """获取原始数据目录。"""
        return self.get_version_dir(task_name, dataset_version) / RAW_DIRNAME

    def get_stage_dir(self, task_name: str, dataset_version: str) -> Path:
        """获取中间处理目录。"""
        return self.get_version_dir(task_name, dataset_version) / STAGE_DIRNAME

    def get_final_dir(self, task_name: str, dataset_version: str) -> Path:
        """获取最终可训练数据目录。"""
        return self.get_version_dir(task_name, dataset_version) / FINAL_DIRNAME

    def get_metadata_path(self, task_name: str, dataset_version: str) -> Path:
        """获取数据版本元数据路径。"""
        return self.get_version_dir(task_name, dataset_version) / METADATA_FILENAME

    def build_layout(self, task_name: str, dataset_version: str) -> DatasetArtifactLayout:
        """构建完整的数据资产目录布局。"""
        version_dir = self.get_version_dir(task_name, dataset_version)
        return DatasetArtifactLayout(
            task_name=task_name,
            dataset_version=dataset_version,
            root_dir=self.root_dir,
            task_dir=self.get_task_dir(task_name),
            version_dir=version_dir,
            raw_dir=version_dir / RAW_DIRNAME,
            stage_dir=version_dir / STAGE_DIRNAME,
            final_dir=version_dir / FINAL_DIRNAME,
            metadata_path=version_dir / METADATA_FILENAME,
        )

    def ensure_layout(self, task_name: str, dataset_version: str) -> DatasetArtifactLayout:
        """创建并返回完整数据资产目录布局。"""
        layout = self.build_layout(task_name, dataset_version)
        layout.task_dir.mkdir(parents=True, exist_ok=True)
        layout.version_dir.mkdir(parents=True, exist_ok=True)
        layout.raw_dir.mkdir(parents=True, exist_ok=True)
        layout.stage_dir.mkdir(parents=True, exist_ok=True)
        layout.final_dir.mkdir(parents=True, exist_ok=True)
        return layout


_model_path_manager: ModelPathManager | None = None
_dataset_path_manager: DatasetPathManager | None = None


def get_model_path_manager() -> ModelPathManager:
    """获取全局的 ModelPathManager 实例。"""
    global _model_path_manager
    if _model_path_manager is None:
        _model_path_manager = ModelPathManager()
    return _model_path_manager


def get_dataset_path_manager() -> DatasetPathManager:
    """获取全局的 DatasetPathManager 实例。"""
    global _dataset_path_manager
    if _dataset_path_manager is None:
        _dataset_path_manager = DatasetPathManager()
    return _dataset_path_manager