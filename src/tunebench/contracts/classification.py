"""分类任务通用契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DatasetSpec:
    """描述一次分类数据处理任务。"""

    task_name: str
    input_path: Path
    dataset_version: str
    text_key: str
    label_key: str
    output_path: Path | None = None
    output_format: str = "jsonl"
    sheet_name: str | int = 0
    validation_ratio: float = 0.0
    split_seed: int = 42
    is_test: bool = False
    allowed_labels: tuple[str, ...] = ()