"""训练与评测后端契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(slots=True)
class LoraConfigSpec:
    """描述一次 LoRA 训练配置。"""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ()
    bias: Literal["none", "all", "lora_only"] = "none"
    modules_to_save: tuple[str, ...] = ()
    use_rslora: bool = False
    use_dora: bool = False


@dataclass(slots=True)
class TrainSpec:
    """描述一次分类训练任务。"""

    task_name: str
    model_name: str | None
    dataset_version: str
    backend: str = "bert"
    model_key: str | None = None
    reasoning_mode: Literal["think", "no_think"] | None = None
    resume_lora: str | None = None
    run_id: str | None = None
    export_dir: Path | None = None
    num_labels: int | None = None
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_train_epochs: int = 3
    max_sequence_length: int = 256
    warmup_ratio: float = 0.0
    seed: int = 42
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalSpec:
    """描述一次分类评测任务。"""

    task_name: str
    run_id: str
    dataset_version: str
    backend: str = "bert"
    artifact_type: str = "merged"
    metric_names: list[str] = field(
        default_factory=lambda: [
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "avg_confidence",
        ]
    )
    batch_size: int = 8
    max_sequence_length: int | None = None
    max_new_tokens: int | None = None
    prompt_engine: Literal["llamafactory", "native"] | None = None
    enable_thinking: bool | None = None
    export_xlsx: bool = True