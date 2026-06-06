"""reasoning 生成环节契约。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ReasoningGenerationSpec:
    """描述一次 reasoning 数据增强任务。"""

    task_name: str
    source_dataset_version: str
    target_dataset_version: str
    teacher_model: str
    endpoint_url: str
    label_profile: str = "l1_5class"
    prompt_version: str = "reasoning_v1"
    api_key_env_var: str = "TUNEBENCH_REASONING_API_KEY"
    max_concurrency: int = 5
    request_timeout_seconds: float = 60.0
    max_attempts: int = 2
    enable_model_verify: bool = False
    resume: bool = False
    sample_limit: int | None = None
    splits: tuple[str, ...] = ("train", "validation")