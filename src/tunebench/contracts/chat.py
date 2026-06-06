"""聊天推理契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ChatSpec:
    """描述一次单轮聊天或单条推理请求。"""

    message: str
    task_name: str | None = None
    run_id: str | None = None
    backend: str = "bert"
    artifact_type: str = "merged"
    external_model_path: str | None = None
    prompt_engine: str | None = None
    max_sequence_length: int | None = None
    instruction: str | None = None
    template_name: str | None = None
    reasoning_mode: str | None = None
    reasoning_suffix_style: str | None = None
    enable_thinking: bool | None = None
    max_new_tokens: int | None = None
    extra_args: dict[str, Any] | None = None


@dataclass(slots=True)
class ChatResult:
    """描述一次聊天或推理输出。"""

    stage: str
    success: bool
    message: str
    output_text: str = ""
    payload: dict[str, Any] | None = None