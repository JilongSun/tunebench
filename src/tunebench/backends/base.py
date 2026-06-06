"""分类任务后端接口定义。"""

from __future__ import annotations

from typing import Protocol

from tunebench.contracts import (
    ChatResult,
    ChatSpec,
    EvalSpec,
    RunPlan,
    StageResult,
    TrainSpec,
)


class ClassificationBackend(Protocol):
    """定义分类训练与评测后端的最小接口。"""

    name: str

    def build_train_plan(self, spec: TrainSpec) -> RunPlan:
        """生成训练计划。"""
        ...

    def run_train(self, spec: TrainSpec) -> StageResult:
        """执行训练。"""
        ...

    def build_evaluate_plan(self, spec: EvalSpec) -> RunPlan:
        """生成评测计划。"""
        ...

    def run_evaluate(self, spec: EvalSpec) -> StageResult:
        """执行评测。"""
        ...

    def build_chat_plan(self, spec: ChatSpec) -> RunPlan:
        """生成聊天计划。"""
        ...

    def run_chat(self, spec: ChatSpec) -> ChatResult:
        """执行单轮聊天或推理。"""
        ...
