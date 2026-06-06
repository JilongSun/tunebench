"""分类任务后端注册表。"""

from __future__ import annotations

from dataclasses import dataclass

from tunebench.backends.bert import BertClassificationChatRunner, BertClassificationEvalRunner, BertClassificationTrainRunner
from tunebench.backends.llamafactory import (
    LlamaFactoryClassificationChatRunner,
    LlamaFactoryClassificationEvalRunner,
    LlamaFactoryClassificationTrainRunner,
)
from tunebench.contracts import ChatResult, ChatSpec, EvalSpec, RunPlan, StageResult, TrainSpec

from .base import ClassificationBackend


@dataclass(slots=True)
class _BertClassificationBackend:
    """基于当前 BERT 实现的分类后端适配器。"""

    chatter: BertClassificationChatRunner
    trainer: BertClassificationTrainRunner
    evaluator: BertClassificationEvalRunner
    name: str = "bert"

    def build_train_plan(self, spec: TrainSpec) -> RunPlan:
        return self.trainer.build_plan(spec)

    def run_train(self, spec: TrainSpec) -> StageResult:
        return self.trainer.run(spec)

    def build_evaluate_plan(self, spec: EvalSpec) -> RunPlan:
        return self.evaluator.build_plan(spec)

    def run_evaluate(self, spec: EvalSpec) -> StageResult:
        return self.evaluator.run(spec)

    def build_chat_plan(self, spec: ChatSpec) -> RunPlan:
        return self.chatter.build_plan(spec)

    def run_chat(self, spec: ChatSpec) -> ChatResult:
        return self.chatter.run(spec)


def _build_bert_backend() -> ClassificationBackend:
    return _BertClassificationBackend(
        chatter=BertClassificationChatRunner(),
        trainer=BertClassificationTrainRunner(),
        evaluator=BertClassificationEvalRunner(),
    )


@dataclass(slots=True)
class _LlamaFactoryClassificationBackend:
    """LlamaFactory 分类后端适配器。"""

    chatter: LlamaFactoryClassificationChatRunner
    trainer: LlamaFactoryClassificationTrainRunner
    evaluator: LlamaFactoryClassificationEvalRunner
    name: str = "llamafactory"

    def build_train_plan(self, spec: TrainSpec) -> RunPlan:
        return self.trainer.build_plan(spec)

    def run_train(self, spec: TrainSpec) -> StageResult:
        return self.trainer.run(spec)

    def build_evaluate_plan(self, spec: EvalSpec) -> RunPlan:
        return self.evaluator.build_plan(spec)

    def run_evaluate(self, spec: EvalSpec) -> StageResult:
        return self.evaluator.run(spec)

    def build_chat_plan(self, spec: ChatSpec) -> RunPlan:
        return self.chatter.build_plan(spec)

    def run_chat(self, spec: ChatSpec) -> ChatResult:
        return self.chatter.run(spec)


def _build_llamafactory_backend() -> ClassificationBackend:
    return _LlamaFactoryClassificationBackend(
        chatter=LlamaFactoryClassificationChatRunner(),
        trainer=LlamaFactoryClassificationTrainRunner(),
        evaluator=LlamaFactoryClassificationEvalRunner(),
    )


_BACKEND_FACTORIES = {
    "bert": _build_bert_backend,
    "llamafactory": _build_llamafactory_backend,
}


def list_classification_backend_names() -> tuple[str, ...]:
    """返回已注册的分类后端名称。"""
    return tuple(sorted(_BACKEND_FACTORIES))


def get_classification_backend(name: str) -> ClassificationBackend:
    """按名称解析分类后端。"""
    try:
        factory = _BACKEND_FACTORIES[name]
    except KeyError as exc:
        available_backends = ", ".join(list_classification_backend_names())
        raise ValueError(f"未注册的分类后端: {name}；当前可用后端: {available_backends}") from exc
    return factory()