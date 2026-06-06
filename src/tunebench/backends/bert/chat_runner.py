"""BERT 分类单条推理后端。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

from tunebench.artifacts import ModelArtifactLayout, ModelPathManager, get_model_path_manager
from tunebench.contracts import ChatResult, ChatSpec, RunPlan
from tunebench.util import get_logger


logger = get_logger("backends.bert.chat_runner")

_BERT_BACKEND = "bert"


class BertClassificationChatRunner:
    """负责当前 BERT 分类模型的单条推理。"""

    def __init__(self, model_path_manager: ModelPathManager | None = None) -> None:
        self.model_path_manager = model_path_manager or get_model_path_manager()

    def _load_metadata(self, metadata_path: Path) -> dict[str, Any]:
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _resolve_tokenizer_source(self, model_dir: Path, model_name: str) -> str:
        if (model_dir / "tokenizer_config.json").exists():
            return str(model_dir)
        return model_name

    def _load_model_and_tokenizer(
        self,
        spec: ChatSpec,
        metadata: dict[str, Any],
        label_to_id: dict[str, int],
        model_layout: ModelArtifactLayout,
    ) -> tuple[Any, PreTrainedTokenizerBase]:
        model_name = str(metadata["model_name"])

        if spec.artifact_type == "merged":
            tokenizer = cast(
                PreTrainedTokenizerBase,
                AutoTokenizer.from_pretrained(
                    self._resolve_tokenizer_source(model_layout.merged_model_dir, model_name),
                    use_fast=True,
                ),
            )
            model = AutoModelForSequenceClassification.from_pretrained(str(model_layout.merged_model_dir))
            return model, tokenizer

        if spec.artifact_type == "lora":
            tokenizer = cast(
                PreTrainedTokenizerBase,
                AutoTokenizer.from_pretrained(
                    self._resolve_tokenizer_source(model_layout.lora_dir, model_name),
                    use_fast=True,
                ),
            )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(label_to_id),
                id2label={index: label for label, index in label_to_id.items()},
                label2id=label_to_id,
            )
            model = PeftModel.from_pretrained(base_model, str(model_layout.lora_dir))
            return model, tokenizer

        raise ValueError(f"artifact_type={spec.artifact_type} 非法，仅支持 merged 或 lora。")

    def build_plan(self, spec: ChatSpec) -> RunPlan:
        if not (spec.task_name and spec.run_id):
            raise ValueError("task_name 和 run_id 不能为空。")
        model_layout = self.model_path_manager.build_layout(_BERT_BACKEND, spec.task_name, spec.run_id)
        return RunPlan(
            stage="chat",
            summary="执行 BERT 分类模型的单条推理。",
            inputs=asdict(spec),
            outputs={
                "metadata": str(model_layout.metadata_path),
                "merged_model_dir": str(model_layout.merged_model_dir),
                "lora_dir": str(model_layout.lora_dir),
            },
            notes=[
                "当前命令会读取训练 metadata 里的 label_to_id，并将预测类别 id 映射回标签原文。",
                "bert 后端只执行单条分类推理，不生成自由文本。",
            ],
        )

    def run(self, spec: ChatSpec) -> ChatResult:
        try:
            logger.info(
                "开始 BERT chat 推理: task=%s, run_id=%s, artifact_type=%s",
                spec.task_name,
                spec.run_id,
                spec.artifact_type,
            )
            if not (spec.task_name and spec.run_id):
                raise ValueError("task_name 和 run_id 不能为空。")
            model_layout = self.model_path_manager.build_layout(_BERT_BACKEND, spec.task_name, spec.run_id)
            metadata = self._load_metadata(model_layout.metadata_path)
            label_to_id = {str(key): int(value) for key, value in metadata["label_to_id"].items()}
            id_to_label = {value: key for key, value in label_to_id.items()}

            model, tokenizer = self._load_model_and_tokenizer(spec, metadata, label_to_id, model_layout)
            model.eval()

            encoded = tokenizer(
                spec.message,
                return_tensors="pt",
                truncation=True,
                max_length=spec.max_sequence_length,
            )

            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.inference_mode():
                outputs = model(**encoded)

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_index = int(torch.argmax(probabilities, dim=-1).item())
            confidence = float(probabilities[0, predicted_index].item())
            output_text = id_to_label[predicted_index]
            payload = {
                "backend": spec.backend,
                "task_name": spec.task_name,
                "run_id": spec.run_id,
                "artifact_type": spec.artifact_type,
                "message": spec.message,
                "output_text": output_text,
                "confidence": confidence,
            }
            return ChatResult(
                stage="chat",
                success=True,
                message="BERT 单条推理完成。",
                output_text=output_text,
                payload=payload,
            )
        except Exception as exc:
            return ChatResult(
                stage="chat",
                success=False,
                message=f"BERT 单条推理失败: {exc}",
            )