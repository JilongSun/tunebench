"""LlamaFactory 训练运行生命周期工具。"""

from __future__ import annotations

import json
from pathlib import Path

from tunebench.artifacts import METADATA_FILENAME, build_classification_train_manifest
from tunebench.classification import StructuredTargetDatasetBundle
from tunebench.contracts import TrainSpec


def write_metadata(output_path: Path, payload: dict[str, object]) -> None:
    """写入 metadata JSON。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_lifecycle_manifest(
    *,
    spec: TrainSpec,
    run_id: str,
    dataset_bundle: StructuredTargetDatasetBundle,
    output_dir: Path,
    backend_config: dict[str, object],
    instruction: str,
    train_result: dict[str, float] | None,
    eval_result: dict[str, float] | None = None,
    status: str | None = None,
) -> dict[str, object]:
    """构建训练阶段不同生命周期状态下的统一 manifest。"""
    manifest = build_classification_train_manifest(
        spec=spec,
        run_id=run_id,
        train_file=dataset_bundle.train_file,
        validation_file=dataset_bundle.validation_file,
        train_records=dataset_bundle.train_records,
        validation_records=dataset_bundle.validation_records,
        label_to_id=dataset_bundle.label_to_id,
        output_dir=output_dir,
        device="llamafactory-cli",
        backend_config=backend_config,
        instruction=instruction,
        train_result=train_result,
        eval_result=eval_result,
    )
    if status is not None:
        manifest["status"] = status
    return manifest


def export_metadata_copy(*, export_dir: Path | None, run_id: str, manifest: dict[str, object]) -> Path | None:
    """按需额外导出一份 metadata 副本。"""
    if export_dir is None:
        return None
    export_metadata_path = export_dir / f"{run_id}_{METADATA_FILENAME}"
    write_metadata(export_metadata_path, manifest)
    return export_metadata_path


__all__ = [
    "build_lifecycle_manifest",
    "export_metadata_copy",
    "write_metadata",
]
