"""LlamaFactory 训练产物同步工具。"""

from __future__ import annotations

import shutil

from tunebench.artifacts import ModelArtifactLayout


def sync_trained_adapter_to_lora_dir(model_layout: ModelArtifactLayout) -> None:
    """将 checkpoints 根目录中的最终 adapter 文件同步到 lora 目录。"""
    if not model_layout.checkpoints_dir.exists():
        raise FileNotFoundError(f"训练输出目录不存在: {model_layout.checkpoints_dir}")

    copied_entry_count = 0
    for source_path in model_layout.checkpoints_dir.iterdir():
        if source_path.name.startswith("checkpoint-"):
            continue

        target_path = model_layout.lora_dir / source_path.name
        if source_path.is_dir():
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)
        copied_entry_count += 1

    if copied_entry_count == 0:
        raise RuntimeError(
            "LlamaFactory 训练已完成，但 checkpoints 根目录下未发现可同步到 lora 目录的最终 adapter 文件。"
        )


__all__ = ["sync_trained_adapter_to_lora_dir"]