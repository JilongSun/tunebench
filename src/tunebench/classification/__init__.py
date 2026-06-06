"""分类任务公共模块。"""

from .dataset_loader import (
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
    VALIDATION_SPLIT_NAME,
    ClassificationDatasetBundle,
    build_label_mapping,
    load_classification_records,
    load_training_dataset_bundle,
    resolve_optional_split_file,
    resolve_split_file,
    validate_classification_records,
    validate_num_labels,
    validate_validation_labels,
)
from .data_preparer import ClassificationDataPreparer
from .structured_target_builder import (
    ClassificationStructuredTargetBuilder,
    StructuredTargetDatasetBundle,
    load_structured_target_dataset_bundle,
    validate_reasoning_records,
    validate_structured_target_records,
)
from .metrics import (
    compute_classification_metrics,
    compute_classification_metrics_bundle,
    extract_label_metrics_from_flattened,
    flatten_label_metrics,
)
from .train_eval_callback import ClassificationTrainEvalCallback

__all__ = [
    "ClassificationDataPreparer",
    "ClassificationStructuredTargetBuilder",
    "ClassificationDatasetBundle",
    "ClassificationTrainEvalCallback",
    "StructuredTargetDatasetBundle",
    "TEST_SPLIT_NAME",
    "TRAIN_SPLIT_NAME",
    "VALIDATION_SPLIT_NAME",
    "build_label_mapping",
    "compute_classification_metrics",
    "compute_classification_metrics_bundle",
    "extract_label_metrics_from_flattened",
    "flatten_label_metrics",
    "load_classification_records",
    "load_structured_target_dataset_bundle",
    "load_training_dataset_bundle",
    "resolve_optional_split_file",
    "resolve_split_file",
    "validate_classification_records",
    "validate_num_labels",
    "validate_reasoning_records",
    "validate_structured_target_records",
    "validate_validation_labels",
]