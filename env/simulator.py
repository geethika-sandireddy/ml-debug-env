from copy import deepcopy


SIMULATED_TASK_DATA = {
    "task_1": {
        "logs": {
            "training_metrics": (
                "epoch=12 train_loss=0.28 train_accuracy=0.90 "
                "val_loss=2.31 val_accuracy=0.10"
            ),
            "augmentation_trace": (
                "train pipeline uses RandomCrop, HorizontalFlip, Normalize; "
                "validation pipeline uses Resize, CenterCrop, ToTensor."
            ),
        },
        "metrics": {
            "train_accuracy": "0.90",
            "val_accuracy": "0.10",
            "distribution_shift_hint": "Validation inputs appear differently scaled than training inputs.",
        },
        "configs": {
            "train_transform": "Compose([Resize(224), ToTensor(), Normalize(mean, std)])",
            "val_transform": "Compose([Resize(224), ToTensor()])",
            "optimizer": "Adam(lr=3e-4)",
        },
        "cause_triggers": {
            ("inspect_config", "val_transform"): "missing_val_normalize",
            ("read_log", "augmentation_trace"): "missing_val_normalize",
        },
        "fixes": {
            "add_normalize_to_val_transform": {
                "description": "Add Normalize(mean, std) to val_transform.",
                "addresses": ["missing_val_normalize"],
            }
        },
    },
    "task_2": {
        "logs": {
            "loss_computation_log": (
                "forward: model returns logits -> softmax(logits) -> CrossEntropyLoss(probabilities, labels)"
            ),
            "training_metrics": (
                "epoch=8 loss=0.71 accuracy=0.43; epoch=12 loss=0.52 accuracy=0.31"
            ),
        },
        "metrics": {
            "train_accuracy": "0.31",
            "val_accuracy": "0.27",
            "loss_curve": "monotonic decrease",
        },
        "configs": {
            "loss_function": "CrossEntropyLoss(softmax(model(x)), y)",
            "model_head": "Linear(768 -> 10)  # returns logits",
        },
        "cause_triggers": {
            ("read_log", "loss_computation_log"): "double_softmax_before_cross_entropy",
            ("inspect_config", "loss_function"): "double_softmax_before_cross_entropy",
        },
        "fixes": {
            "remove_redundant_softmax_before_cross_entropy": {
                "description": "Feed raw logits directly into CrossEntropyLoss.",
                "addresses": ["double_softmax_before_cross_entropy"],
            }
        },
    },
    "task_3": {
        "logs": {
            "data_pipeline": (
                "samples are grouped by patient_id, but split is random over images. "
                "The same patient can appear in train and validation."
            ),
            "evaluation_log": (
                "dashboard highlights overall accuracy only; minority classes fluctuate heavily."
            ),
            "test_summary": "train=0.94 val=0.91 test=0.23 with zero runtime exceptions.",
        },
        "metrics": {
            "train_accuracy": "0.94",
            "val_accuracy": "0.91",
            "test_accuracy": "0.23",
            "class_distribution": "class_0=78%, class_1=15%, class_2=7%",
        },
        "configs": {
            "split_strategy": "Random image-level split with shuffle=True",
            "sampler": "DataLoader(..., shuffle=True)",
            "evaluation_metric": "primary_metric='accuracy'",
        },
        "cause_triggers": {
            ("read_log", "data_pipeline"): "subject_leakage_between_splits",
            ("inspect_config", "split_strategy"): "subject_leakage_between_splits",
            ("check_metric", "class_distribution"): "severe_class_imbalance",
            ("inspect_config", "evaluation_metric"): "wrong_primary_metric",
            ("read_log", "evaluation_log"): "wrong_primary_metric",
        },
        "fixes": {
            "enforce_subject_wise_dataset_split": {
                "description": "Split by patient_id so subjects do not leak across train and validation.",
                "addresses": ["subject_leakage_between_splits"],
            },
            "add_weighted_sampler_or_class_weighting": {
                "description": "Handle skewed classes with a weighted sampler or class-weighted loss.",
                "addresses": ["severe_class_imbalance"],
            },
            "report_macro_f1_instead_of_accuracy": {
                "description": "Use macro-F1 as the primary metric for imbalanced evaluation.",
                "addresses": ["wrong_primary_metric"],
            },
        },
    },
}


def get_task_data(task_id: str) -> dict:
    if task_id not in SIMULATED_TASK_DATA:
        raise KeyError(f"Unknown task_id '{task_id}'")
    return deepcopy(SIMULATED_TASK_DATA[task_id])
