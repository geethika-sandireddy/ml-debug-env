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
            "lr_schedule_trace": (
                "Optimizer schedule stays constant for the last few epochs; "
                "validation drift seems preprocessing-related rather than optimization."
            ),
        },
        "metrics": {
            "train_accuracy": "0.90",
            "val_accuracy": "0.10",
            "distribution_shift_hint": "Validation inputs appear differently scaled than training inputs.",
            "train_loss_pattern_hint": "train_loss decreases steadily while val_loss increases (classic preprocessing mismatch signature).",
        },
        "configs": {
            "train_transform": "Compose([Resize(224), ToTensor(), Normalize(mean, std)])",
            "val_transform": "Compose([Resize(224), ToTensor()])",
            "optimizer": "Adam(lr=3e-4)",
            "batch_size": "64",
        },
        "evidence_triggers": {
            ("inspect_config", "val_transform"): ["ev_val_transform_no_normalize"],
            ("read_log", "augmentation_trace"): ["ev_augmentation_trace_preproc_mismatch"],
            # Distractor evidence (looks relevant but doesn't confirm the real cause)
            ("inspect_config", "optimizer"): ["ev_lr_schedule_hint"],
            ("read_log", "lr_schedule_trace"): ["ev_lr_schedule_hint"],
        },
        "cause_requirements": {
            "missing_val_normalize": [
                "ev_val_transform_no_normalize",
                "ev_augmentation_trace_preproc_mismatch",
            ]
        },
        "fixes": {
            "add_normalize_to_val_transform": {
                "description": "Add Normalize(mean, std) to val_transform.",
                "addresses": ["missing_val_normalize"],
            }
            ,
            # Wrong but tempting fixes
            "reduce_lr_schedule": {
                "description": "Lower learning rate and retry validation; assumes optimization issue.",
                "addresses": [],
            },
            "add_label_smoothing_in_val": {
                "description": "Enable label smoothing in validation; assumes calibration problem.",
                "addresses": [],
            },
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
            "gradient_debug_notes": (
                "The loss path shows probabilities being fed into CE; "
                "this can create the appearance of decreasing loss while hurting accuracy."
            ),
        },
        "metrics": {
            "train_accuracy": "0.31",
            "val_accuracy": "0.27",
            "loss_curve": "monotonic decrease",
            "gradient_entropy_hint": "Entropy of outputs drifts toward an overly-confident distribution.",
        },
        "configs": {
            "loss_function": "CrossEntropyLoss(softmax(model(x)), y)",
            "model_head": "Linear(768 -> 10)  # returns logits",
            "label_smoothing": "0.0",
        },
        "evidence_triggers": {
            ("read_log", "loss_computation_log"): ["ev_loss_log_softmax_before_ce"],
            ("inspect_config", "loss_function"): ["ev_loss_function_expects_logits"],
            # Distractor evidence
            ("inspect_config", "label_smoothing"): ["ev_label_smoothing_hint"],
            ("check_metric", "gradient_entropy_hint"): ["ev_gradient_entropy_hint"],
        },
        "cause_requirements": {
            "double_softmax_before_cross_entropy": [
                "ev_loss_log_softmax_before_ce",
                "ev_loss_function_expects_logits",
            ]
        },
        "fixes": {
            "remove_redundant_softmax_before_cross_entropy": {
                "description": "Feed raw logits directly into CrossEntropyLoss.",
                "addresses": ["double_softmax_before_cross_entropy"],
            }
            ,
            # Wrong but tempting fixes
            "apply_label_smoothing_instead": {
                "description": "Increase label smoothing and keep the softmax pipeline unchanged.",
                "addresses": [],
            },
            "increase_batch_size": {
                "description": "Increase batch size; assumes this will stabilize training.",
                "addresses": [],
            },
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
            "confusion_matrix_notes": (
                "Confusion matrix shows minority class predictions collapsing despite strong overall accuracy."
            ),
        },
        "metrics": {
            "train_accuracy": "0.94",
            "val_accuracy": "0.91",
            "test_accuracy": "0.23",
            "class_distribution": "class_0=78%, class_1=15%, class_2=7%",
            "minority_recall_table": "recall(class_1)=0.18, recall(class_2)=0.09; macro recall far below accuracy view",
        },
        "configs": {
            "split_strategy": "Random image-level split with shuffle=True",
            "sampler": "DataLoader(..., shuffle=True)",
            "evaluation_metric": "primary_metric='accuracy'",
            "early_stopping_patience": "5",
        },
        "evidence_triggers": {
            ("read_log", "data_pipeline"): ["ev_patient_leak_from_data_pipeline"],
            ("inspect_config", "split_strategy"): ["ev_split_strategy_patient_level"],
            ("check_metric", "class_distribution"): ["ev_class_distribution_imbalance"],
            ("check_metric", "minority_recall_table"): ["ev_minority_recall_low"],
            ("inspect_config", "evaluation_metric"): ["ev_eval_metric_accuracy_only"],
            ("read_log", "evaluation_log"): ["ev_eval_log_dashboard_misleading"],
            # Distractor evidence
            ("inspect_config", "sampler"): ["ev_sampler_shuffle_hint"],
            ("check_metric", "test_accuracy"): ["ev_test_low_hint"],
            ("read_log", "confusion_matrix_notes"): ["ev_confusion_matrix_hint"],
        },
        "cause_requirements": {
            "subject_leakage_between_splits": [
                "ev_patient_leak_from_data_pipeline",
                "ev_split_strategy_patient_level",
            ],
            "severe_class_imbalance": [
                "ev_class_distribution_imbalance",
                "ev_minority_recall_low",
            ],
            "wrong_primary_metric": [
                "ev_eval_metric_accuracy_only",
                "ev_eval_log_dashboard_misleading",
            ],
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
            # Wrong but tempting fixes
            "increase_epochs": {
                "description": "Train longer and retry; assumes performance drop is just undertraining.",
                "addresses": [],
            },
            "use_smaller_lr": {
                "description": "Lower learning rate and retry; assumes optimization instability.",
                "addresses": [],
            },
        },
    },
}


def get_task_data(task_id: str) -> dict:
    if task_id not in SIMULATED_TASK_DATA:
        raise KeyError(f"Unknown task_id '{task_id}'")
    return deepcopy(SIMULATED_TASK_DATA[task_id])
