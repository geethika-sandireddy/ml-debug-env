from copy import deepcopy

TASKS = {
    "task_1": {
        "id": "task_1",
        "name": "Validation Transform Bug",
        "difficulty": "easy",
        "description": (
            "Train accuracy is strong but validation accuracy is stuck. "
            "Find the transform mismatch and repair it."
        ),
        "required_causes": ["missing_val_normalize"],
        "required_fixes": ["add_normalize_to_val_transform"],
        "required_evidence": [
            "ev_val_transform_no_normalize",
            "ev_augmentation_trace_preproc_mismatch",
        ],
        "max_steps": 8,
    },
    "task_2": {
        "id": "task_2",
        "name": "Double Softmax",
        "difficulty": "medium",
        "description": (
            "Loss decreases while accuracy decreases. "
            "Inspect the loss path and remove the redundant softmax."
        ),
        "required_causes": ["double_softmax_before_cross_entropy"],
        "required_fixes": ["remove_redundant_softmax_before_cross_entropy"],
        "required_evidence": [
            "ev_loss_log_softmax_before_ce",
            "ev_loss_function_expects_logits",
        ],
        "max_steps": 8,
    },
    "task_3": {
        "id": "task_3",
        "name": "Silent Generalization Failure",
        "difficulty": "hard",
        "description": (
            "Train and validation look strong while test accuracy collapses. "
            "Uncover all hidden data and evaluation issues."
        ),
        "required_causes": [
            "subject_leakage_between_splits",
            "severe_class_imbalance",
            "wrong_primary_metric",
        ],
        "required_fixes": [
            "enforce_subject_wise_dataset_split",
            "add_weighted_sampler_or_class_weighting",
            "report_macro_f1_instead_of_accuracy",
        ],
        "required_evidence": [
            "ev_patient_leak_from_data_pipeline",
            "ev_split_strategy_patient_level",
            "ev_class_distribution_imbalance",
            "ev_minority_recall_low",
            "ev_eval_metric_accuracy_only",
            "ev_eval_log_dashboard_misleading",
        ],
        "max_steps": 12,
    },
}


def get_task(task_id: str) -> dict:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id '{task_id}'")
    return deepcopy(TASKS[task_id])


def list_tasks() -> list[dict]:
    return [deepcopy(TASKS[task_id]) for task_id in sorted(TASKS)]


def get_tasks() -> list[dict]:
    return list_tasks()


def get_action_schema() -> dict:
    return {
        "type": "object",
        "required": ["action_type"],
        "properties": {
            "action_type": {
                "type": "string",
                "enum": ["read_log", "check_metric", "inspect_config", "propose_fix", "apply_fix"],
            },
            "target": {"type": "string"},
            "value": {"type": ["string", "null"]},
        },
    }
