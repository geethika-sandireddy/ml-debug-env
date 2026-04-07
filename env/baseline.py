from env.environment import Action, MLDebugEnv
from env.graders import grade


TASK_PLANS = {
    "task_1": [
        Action(action_type="inspect_config", target="val_transform"),
        Action(action_type="read_log", target="augmentation_trace"),
        Action(action_type="propose_fix", target="add_normalize_to_val_transform"),
        Action(action_type="apply_fix"),
    ],
    "task_2": [
        Action(action_type="read_log", target="loss_computation_log"),
        Action(action_type="inspect_config", target="loss_function"),
        Action(action_type="propose_fix", target="remove_redundant_softmax_before_cross_entropy"),
        Action(action_type="apply_fix"),
    ],
    "task_3": [
        Action(action_type="read_log", target="data_pipeline"),
        Action(action_type="inspect_config", target="split_strategy"),
        Action(action_type="propose_fix", target="enforce_subject_wise_dataset_split"),
        Action(action_type="apply_fix"),
        Action(action_type="check_metric", target="class_distribution"),
        Action(action_type="check_metric", target="minority_recall_table"),
        Action(action_type="propose_fix", target="add_weighted_sampler_or_class_weighting"),
        Action(action_type="apply_fix"),
        Action(action_type="inspect_config", target="evaluation_metric"),
        Action(action_type="read_log", target="evaluation_log"),
        Action(action_type="propose_fix", target="report_macro_f1_instead_of_accuracy"),
        Action(action_type="apply_fix"),
    ],
}


def select_next_action(env: MLDebugEnv) -> Action:
    if not env.task_id:
        return Action(action_type="read_log", target="training_metrics")

    if env.pending_fix:
        return Action(action_type="apply_fix")

    for action in TASK_PLANS[env.task_id]:
        if action.action_type == "apply_fix":
            continue

        if action.action_type == "read_log" and action.target in env.visible_logs:
            continue
        if action.action_type == "check_metric" and action.target in env.visible_metrics:
            continue
        if action.action_type == "inspect_config" and action.target in env.visible_config:
            continue
        if action.action_type == "propose_fix":
            if action.target in env.applied_fixes:
                continue
        return action

    return Action(action_type="read_log", target="training_metrics")


def run_baseline_suite() -> dict:
    results = []
    for task_id in ("task_1", "task_2", "task_3"):
        env = MLDebugEnv()
        try:
            env.reset(task_id=task_id)
            rewards = []
            while not env.done and env.step_count < env.task["max_steps"]:
                action = select_next_action(env)
                result = env.step(action)
                rewards.append(result.reward.value)

            score = grade(
                task_id=task_id,
                applied_fixes=env.applied_fixes,
                found_causes=env.found_causes,
                evidence=env.evidence,
                steps=env.step_count,
                premature_required_fixes=env.premature_required_fixes,
            wrong_fix_attempts=env.wrong_fix_attempts,
            )
            results.append(
                {
                    "task_id": task_id,
                    "steps": env.step_count,
                    "score": score,
                    "rewards": rewards,
                    "applied_fixes": list(env.applied_fixes),
                    "found_causes": sorted(env.found_causes),
                }
            )
        finally:
            env.close()

    return {"env": "ml-debug-env", "results": results}
