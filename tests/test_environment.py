"""Direct unit tests for MLDebugEnv (env/environment.py).

These exercise the environment's state machine in isolation, independent of
the HTTP layer covered in test_api.py. Task 1 (the easiest, two-evidence,
one-fix task) is used as the fixture task throughout.
"""

import pytest

from env.environment import Action, MLDebugEnv


@pytest.fixture
def env() -> MLDebugEnv:
    e = MLDebugEnv()
    e.reset(task_id="task_1")
    return e


def test_reset_returns_clean_observation(env: MLDebugEnv) -> None:
    obs = env.observation()
    assert obs.task_id == "task_1"
    assert obs.step_count == 0
    assert obs.done is False
    assert obs.applied_fixes == []
    assert obs.visible_logs == {}


def test_step_before_reset_raises() -> None:
    fresh = MLDebugEnv()
    with pytest.raises(RuntimeError):
        fresh.step(Action(action_type="read_log", target="training_metrics"))


def test_step_on_closed_env_raises(env: MLDebugEnv) -> None:
    env.close()
    with pytest.raises(RuntimeError):
        env.step(Action(action_type="read_log", target="training_metrics"))


def test_reset_after_close_reopens_env(env: MLDebugEnv) -> None:
    env.close()
    obs = env.reset(task_id="task_1")
    assert env.closed is False
    assert obs.done is False
    # Should not raise now that the env has been reset again.
    result = env.step(Action(action_type="read_log", target="training_metrics"))
    assert result.done is False


def test_invalid_target_is_rejected(env: MLDebugEnv) -> None:
    result = env.step(Action(action_type="read_log", target="does_not_exist"))
    assert result.info["last_action_error"] == "unknown_log"
    assert result.reward.value == 0.0
    assert env.visible_logs == {}


def test_unknown_action_type_is_rejected() -> None:
    """action_type is a Literal, so pydantic itself should refuse bad values."""
    with pytest.raises(ValueError):
        Action(action_type="delete_everything", target="x")


def test_repeated_action_is_penalized(env: MLDebugEnv) -> None:
    action = Action(action_type="read_log", target="training_metrics")
    first = env.step(action)
    assert first.reward.value > 0.0

    second = env.step(action)
    assert second.info["last_action_error"] == "repeated_action"
    assert second.reward.value == 0.0


def test_propose_fix_without_evidence_is_rejected(env: MLDebugEnv) -> None:
    result = env.step(
        Action(action_type="propose_fix", target="add_normalize_to_val_transform")
    )
    assert result.info["last_action_error"] == "insufficient_evidence"
    assert env.pending_fix is None


def test_propose_fix_not_required_is_wrong_fix(env: MLDebugEnv) -> None:
    # "reduce_lr_schedule" is a real, valid fix in task_1's data (a tempting
    # distractor) but is not one of the task's required_fixes.
    result = env.step(Action(action_type="propose_fix", target="reduce_lr_schedule"))
    assert result.info["last_action_error"] == "wrong_fix"
    assert env.wrong_fix_attempts == 1


def test_propose_fix_unknown_to_task_is_unknown_fix(env: MLDebugEnv) -> None:
    result = env.step(Action(action_type="propose_fix", target="totally_unrelated_fix"))
    assert result.info["last_action_error"] == "unknown_fix"


def test_apply_fix_without_pending_fix_is_rejected(env: MLDebugEnv) -> None:
    result = env.step(Action(action_type="apply_fix"))
    assert result.info["last_action_error"] == "no_pending_fix"
    assert result.reward.value == 0.0


def test_full_solve_path_completes_task_1(env: MLDebugEnv) -> None:
    steps = [
        Action(action_type="inspect_config", target="val_transform"),
        Action(action_type="read_log", target="augmentation_trace"),
        Action(action_type="propose_fix", target="add_normalize_to_val_transform"),
        Action(action_type="apply_fix"),
    ]
    result = None
    for action in steps:
        result = env.step(action)

    assert result is not None
    assert result.done is True
    assert env.applied_fixes == ["add_normalize_to_val_transform"]
    assert "missing_val_normalize" in env.found_causes


def test_episode_ends_at_step_budget(env: MLDebugEnv) -> None:
    max_steps = env.task["max_steps"]
    result = None
    for i in range(max_steps):
        # Alternate targets so the "repeated action" guard doesn't mask the
        # step-budget behavior we're actually testing here.
        result = env.step(Action(action_type="inspect_config", target=f"nonexistent_{i}"))

    assert result is not None
    assert result.done is True
    assert env.step_count == max_steps


def test_step_after_done_is_a_noop(env: MLDebugEnv) -> None:
    steps = [
        Action(action_type="inspect_config", target="val_transform"),
        Action(action_type="read_log", target="augmentation_trace"),
        Action(action_type="propose_fix", target="add_normalize_to_val_transform"),
        Action(action_type="apply_fix"),
    ]
    for action in steps:
        env.step(action)
    assert env.done is True

    step_count_before = env.step_count
    result = env.step(Action(action_type="read_log", target="training_metrics"))
    assert result.info["last_action_error"] == "episode_already_done"
    assert result.reward.value == 0.0
    # step_count should not advance once the episode is finished.
    assert env.step_count == step_count_before


def test_state_reflects_current_score(env: MLDebugEnv) -> None:
    # Before any evidence/diagnosis, only the (undiscounted, zero-step)
    # efficiency term contributes, so the baseline score is 0.1, not 0.0 --
    # see the 0.10 * efficiency term in env/graders.py:grade().
    state = env.state()
    assert state["task_id"] == "task_1"
    assert state["current_score"] == 0.1

    env.step(Action(action_type="inspect_config", target="val_transform"))
    updated = env.state()
    assert updated["current_score"] > state["current_score"]
