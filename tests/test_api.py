from fastapi.testclient import TestClient

from env import Action, MLDebugEnv, grade
from main import app


client = TestClient(app)


def test_health_and_tasks_endpoints() -> None:
    health = client.get("/")
    tasks = client.get("/tasks")

    assert health.status_code == 200
    assert tasks.status_code == 200
    assert health.json()["name"] == "ml-debug-env"
    assert len(tasks.json()["tasks"]) == 3


def test_default_session_reset_and_step() -> None:
    reset = client.post("/reset", json={"task_id": "task_1"})
    assert reset.status_code == 200
    session_id = reset.headers["X-Session-Id"]
    assert session_id

    step = client.post(
        "/step",
        headers={"X-Session-Id": session_id},
        json={"action_type": "inspect_config", "target": "val_transform"},
    )
    assert step.status_code == 200
    assert step.json()["observation"]["task_id"] == "task_1"


def test_isolated_sessions_keep_independent_tasks() -> None:
    client.post("/reset", headers={"X-Session-Id": "alpha"}, json={"task_id": "task_1"})
    client.post("/reset", headers={"X-Session-Id": "beta"}, json={"task_id": "task_2"})

    state_alpha = client.get("/state", headers={"X-Session-Id": "alpha"})
    state_beta = client.get("/state", headers={"X-Session-Id": "beta"})

    assert state_alpha.status_code == 200
    assert state_beta.status_code == 200
    assert state_alpha.json()["task_id"] == "task_1"
    assert state_beta.json()["task_id"] == "task_2"


def test_step_before_reset_returns_400() -> None:
    response = client.post(
        "/step",
        headers={"X-Session-Id": "fresh"},
        json={"action_type": "read_log", "target": "training_metrics"},
    )

    assert response.status_code == 400
    assert "reset" in response.json()["detail"].lower()


def test_invalid_task_returns_400() -> None:
    response = client.post("/reset", json={"task_id": "not_a_task"})

    assert response.status_code == 400
    assert "unknown task_id" in response.json()["detail"].lower()


def test_grader_stays_in_bounds() -> None:
    score = grade(
        task_id="task_3",
        applied_fixes=["increase_epochs"],
        found_causes=set(),
        evidence=set(),
        steps=12,
        premature_required_fixes=0,
    )

    assert 0.0 <= score <= 1.0


def test_reference_plan_solves_hard_task() -> None:
    env = MLDebugEnv()
    env.reset(task_id="task_3")

    actions = [
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
    ]

    for action in actions:
        result = env.step(action)

    assert result.done is True
    assert set(env.applied_fixes) == {
        "enforce_subject_wise_dataset_split",
        "add_weighted_sampler_or_class_weighting",
        "report_macro_f1_instead_of_accuracy",
    }
