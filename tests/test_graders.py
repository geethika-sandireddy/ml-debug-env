from env.graders import grade
from env.tasks import get_task


def test_blind_fix_is_capped_without_diagnosis() -> None:
    task = get_task("task_3")
    blind_fix_score = grade(
        task_id="task_3",
        applied_fixes=task["required_fixes"],
        found_causes=set(),
        evidence=set(),
        steps=4,
        premature_required_fixes=0,
        wrong_fix_attempts=0,
    )
    assert blind_fix_score <= 0.55


def test_full_diagnosis_and_fixes_scores_high() -> None:
    task = get_task("task_3")
    score = grade(
        task_id="task_3",
        applied_fixes=task["required_fixes"],
        found_causes=set(task["required_causes"]),
        evidence=set(task["required_evidence"]),
        steps=12,
        premature_required_fixes=0,
        wrong_fix_attempts=0,
    )
    assert score >= 0.9


def test_wrong_fixes_and_attempts_are_penalized() -> None:
    task = get_task("task_1")
    clean_score = grade(
        task_id="task_1",
        applied_fixes=task["required_fixes"],
        found_causes=set(task["required_causes"]),
        evidence=set(task["required_evidence"]),
        steps=4,
        premature_required_fixes=0,
        wrong_fix_attempts=0,
    )
    noisy_score = grade(
        task_id="task_1",
        applied_fixes=task["required_fixes"] + ["totally_wrong_fix"],
        found_causes=set(task["required_causes"]),
        evidence=set(task["required_evidence"]),
        steps=4,
        premature_required_fixes=0,
        wrong_fix_attempts=2,
    )
    assert noisy_score < clean_score
