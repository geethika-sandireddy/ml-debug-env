from env.tasks import get_task


def grade(
    task_id: str,
    applied_fixes: list[str],
    found_causes: set[str],
    evidence: set[str] | None = None,
    steps: int | None = None,
    premature_required_fixes: int = 0,
) -> float:
    task = get_task(task_id)
    required_causes = task["required_causes"]
    required_fixes = task["required_fixes"]
    required_evidence = task.get("required_evidence", [])
    evidence = evidence or set()

    discovered_causes = sum(1 for cause in required_causes if cause in found_causes)
    diagnosis_quality = (discovered_causes / len(required_causes)) if required_causes else 0.0

    discovered_evidence = sum(1 for ev in required_evidence if ev in evidence)
    evidence_quality = (discovered_evidence / len(required_evidence)) if required_evidence else 0.0

    required_applied = sum(1 for fix in required_fixes if fix in applied_fixes)
    required_fix_ratio = (required_applied / len(required_fixes)) if required_fixes else 0.0

    wrong_applied = sum(1 for fix in applied_fixes if fix not in required_fixes)
    wrong_penalty = 0.15 * (wrong_applied / max(1, len(required_fixes)))
    fix_quality = max(0.0, min(1.0, required_fix_ratio - wrong_penalty))

    if steps is None:
        efficiency = 1.0
    else:
        max_steps = task.get("max_steps", 1)
        if max_steps <= 1:
            efficiency = 1.0
        else:
            used_ratio = min(1.0, max(0.0, (steps - 1) / (max_steps - 1)))
            efficiency = max(0.0, 1.0 - 0.35 * used_ratio)

    premature_penalty = 0.1 * min(1.0, premature_required_fixes / max(1, len(required_fixes)))

    score = (
        0.25 * evidence_quality
        + 0.25 * diagnosis_quality
        + 0.35 * fix_quality
        + 0.15 * efficiency
    )
    score = max(0.0, min(1.0, score - premature_penalty))
    return round(score, 2)
