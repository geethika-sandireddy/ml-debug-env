from env.tasks import get_task


def grade(task_id: str, applied_fixes: list[str], found_causes: set[str]) -> float:
    task = get_task(task_id)
    required_causes = task["required_causes"]
    required_fixes = task["required_fixes"]

    cause_score = 0.0
    if required_causes:
        discovered = sum(1 for cause in required_causes if cause in found_causes)
        cause_score = 0.4 * (discovered / len(required_causes))

    fix_score = 0.0
    if required_fixes:
        applied = sum(1 for fix in required_fixes if fix in applied_fixes)
        fix_score = 0.6 * (applied / len(required_fixes))

    return round(min(1.0, cause_score + fix_score), 2)
