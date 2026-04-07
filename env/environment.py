from typing import Any, Literal

from pydantic import BaseModel, Field

from env.graders import grade
from env.simulator import get_task_data
from env.tasks import get_task, list_tasks


class Action(BaseModel):
    action_type: Literal["read_log", "check_metric", "inspect_config", "propose_fix", "apply_fix"]
    target: str = ""
    value: str | None = None


class Observation(BaseModel):
    task_id: str
    description: str
    difficulty: str
    visible_logs: dict[str, str] = Field(default_factory=dict)
    visible_metrics: dict[str, str] = Field(default_factory=dict)
    visible_config: dict[str, str] = Field(default_factory=dict)
    step_count: int = 0
    pending_fix: str | None = None
    applied_fixes: list[str] = Field(default_factory=list)
    message: str = ""
    done: bool = False


class Reward(BaseModel):
    value: float
    reason: str


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class MLDebugEnv:
    def __init__(self) -> None:
        self.available_tasks = list_tasks()
        self.task_id: str | None = None
        self.task: dict[str, Any] | None = None
        self.task_data: dict[str, Any] | None = None
        self.step_count = 0
        self.pending_fix: str | None = None
        self.applied_fixes: list[str] = []
        self.found_causes: set[str] = set()
        self.evidence: set[str] = set()
        self.premature_required_fixes = 0
        self.wrong_fix_attempts = 0
        self.visible_logs: dict[str, str] = {}
        self.visible_metrics: dict[str, str] = {}
        self.visible_config: dict[str, str] = {}
        self.done = False
        self.message = ""
        self.last_action_error: str | None = None
        self._seen_actions: set[tuple[str, str]] = set()
        self.closed = False

    def reset(self, task_id: str | None = None) -> Observation:
        self.closed = False
        chosen = task_id or self.available_tasks[0]["id"]
        self.task = get_task(chosen)
        self.task_data = get_task_data(chosen)
        self.task_id = chosen
        self.step_count = 0
        self.pending_fix = None
        self.applied_fixes = []
        self.found_causes = set()
        self.evidence = set()
        self.premature_required_fixes = 0
        self.wrong_fix_attempts = 0
        self.visible_logs = {}
        self.visible_metrics = {}
        self.visible_config = {}
        self.done = False
        self.message = "Episode reset. Inspect logs, metrics, and configs to diagnose the failure."
        self.last_action_error = None
        self._seen_actions = set()
        return self.observation()

    def observation(self) -> Observation:
        if not self.task:
            raise RuntimeError("Environment has not been reset.")
        return Observation(
            task_id=self.task["id"],
            description=self.task["description"],
            difficulty=self.task["difficulty"],
            visible_logs=self.visible_logs,
            visible_metrics=self.visible_metrics,
            visible_config=self.visible_config,
            step_count=self.step_count,
            pending_fix=self.pending_fix,
            applied_fixes=self.applied_fixes,
            message=self.message,
            done=self.done,
        )

    def state(self) -> dict[str, Any]:
        if not self.task:
            return {
                "task_id": None,
                "done": False,
                "message": "Call /reset before stepping through the environment.",
            }

        return {
            "task_id": self.task["id"],
            "task_name": self.task["name"],
            "description": self.task["description"],
            "difficulty": self.task["difficulty"],
            "step_count": self.step_count,
            "max_steps": self.task["max_steps"],
            "pending_fix": self.pending_fix,
            "applied_fixes": list(self.applied_fixes),
            "found_causes": sorted(self.found_causes),
            "visible_logs": self.visible_logs,
            "visible_metrics": self.visible_metrics,
            "visible_config": self.visible_config,
            "done": self.done,
            "message": self.message,
            "last_action_error": self.last_action_error,
            "current_score": grade(
                task_id=self.task["id"],
                applied_fixes=self.applied_fixes,
                found_causes=self.found_causes,
                evidence=self.evidence,
                premature_required_fixes=self.premature_required_fixes,
                wrong_fix_attempts=self.wrong_fix_attempts,
                steps=self.step_count,
            ),
        }

    def step(self, action: Action) -> StepResult:
        if self.closed:
            raise RuntimeError("Environment is closed. Call reset() to start a new episode.")
        if not self.task or not self.task_data or not self.task_id:
            raise RuntimeError("Environment has not been reset.")

        if self.done:
            return StepResult(
                observation=self.observation(),
                reward=Reward(value=0.0, reason="Episode already finished."),
                done=True,
                info={"last_action_error": "episode_already_done"},
            )

        self.step_count += 1
        self.last_action_error = None
        reward_value = 0.0
        reward_reason = "No useful progress."
        action_key = (action.action_type, action.target)
        STEP_COST = 0.01

        if action_key in self._seen_actions and action.action_type != "apply_fix":
            self.last_action_error = "repeated_action"
            self.message = "That action was already taken and revealed nothing new."
            reward_reason = "Repeated action penalty."
        else:
            self._seen_actions.add(action_key)
            reward_value, reward_reason = self._apply_action(action)

        # Slightly lower reward later in the episode (keeps trajectories from dragging).
        if reward_value > 0.0:
            efficiency = max(0.6, 1.0 - 0.03 * (self.step_count - 1))
            reward_value *= efficiency
        reward_value = max(0.0, reward_value - STEP_COST)

        if self.step_count >= self.task["max_steps"] and not self.done:
            self.done = True
            self.message = "Step budget exhausted before all fixes were applied."

        observation = self.observation()
        return StepResult(
            observation=observation,
            reward=Reward(value=round(max(0.0, min(1.0, reward_value)), 2), reason=reward_reason),
            done=observation.done,
            info={
                "last_action_error": self.last_action_error,
                "current_score": grade(
                    task_id=self.task_id,
                    applied_fixes=self.applied_fixes,
                    found_causes=self.found_causes,
                    evidence=self.evidence,
                    premature_required_fixes=self.premature_required_fixes,
                    wrong_fix_attempts=self.wrong_fix_attempts,
                    steps=self.step_count,
                ),
            },
        )

    def close(self) -> None:
        self.closed = True

    def _apply_action(self, action: Action) -> tuple[float, str]:
        if action.action_type == "read_log":
            return self._reveal("logs", action.target)
        if action.action_type == "check_metric":
            return self._reveal("metrics", action.target)
        if action.action_type == "inspect_config":
            return self._reveal("configs", action.target)
        if action.action_type == "propose_fix":
            return self._propose_fix(action.target)
        if action.action_type == "apply_fix":
            return self._apply_fix()
        self.last_action_error = "unknown_action"
        self.message = f"Unsupported action '{action.action_type}'."
        return 0.0, "Unsupported action."

    def _reveal(self, bucket: str, target: str) -> tuple[float, str]:
        store_map = {
            "logs": self.visible_logs,
            "metrics": self.visible_metrics,
            "configs": self.visible_config,
        }
        source = self.task_data[bucket]
        if target not in source:
            self.last_action_error = f"unknown_{bucket[:-1]}"
            self.message = f"{target} is not a valid {bucket[:-1]} target for this task."
            return 0.0, "Invalid target."

        store_map[bucket][target] = source[target]
        reward = 0.05
        reason = f"Revealed {bucket[:-1]} '{target}'."

        action_type = "read_log" if bucket == "logs" else "check_metric" if bucket == "metrics" else "inspect_config"
        evidence_ids = self.task_data.get("evidence_triggers", {}).get((action_type, target), [])
        for evidence_id in evidence_ids:
            if evidence_id not in self.evidence:
                self.evidence.add(evidence_id)
                reward += 0.05
                reason = f"Added evidence '{evidence_id}'."

        for cause_id, required_evidence in self.task_data.get("cause_requirements", {}).items():
            if cause_id in self.found_causes:
                continue
            if all(req in self.evidence for req in required_evidence):
                self.found_causes.add(cause_id)
                reward += 0.2
                reason = f"Confirmed root cause '{cause_id}'."

        self.message = reason
        if self._all_fixes_applied():
            self.done = True
        return reward, reason

    def _propose_fix(self, fix_id: str) -> tuple[float, str]:
        fixes = self.task_data["fixes"]
        if fix_id not in fixes:
            self.last_action_error = "unknown_fix"
            self.message = f"{fix_id} is not a recognized fix for this task."
            return 0.0, "Invalid fix."

        if fix_id in self.applied_fixes:
            self.last_action_error = "fix_already_applied"
            self.message = f"{fix_id} was already applied."
            return 0.0, "Fix already applied."

        required_fixes = self.task["required_fixes"] if self.task else []
        is_required_fix = fix_id in required_fixes
        addressed_causes = fixes[fix_id].get("addresses", [])
        missing_causes = [cause for cause in addressed_causes if cause not in self.found_causes]

        if not is_required_fix:
            self.last_action_error = "wrong_fix"
            self.wrong_fix_attempts += 1
            self.pending_fix = None
            self.message = f"Queued wrong fix '{fix_id}' (not required for this task)."
            return 0.0, self.message

        if missing_causes:
            self.last_action_error = "insufficient_evidence"
            self.pending_fix = None
            self.message = f"Queued required fix '{fix_id}', but evidence is missing."
            return 0.0, self.message

        self.pending_fix = fix_id
        self.last_action_error = None
        self.message = f"Queued required fix '{fix_id}' for application."
        return 0.15, self.message

    def _apply_fix(self) -> tuple[float, str]:
        if not self.pending_fix:
            self.last_action_error = "no_pending_fix"
            self.message = "You need to propose a fix before applying one."
            return 0.0, "No pending fix."

        fix_id = self.pending_fix
        self.pending_fix = None
        if fix_id in self.applied_fixes:
            self.last_action_error = "fix_already_applied"
            self.message = f"{fix_id} was already applied."
            return 0.0, "Fix already applied."

        self.applied_fixes.append(fix_id)
        required_fixes = self.task["required_fixes"] if self.task else []
        is_required_fix = fix_id in required_fixes
        addressed_causes = self.task_data["fixes"].get(fix_id, {}).get("addresses", [])
        missing_causes_now = [cause for cause in addressed_causes if cause not in self.found_causes]

        if not is_required_fix:
            self.last_action_error = "wrong_fix_applied"
            self.wrong_fix_attempts += 1
            reward = 0.0
            self.message = f"Applied wrong fix '{fix_id}'."
        elif missing_causes_now:
            self.last_action_error = "premature_fix"
            self.premature_required_fixes += 1
            reward = 0.15
            self.message = f"Applied required fix '{fix_id}' early (missing evidence)."
        else:
            self.last_action_error = None
            reward = 0.45
            self.message = f"Applied correct fix '{fix_id}'."

        if self._all_fixes_applied():
            self.done = True
            reward = 0.55
            self.message = "All required fixes applied. Episode solved."

        return reward, self.message

    def _all_fixes_applied(self) -> bool:
        if not self.task:
            return False
        required_fixes = self.task["required_fixes"]
        required_causes = self.task["required_causes"]
        required_evidence = self.task.get("required_evidence", [])
        return (
            all(fix in self.applied_fixes for fix in required_fixes)
            and all(cause in self.found_causes for cause in required_causes)
            and all(ev in self.evidence for ev in required_evidence)
        )
