from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from env import Action, MLDebugEnv, grade
from env.baseline import run_baseline_suite
from env.tasks import list_tasks


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default=None, description="Task to reset into.")


app = FastAPI(
    title="ML Training Run Debugger",
    description=(
        "OpenEnv-style RL environment where an agent debugs failing ML training runs "
        "by inspecting logs, metrics, and configs before proposing and applying fixes."
    ),
    version="1.0.0",
)

env = MLDebugEnv()


@app.get("/")
def health() -> dict:
    return {
        "ok": True,
        "name": "ml-debug-env",
        "task_count": len(list_tasks()),
    }


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()) -> dict:
    observation = env.reset(task_id=request.task_id)
    return observation.model_dump()


@app.post("/step")
def step(action: Action) -> dict:
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state() -> dict:
    return env.state()


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": list_tasks()}


@app.get("/grader")
def grader() -> dict:
    return {
        "task_id": env.task_id,
        "score": grade(
            task_id=env.task_id,
            applied_fixes=env.applied_fixes,
            found_causes=env.found_causes,
        )
        if env.task_id
        else 0.0,
        "found_causes": sorted(env.found_causes),
        "applied_fixes": list(env.applied_fixes),
    }


@app.get("/baseline")
def baseline() -> dict:
    return run_baseline_suite()
