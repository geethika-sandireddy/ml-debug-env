from collections import OrderedDict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from env import Action, MLDebugEnv, grade
from env.baseline import run_baseline_suite
from env.tasks import list_tasks


class ResetRequest(BaseModel):
    task_id: str | None = Field(default=None, description="Task to reset into.")


app = FastAPI(
    title="ML Training Run Debugger",
    description=(
        "OpenEnv-style RL environment where an agent debugs failing ML training runs "
        "by inspecting logs, metrics, and configs before proposing and applying fixes."
    ),
    version="1.0.0",
)

_sessions: "OrderedDict[str, MLDebugEnv]" = OrderedDict()
MAX_SESSIONS = 256
# Stateless HTTP clients (no X-Session-Id, no cookie) share this bucket so /reset then /step works.
DEFAULT_SESSION_ID = "default"


def _get_env(session_id: str) -> MLDebugEnv:
    if session_id in _sessions:
        env = _sessions[session_id]
        _sessions.move_to_end(session_id)
        return env
    if len(_sessions) >= MAX_SESSIONS:
        # LRU eviction: remove least-recently-used session first.
        _sessions.popitem(last=False)
    if session_id not in _sessions:
        _sessions[session_id] = MLDebugEnv()
    return _sessions[session_id]


def _session_id_from_request(request: Request) -> str:
    header_value = request.headers.get("X-Session-Id")
    if header_value:
        return header_value
    cookie_value = request.cookies.get("openenv_session_id")
    if cookie_value:
        return cookie_value
    return ""


def _effective_session_id(request: Request) -> str:
    return _session_id_from_request(request) or DEFAULT_SESSION_ID


@app.get("/")
def health() -> dict:
    return {
        "ok": True,
        "name": "ml-debug-env",
        "task_count": len(list_tasks()),
    }


@app.post("/reset")
def reset(request: Request, body: ResetRequest | None = None) -> JSONResponse:
    body = body or ResetRequest()
    session_id = _session_id_from_request(request) or DEFAULT_SESSION_ID
    env = _get_env(session_id)
    try:
        observation = env.reset(task_id=body.task_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resp = JSONResponse(content=observation.model_dump())
    resp.headers["X-Session-Id"] = session_id
    resp.set_cookie("openenv_session_id", session_id, httponly=True, samesite="lax")
    return resp


@app.post("/step")
def step(request: Request, action: Action) -> dict:
    session_id = _effective_session_id(request)
    env = _get_env(session_id)
    try:
        result = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()


@app.get("/state")
def state(request: Request) -> dict:
    session_id = _effective_session_id(request)
    env = _get_env(session_id)
    return env.state()


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": list_tasks()}


@app.get("/grader")
def grader(request: Request) -> dict:
    session_id = _effective_session_id(request)
    env = _get_env(session_id)
    return {
        "task_id": env.task_id,
        "score": grade(
            task_id=env.task_id,
            applied_fixes=env.applied_fixes,
            found_causes=env.found_causes,
            evidence=env.evidence,
            premature_required_fixes=env.premature_required_fixes,
            wrong_fix_attempts=env.wrong_fix_attempts,
            steps=env.step_count,
        )
        if env.task_id
        else 0.0,
        "found_causes": sorted(env.found_causes),
        "applied_fixes": list(env.applied_fixes),
    }


@app.get("/baseline")
def baseline() -> dict:
    return run_baseline_suite()
