"""Baseline runner: OpenAI client if HF_TOKEN is set, else built-in policy. Logs [START]/[STEP]/[END]."""

import json
import os
from typing import Optional

from openai import OpenAI

from env import Action, MLDebugEnv, grade
from env.baseline import select_next_action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
MAX_STEPS = 12
TEMPERATURE = 0.0


def build_client() -> Optional[OpenAI]:
    if not HF_TOKEN:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def prompt_for_action(observation: dict) -> str:
    return (
        "You are debugging a failing ML training run inside an RL environment.\n"
        "Return exactly one JSON object with keys action_type, target, value.\n"
        "Allowed action_type values: read_log, check_metric, inspect_config, propose_fix, apply_fix.\n"
        "Choose the best next action from the current observation.\n"
        f"Observation:\n{json.dumps(observation, sort_keys=True)}"
    )


def extract_action(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def format_action(action: Action) -> str:
    if action.target:
        return f"{action.action_type}('{action.target}')"
    return f"{action.action_type}()"


def choose_action(client: Optional[OpenAI], env: MLDebugEnv) -> Action:
    if client is None:
        return select_next_action(env)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=120,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior ML engineer. "
                        "Respond with JSON only and no markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt_for_action(env.observation().model_dump()),
                },
            ],
        )
        parsed = extract_action(completion.choices[0].message.content or "")
        if parsed:
            try:
                return Action(**parsed)
            except Exception:
                pass
        return select_next_action(env)
    except Exception:
        return select_next_action(env)


def run_task(task_id: str, client: Optional[OpenAI]) -> float:
    env = MLDebugEnv()
    observation = env.reset(task_id=task_id)
    rewards: list[float] = []

    print(f"[START] task={task_id} env=ml-debug-env model={MODEL_NAME}", flush=True)

    try:
        for step_num in range(1, MAX_STEPS + 1):
            if observation.done:
                break

            action = choose_action(client, env)
            result = env.step(action)
            observation = result.observation
            rewards.append(round(result.reward.value, 2))

            error_value = result.info.get("last_action_error") or "null"
            done_text = "true" if result.done else "false"
            print(
                f"[STEP] step={step_num} action={format_action(action)} "
                f"reward={result.reward.value:.2f} done={done_text} error={error_value}",
                flush=True,
            )

            if result.done:
                break

        score = grade(
            task_id=task_id,
            applied_fixes=env.applied_fixes,
            found_causes=env.found_causes,
            evidence=env.evidence,
            premature_required_fixes=env.premature_required_fixes,
            steps=env.step_count,
        )
        return score
    except Exception:
        return 0.0
    finally:
        env.close()
        rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
        score = (
            grade(
                task_id=task_id,
                applied_fixes=env.applied_fixes,
                found_causes=env.found_causes,
                evidence=env.evidence,
                premature_required_fixes=env.premature_required_fixes,
                steps=env.step_count,
            )
            if env.task_id
            else 0.0
        )
        success_threshold = 0.1  # competition-style threshold
        print(
            f"[END] success={'true' if score >= success_threshold else 'false'} steps={len(rewards)} "
            f"score={score:.2f} rewards={rewards_str}",
            flush=True,
        )


def main() -> int:
    client = build_client()
    for task_id in ("task_1", "task_2", "task_3"):
        run_task(task_id, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
