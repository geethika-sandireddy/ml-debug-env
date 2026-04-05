"""
inference.py - CORRECTED VERSION
Baseline inference script using OpenAI client.
Reads API credentials from environment variables.
MANDATORY STDOUT FORMAT: [START], [STEP], [END]
"""

import os
import json
import re
import sys
from openai import OpenAI
from env.environment import MLDebugEnv, Action
from env.graders import grade

# ─────────────────────────────────────────────
# Config from environment variables (REQUIRED)
# ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

# VALIDATION: Must have API key
if not API_KEY:
    print("[ERROR] HF_TOKEN or OPENAI_API_KEY environment variable required")
    sys.exit(1)

MAX_STEPS = 15
TEMPERATURE = 0.0

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """
You are an expert ML engineer debugging a failing training run.
Available actions:
- read_log: Read a log source (e.g. target: "val_metrics", "training_metrics", "loss_computation_log")
- check_metric: Check a metric (e.g. target: "val_accuracy", "class_distribution")
- inspect_config: Inspect config (e.g. target: "val_transform", "loss_function")
- propose_fix: Propose a fix (e.g. target: "add_normalize_to_val_transform")
- apply_fix: Apply the pending fix

Respond ONLY with JSON: {"action_type": "read_log", "target": "val_metrics", "value": null}"


def parse_action(response_text: str) -> dict:
    """Robustly extract JSON from LLM response."""
    if not response_text:
        return {}
    try:
        return json.loads(response_text.strip())
    except Exception:
        pass
    try:
        match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {}


def format_action_str(action_data: dict) -> str:
    """Format action for stdout: read_log('val_metrics')"""
    action_type = action_data.get("action_type", "unknown")
    target = action_data.get("target", "")
    return f"{action_type}('{target}')"


def run_task(task_id: str) -> tuple:
    """Run one task. Return (score, steps_taken, rewards_list)"""
    
    env = MLDebugEnv()
    obs = env.reset(task_id=task_id)
    
    # Emit [START] line
    print(f"[START] task={{task_id}} env=ml-debug-env model={{MODEL_NAME}}", flush=True)
    
    history = []
    rewards_list = []
    error_msg = None
    
    for step_num in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        user_content = f"""Current state:
- Task: {{obs.description}}
- Step: {{obs.step_count}}
- Message: {{obs.message}}
- Applied fixes so far: {{obs.applied_fixes}}
- Pending fix (needs apply_fix): {{obs.pending_fix}}
- Last actions: {{history[-3:]}}

What is your next action? Respond with JSON only."""

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=TEMPERATURE,
                max_tokens=100,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            response_text = ""
            error_msg = str(e)

        action_data = parse_action(response_text)

        # Fallback if parsing fails
        if not action_data.get("action_type"):
            action_data = {"action_type": "read_log", "target": "training_metrics", "value": None}

        action = Action(**action_data)
        result = env.step(action)
        obs = result.observation

        # Format reward to 2 decimal places
        reward_value = round(result.reward.value, 2)
        rewards_list.append(reward_value)
        
        # Format done as lowercase boolean
        done_str = "true" if obs.done else "false"
        error_str = error_msg if error_msg else "null"
        
        # Format action string
        action_str = format_action_str(action_data)
        
        # Emit [STEP] line - EXACT FORMAT REQUIRED
        print(
            f"[STEP] step={{step_num}} action={{action_str}} reward={{reward_value:.2f}} done={{done_str}} error={{error_str}}",
            flush=True
        )
        
        history.append(f"{{action_data['action_type']}}:{{action_data['target']}} -> {{reward_value:.2f}}")

    # Calculate final score
    score = grade(
        task_id=task_id,
        applied_fixes=env.applied_fixes,
        found_causes=env.found_causes,
    )
    
    # Format rewards list
    rewards_str = ",".join([f"{{r:.2f}}" for r in rewards_list])
    
    # Format success as lowercase boolean
    success_str = "true" if score > 0.0 else "false"
    
    # Emit [END] line - EXACT FORMAT REQUIRED
    print(
        f"[END] success={{success_str}} steps={{len(rewards_list)}} score={{score:.2f}} rewards={{rewards_str}}",
        flush=True
    )
    
    return score, len(rewards_list), rewards_list


def main():
    """Run all 3 tasks and emit structured output"""
    scores = {}
    
    for task_id in ["task_1", "task_2", "task_3"]:
        try:
            score, steps, rewards = run_task(task_id)
            scores[task_id] = score
        except Exception as e:
            print(f"[ERROR] Task {{task_id}} failed: {{e}}", flush=True)
            scores[task_id] = 0.0

    return scores


if __name__ == "__main__":
    scores = main()
    sys.exit(0)