# ML Training Run Debugger

OpenEnv environment where the agent debugs broken training runs: read logs and metrics, inspect configs, then propose and apply fixes. FastAPI server, three graded tasks (easy → hard).

**Live Space:** [geetss-ml-debug-env.hf.space](https://geetss-ml-debug-env.hf.space)

## Motivation

These are recurring real issues: train/val preprocessing mismatch, softmax applied twice before CE, and “good” train/val numbers that hide leakage, imbalance, or a misleading metric. The env encodes each as a small scripted scenario with deterministic grading.

## Tasks

### Task 1 — easy: train/val scaling mismatch

- Symptom: high train accuracy, flat validation.
- Likely cause: validation pipeline missing normalization that training has.
- Goal: find it in the configs, propose the matching fix, apply it.

### Task 2 — medium: double softmax

- Symptom: loss goes down while accuracy goes down.
- Likely cause: logits passed through softmax twice on the way to cross-entropy.
- Goal: trace the loss path and apply the fix.

### Task 3 — hard: silent generalization failure

- Symptom: strong train/val, weak test, no crashes.
- Causes: split leakage, class skew, wrong headline metric.
- Goal: surface each issue and apply all three fixes.

## Actions

- `read_log(target)`
- `check_metric(target)`
- `inspect_config(target)`
- `propose_fix(target)`
- `apply_fix()`

## Observations

Fields include `task_id`, `description`, `difficulty`, `visible_logs`, `visible_metrics`, `visible_config`, `step_count`, `pending_fix`, `applied_fixes`, `message`, `done`.

## Local run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Baseline (`inference.py`)

Uses the OpenAI-compatible client with env vars below. If there is no token, it falls back to a small scripted policy so the script still finishes.

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token"
python inference.py
```

Required for the competition: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.

Stdout format (per task): one `[START]`, one `[STEP]` per `step()`, one `[END]` after `close()`.

## Docker

```bash
docker build -t ml-debug-env .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="your_token" \
  ml-debug-env
```

## HTTP API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Health |
| POST | `/reset` | New episode |
| POST | `/step` | One action |
| GET | `/state` | Current state |
| GET | `/tasks` | Task list |
| GET | `/grader` | Score snapshot |
| GET | `/baseline` | Scripted baseline |

## Notes

- Typed models: `Action`, `Observation`, `Reward` (see `env/environment.py`).
- `openenv.yaml` describes tasks and endpoints.
- Graders are deterministic; scores are in `[0.0, 1.0]`.

## Baseline scores (scripted policy)

With the bundled planner: task_1, task_2, task_3 each reach `1.00` when run to completion.
