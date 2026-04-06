# ML Training Run Debugger

An OpenEnv-compliant RL environment where an agent acts as a senior ML engineer debugging failing training runs.

## What This Environment Does

The agent investigates broken ML training sessions by reading logs, checking metrics, inspecting configs, and applying fixes.

## Tasks

### Task 1 - Easy: Train/Val Scaling Mismatch
- Symptom: Train accuracy 90%, Val accuracy stuck at 10%. Zero errors.
- Root cause: `val_transform` is missing the `Normalize` step.
- Agent must: inspect both transforms, identify the missing step, propose the fix, and apply it.

### Task 2 - Medium: Double Softmax Bug
- Symptom: Loss decreases every epoch while accuracy also decreases every epoch.
- Root cause: Softmax is applied twice in the loss path.
- Agent must: inspect the loss computation, identify the double softmax bug, and apply the fix.

### Task 3 - Hard: Silent Data Pipeline Failure
- Symptom: Train 94%, validation 91%, test 23%, with zero runtime errors.
- Root causes: data leakage, class imbalance, and the wrong evaluation metric.
- Agent must: inspect data pipeline, class distribution, and evaluation config separately, then apply all three fixes.

## Action Space

- `read_log(target)`
- `check_metric(target)`
- `inspect_config(target)`
- `propose_fix(target)`
- `apply_fix()`

## Observation Space

- `task_id`
- `description`
- `difficulty`
- `visible_logs`
- `visible_metrics`
- `visible_config`
- `step_count`
- `pending_fix`
- `applied_fixes`
- `message`
- `done`

## Setup

### Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_huggingface_api_key_here"
python inference.py
```

Required environment variables for submission:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

`inference.py` prints strict structured logs in this order for each task:
- `[START]`
- `[STEP]` (one line per call to `step`)
- `[END]` (always emitted once per task)

### Docker

```bash
docker build -t ml-debug-env .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="your_token_here" \
  ml-debug-env
```

## API Endpoints

- GET `/` - Health check
- POST `/reset` - Start new episode
- POST `/step` - Take one action
- GET `/state` - Current state
- GET `/tasks` - Task list
- GET `/grader` - Current grader score
- GET `/baseline` - Scripted baseline results

## OpenEnv Compliance

- Typed Pydantic models: `Action`, `Observation`, and `Reward`
- `reset()` returns the initial observation
- `step()` returns observation, reward, done, and info
- `state()` returns the current environment state
- `openenv.yaml` documents task specifications and endpoints
- Deterministic graders return scores in the range `[0.0, 1.0]`

## Baseline Scores

- `task_1`: `1.00`
- `task_2`: `1.00`
- `task_3`: `1.00`
