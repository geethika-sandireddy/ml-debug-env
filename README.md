# ML Training Run Debugger

An OpenEnv-compliant RL environment where an AI agent acts as a senior ML engineer debugging failing training runs.

## What This Environment Does

The agent investigates broken ML training sessions by reading logs, checking metrics, inspecting configs, and applying fixes.

## Tasks

### Task 1 — Easy: Train/Val Scaling Mismatch
- Symptom: Train accuracy 90%, Val accuracy stuck at 10%. Zero errors.
- Root cause: val_transform is missing the Normalize step
- Agent must: Inspect both transforms, identify the missing step, propose and apply fix.

### Task 2 — Medium: Double Softmax Bug
- Symptom: Loss decreasing every epoch. Accuracy also decreasing every epoch.
- Root cause: Softmax applied twice in loss function
- Agent must: Read loss computation log, identify double softmax, propose and apply fix.

### Task 3 — Hard: Silent Data Pipeline Failure
- Symptom: Train 94%, Val 91%, Test 23%. Zero errors.
- Root causes: data leakage + class imbalance + wrong evaluation metric
- Agent must: Inspect data pipeline, class distribution, and evaluation config separately.

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

- POST /reset - Start new episode
- POST /step - Take one action  
- GET /state - Current state
- GET /tasks - Task list
- GET /grader - Current grader score

## OpenEnv Compliance

- Typed Pydantic models (Action, Observation, Reward)
- reset() returns initial observation
- step() returns (observation, reward, done, info)
- state() returns current environment state
- openenv.yaml with task specifications
- Deterministic graders (0.0–1.0 scores)