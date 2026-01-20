## TDSP MLOps Training Template

Kindly visit the confluence page for more details about the template:
- [TDSP MLOps Template](https://toyota.atlassian.net/wiki/spaces/TDSP/pages/519832860/TDSP+MLOps+Template)

## How to configure the pipeline
- [AWS Sagemaker Training Configurations](https://toyota.atlassian.net/wiki/spaces/TDSP/pages/795610110/AWS+Sagemaker+Training+Configurations)

---

## W&B Logging

This repo uses Weights & Biases for experiment tracking and model registry.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_WANDB` | Set to `false` to disable W&B logging | `true` |
| `WANDB_DISABLED` | Set to `true` to disable W&B logging | `false` |
| `WANDB_MODE` | W&B mode: `online`, `offline`, `disabled`, `dryrun` | (auto) |
| `WANDB_API_KEY` | W&B API key (required unless offline mode) | — |
| `ENV` | Environment name (`dev`, `qa`, `prod`) | — |

### Safe Logging

All W&B calls use safe wrappers (`src/utilities/wandb_utils.py`) that:
- Never fail the training job due to logging errors
- Auto-disable if `WANDB_API_KEY` is missing
- Respect `ENABLE_WANDB=false` or `WANDB_MODE=disabled`

---

## Run-Tracker Table Schema (Cross-Repo Contract)

This section documents the **shared schema** for W&B run-tracker tables across training and inference repos.

> **Architecture note**: This schema is a *contract*, not code to copy. Each repo implements its own logging following this schema. For a shared library, consider adding to `tdsp-mlops-wandb` base image.

### Schema Version

```
run_tracker_schema_version: "1.0.0"
```

### Required Config Fields (`wandb.config`)

| Field | Type | Description |
|-------|------|-------------|
| `run_tracker_schema_version` | str | Schema version for compatibility |
| `run_kind` | str | `"training"` or `"inference"` |
| `env` | str | Environment: `dev`, `qa`, `prod` |
| `model_name` | str | Model identifier |
| `model_version` | str | Model version tag |

### Recommended Config Fields

| Field | Type | Used By | Description |
|-------|------|---------|-------------|
| `model_alias` | str | both | Model alias (dev/staging/prod) |
| `model_registry_path` | str | both | W&B registry path |
| `trained_from_run_id` | str | inference | Links back to training run |
| `inference_kind` | str | inference | `"api"` or `"batch"` |
| `endpoint` | str | api | API endpoint name |
| `batch_id` | str | batch | Batch job identifier |
| `dataset_uri` | str | both | S3/GCS path to data |
| `hyperparameters` | dict | training | Hyperparameters used |

### Required Summary Fields (`wandb.run.summary`)

| Field | Type | Description |
|-------|------|-------------|
| `status` | str | `"success"`, `"failed"`, `"partial"` |

### Training Summary Fields

| Field | Type | Description |
|-------|------|-------------|
| `train_score` | float | Training accuracy/metric |
| `test_score` | float | Test/validation accuracy/metric |

### Inference Summary Fields (API)

| Field | Type | Description |
|-------|------|-------------|
| `n_requests` | int | Total requests processed |
| `latency_ms_p50` | float | 50th percentile latency |
| `latency_ms_p95` | float | 95th percentile latency |
| `latency_ms_p99` | float | 99th percentile latency |
| `error_rate` | float | Error rate (0.0-1.0) |

### Inference Summary Fields (Batch)

| Field | Type | Description |
|-------|------|-------------|
| `n_records` | int | Total records processed |
| `n_predictions` | int | Successful predictions |
| `n_errors` | int | Failed records |

---

## What This Repo Logs

### Config Fields
- `run_tracker_schema_version`, `run_kind` ("training")
- `env`, `model_name`, `model_version`, `model_alias`
- `model_registry_path`, `hyperparameters`

### Metrics
- `Training Score`, `Test Score`

### Summary
- `train_score`, `test_score`, `status`

### Model Artifact Metadata

The model artifact includes metadata for inference run linkage:

```json
{
  "trained_from_run_id": "<wandb_run_id>",
  "model_version": "v1",
  "model_alias": "dev",
  "model_registry_path": "tdspds/Churn Prediction/ChurnModelTesting",
  "trained_in_env": "dev",
  "schema_version": "1.0.0"
}
```

Inference repos should read this metadata and include `trained_from_run_id` in their run config.

---

## Inference Repos: Implementation Guide

**Do NOT copy code from this repo.** Instead, implement the schema contract directly:

### API Inference Example

```python
import wandb

# Initialize with standardized config
wandb.init(
    project="Churn Prediction",
    entity="tdspds",
    job_type="inference",
    tags=["prod", "api", "v1"],
    config={
        "run_tracker_schema_version": "1.0.0",
        "run_kind": "inference",
        "inference_kind": "api",
        "env": "prod",
        "model_name": "ChurnModelTesting",
        "model_version": "v1",
        "trained_from_run_id": "abc123",  # from model artifact metadata
        "endpoint": "churn-prediction-endpoint",
    },
)

# ... serve requests ...

# Log summary metrics
wandb.run.summary["n_requests"] = 1000
wandb.run.summary["latency_ms_p50"] = 45.2
wandb.run.summary["latency_ms_p95"] = 120.5
wandb.run.summary["error_rate"] = 0.001
wandb.run.summary["status"] = "success"

wandb.finish()
```

### Batch Inference Example

```python
import wandb

wandb.init(
    project="Churn Prediction",
    entity="tdspds",
    job_type="inference",
    tags=["prod", "batch", "v1"],
    config={
        "run_tracker_schema_version": "1.0.0",
        "run_kind": "inference",
        "inference_kind": "batch",
        "env": "prod",
        "model_name": "ChurnModelTesting",
        "model_version": "v1",
        "trained_from_run_id": "abc123",
        "batch_id": "batch-20240115-001",
        "dataset_uri": "s3://bucket/batch/input/",
    },
)

# ... process batch ...

wandb.run.summary["n_records"] = 50000
wandb.run.summary["n_predictions"] = 49950
wandb.run.summary["n_errors"] = 50
wandb.run.summary["status"] = "success"

wandb.finish()
```

### Safe Logging Pattern

For graceful failure handling, wrap W&B calls:

```python
import os
import logging

def is_wandb_enabled():
    if os.getenv("ENABLE_WANDB", "true").lower() == "false":
        return False
    if os.getenv("WANDB_DISABLED", "false").lower() == "true":
        return False
    if os.getenv("WANDB_MODE", "").lower() == "disabled":
        return False
    if not os.getenv("WANDB_API_KEY") and os.getenv("WANDB_MODE", "").lower() not in ("offline", "dryrun"):
        return False
    return True

def safe_wandb_init(**kwargs):
    if not is_wandb_enabled():
        return None
    try:
        import wandb
        return wandb.init(**kwargs)
    except Exception as e:
        logging.error(f"W&B init failed: {e}")
        return None
```

---

## Creating the Run-Tracker Table in W&B

1. Go to W&B project → **Reports** → **Create Report**
2. Add a **Runs Table** panel
3. Filter by `job_type` = `"training"` or `"inference"`
4. Add columns: `env`, `model_version`, `status`, `train_score`/`test_score` (training) or `n_requests`/`error_rate` (inference)
5. Group by `model_name` or `env` as needed

This gives a "one row per run" view automatically.

---

## Architecture Recommendations

### For Shared Schema Library

If you need a true shared library across repos, add it to the **`tdsp-mlops-wandb` base image**:

```dockerfile
# This repo already uses:
FROM docker-prod.artifactory.tmna-devops.com/tdsp/tdsp-mlops-wandb:1.0.4 AS wandb
COPY --from=wandb /opt/ml/code/src/initialize_tdspds.sh src/initialize_tdspds.sh
COPY --from=wandb /opt/ml/code/src/initiate_wandb.py src/initiate_wandb.py
COPY --from=wandb /opt/ml/code/src/utils src/utils
```

The `tdsp-mlops-wandb` image is the right place for:
- Shared schema constants
- Safe wrapper utilities
- Common W&B initialization logic

This ensures all repos using the base image get the same schema/utilities automatically.
