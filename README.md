# TDSP DS Training Template

SageMaker training job template with W&B integration for model tracking and registry.

## Documentation

- [Template Overview](https://toyota.atlassian.net/wiki/spaces/TDSP/pages/519832860/TDSP+MLOps+Template)
- [AWS SageMaker Training Configurations](https://toyota.atlassian.net/wiki/spaces/TDSP/pages/795610110/AWS+Sagemaker+Training+Configurations)

---

## W&B Run-Tracker Integration

### Can We Implement Inference Run-Tracker Here?

**No** — this is a **training repository only**. It trains models and registers them to W&B Model Registry. There are no API handlers or batch scoring entrypoints here, so inference-run logging must be implemented in the respective inference repositories.

**What this repo provides:**
1. Shared logging contract (`src/wandb_tracking/schema.py`) that inference repos should copy/vendor
2. Safe W&B wrapper (`src/wandb_tracking/wandb_logger.py`) with env toggles and graceful failure
3. Model artifacts with linkage metadata (`trained_from_run_id`, `model_version`, etc.) so inference runs can reference the exact training run

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         W&B Run-Tracker Table                               │
│  (Unified view of all training + inference runs)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Filter by: job_type, env:*, model:*, inference:api|batch                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────┴───────┐          ┌───────┴───────┐          ┌───────┴───────┐
│ Training Repo │          │  API Inference │          │ Batch Inference│
│   (this repo) │          │     Repo       │          │     Repo       │
├───────────────┤          ├───────────────┤          ├───────────────┤
│ job_type:     │          │ job_type:     │          │ job_type:     │
│   training    │          │   inference   │          │   inference   │
│               │          │ inference_kind│          │ inference_kind│
│               │          │   api         │          │   batch       │
└───────┬───────┘          └───────────────┘          └───────────────┘
        │
        │ Registers model artifact with metadata:
        │   - trained_from_run_id
        │   - model_version, model_alias
        │   - model_registry_path
        ▼
┌───────────────┐
│ W&B Model     │
│ Registry      │◄─── Inference repos load model + extract linkage metadata
└───────────────┘
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_WANDB` | `true` | Set to `false` to disable all W&B operations |
| `WANDB_MODE` | `online` | `online`, `offline`, or `disabled` |
| `WANDB_API_KEY` | (required) | W&B API key (required for online mode) |
| `ENV` | (required) | Environment: `dev`, `qa`, `prod` |
| `GIT_SHA` | (optional) | Git commit SHA for code provenance |
| `GIT_BRANCH` | (optional) | Git branch name |
| `IMAGE_URI` | (optional) | Container image URI |

### Shared Schema (src/wandb_tracking/schema.py)

Copy the `src/wandb_tracking/` package to inference repos to maintain consistent field names.

#### Required Config Fields (wandb.config)

```python
# Common fields (all run types)
run_tracker_schema_version: str  # e.g., "1.0.0"
run_kind: str                    # "training" or "inference"
env: str                         # "dev", "qa", "prod"
service_name: str                # e.g., "churn-model-training"
model_name: str
model_version: str
model_alias: str
model_registry_path: str         # "{entity}/{project}/{model_name}"
pod_name: str
project_name: str
code_git_sha: str
image_uri: str

# Training-specific
training_job_name: str
feature_group_arn: str
dataset_uri: str
hyperparameters_logged: bool

# Inference-specific (for inference repos)
inference_kind: str              # "api" or "batch"
trained_from_run_id: str         # Links back to training run
endpoint_name: str               # API only
batch_id: str                    # Batch only
```

#### Required Summary Fields (wandb.run.summary)

```python
# Common
status: str           # "success", "failed", "partial"
duration_ms: int

# Training
train_score: float
test_score: float
n_train_samples: int
n_test_samples: int

# API Inference
n_requests: int
latency_ms_p50: float
latency_ms_p95: float
latency_ms_p99: float
error_rate: float

# Batch Inference
n_records: int
n_predictions: int
n_errors: int
records_per_second: float
```

#### W&B Tags (for filtering)

Use consistent tag prefixes:
- `env:dev`, `env:qa`, `env:prod`
- `kind:training`, `kind:inference`
- `model:<model_name>`
- `inference:api`, `inference:batch`

### Creating the Run-Tracker Table in W&B

1. Go to W&B Project → **Runs** table
2. Add columns for the schema fields above
3. Filter by `job_type` and tags
4. Save as a **Report** for team access

Alternatively, create a W&B Workspace with:
- Panel: **Runs Table**
- Columns: `run_kind`, `env`, `model_name`, `status`, `duration_ms`, etc.
- Filters: `job_type == "inference"` (for inference-only view)

### Inference Repo Integration Guide

1. **Copy the wandb_tracking package** from this repo to your inference repo
2. **Initialize W&B** at inference start:

```python
from wandb_tracking import WandbLogger, InferenceRunConfig, get_run_tags

config = InferenceRunConfig(
    env=os.getenv("ENV"),
    inference_kind="api",  # or "batch"
    model_name="ChurnModel",
    model_version="v1",
    model_artifact_ref="tdspds/Churn Prediction/ChurnModelTesting:latest",
    trained_from_run_id="<from_model_artifact_metadata>",
    endpoint_name="churn-api-endpoint",
    # ... other fields
)

logger = WandbLogger()
logger.init(
    entity="tdspds",
    project="Churn Prediction",
    name=f"inference-{config.batch_id or config.endpoint_name}",
    job_type="inference",
    tags=get_run_tags(config.env, config.run_kind, config.model_name, config.inference_kind),
)
logger.config_update(config.to_dict())
```

3. **Log metrics** during inference:

```python
# For API: log per-request latency (sampled)
logger.log({"request_latency_ms": latency})

# For batch: log progress
logger.log({"records_processed": count})
```

4. **Update summary** at end:

```python
from wandb_tracking import InferenceRunSummary, RunStatus

summary = InferenceRunSummary(
    status=RunStatus.SUCCESS.value,
    duration_ms=int((end_time - start_time) * 1000),
    n_records=total_records,
    n_predictions=successful_predictions,
    n_errors=error_count,
    error_rate=error_count / total_records,
    # API-specific
    latency_ms_p50=compute_percentile(latencies, 50),
    latency_ms_p95=compute_percentile(latencies, 95),
    latency_ms_p99=compute_percentile(latencies, 99),
)
logger.summary_update(summary.to_dict())
logger.finish()
```

5. **Extract trained_from_run_id** from model artifact:

```python
# When loading model from W&B artifact
artifact = wandb.use_artifact("tdspds/Churn Prediction/ChurnModelTesting:latest")
trained_from_run_id = artifact.metadata.get("trained_from_run_id", "")
```

### Safety Features

The `WandbLogger` wrapper provides:
- **Env toggle**: `ENABLE_WANDB=false` disables all W&B calls
- **Offline mode**: `WANDB_MODE=offline` for local-only logging
- **Graceful failure**: Missing API key auto-disables; exceptions are logged but don't fail the job
- **No PII**: Schema focuses on versions, URIs, IDs — no raw input data

### Tradeoffs: UI Table vs Artifact Table

| Approach | Pros | Cons |
|----------|------|------|
| **W&B Runs Table (UI)** | Built-in, no extra code, one row per run automatically, real-time | Requires W&B access, limited custom columns |
| **Table Artifact** | Queryable via API, custom schema, can include predictions sample | Concurrency issues if multiple runs update same artifact, extra code |

**Recommendation**: Use W&B Runs Table filtered by `job_type` and tags. Only create a consolidated table artifact if you need to export/query outside W&B.

---

## Project Structure

```
├── deploy/
│   ├── dev/
│   │   ├── pipelines-config.yml   # Training job config
│   │   └── tdspds-config.yml      # Model/team config
│   ├── qa/
│   └── prod/
├── src/
│   ├── main_script.sh             # Entrypoint
│   ├── train.py                   # Training script
│   ├── utilities/
│   │   └── fetch_features.py      # Feature store integration
│   └── wandb_tracking/            # W&B logging module (vendorable)
│       ├── __init__.py
│       ├── schema.py              # Shared config/summary schemas
│       └── wandb_logger.py        # Safe W&B wrapper
├── test/
│   └── train_test.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Configuration

### deploy/{env}/tdspds-config.yml

```yaml
TeamDetails:
  TeamName: tdspds
  PodName: tdspds

Model:
  ProjectName: "Churn Prediction"
  ModelName: ChurnModelTesting
  ModelVersion: v1
  ModelAlias: dev

FeatureStore:
  FeatureGroupName: "arn:aws:sagemaker:..."
  FeatureStoreRoleARN: "arn:aws:iam::..."
  FeatureStoreOutputPath: "s3://..."
```

### deploy/{env}/pipelines-config.yml

Contains SageMaker training job configuration: instance type, hyperparameters, input data channels, etc.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment
export ENV=dev
export ENABLE_WANDB=false  # Disable W&B for local testing

# Run (requires SageMaker paths, so typically run in container)
python src/train.py
```

## Testing

```bash
pytest test/
```
