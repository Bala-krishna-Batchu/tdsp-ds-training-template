## TDSP MLOps Training Template

Kindly visit the confluence page for more details about the template:
- [TDSP MLOps Template](https://toyota.atlassian.net/wiki/spaces/TDSP/pages/519832860/TDSP+MLOps+Template)

## How to configure the pipeline
- [AWS Sagemaker Training Configurations](https://toyota.atlassian.net/wiki/spaces/TDSP/pages/795610110/AWS+Sagemaker+Training+Configurations)

---

## W&B Logging and Run-Tracker Schema

This repository uses [Weights & Biases](https://wandb.ai) for experiment tracking and model registry.

### Can We Implement Inference Run-Tracker Here?

**No.** This is a **training-only** repository. The entrypoint (`src/main_script.sh` → `src/train.py`) only executes model training, not inference.

For inference run-tracking, implement the shared schema in your:
- **API inference repo** (online serving)
- **Batch inference repo** (offline scoring jobs)

This repo provides:
1. **Shared logging contract** (`src/wandb_tracking/schema.py`) — copy/vendor to inference repos
2. **Safe W&B wrapper** (`src/wandb_tracking/wandb_logger.py`) — copy/vendor to inference repos
3. **Training run metadata** with linkage fields for inference runs to reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_WANDB` | Set to `false` to disable W&B logging | `true` |
| `WANDB_DISABLED` | Set to `true` to disable W&B logging | `false` |
| `WANDB_MODE` | W&B mode: `online`, `offline`, `disabled`, `dryrun` | (auto) |
| `WANDB_API_KEY` | W&B API key (required unless offline mode) | — |
| `ENV` | Environment name (`dev`, `qa`, `prod`) | `dev` |

### What Gets Logged (Training)

**Config fields** (for filtering in W&B UI):
- `run_tracker_schema_version`, `run_kind` ("training")
- `env`, `service_name`, `model_name`, `model_version`, `model_alias`
- `model_registry_path`, `dataset_uri`, `feature_group_arn`
- `hyperparameters`, `code_git_sha`, `image_uri`

**Metrics logged**:
- `Training Score` (training accuracy)
- `Test Score` (test accuracy)
- `train_samples`, `test_samples`, `n_features`
- `best_hyperparameters`

**Summary fields**:
- `train_score`, `test_score`, `status`, `duration_ms`

**Artifacts**:
- Model file (serialized RandomForestClassifier via joblib)
- Artifact metadata includes `trained_from_run_id` for inference linkage

### Model Artifact Metadata (for Inference Linkage)

The model artifact includes metadata that inference repos should use:

```python
{
    "trained_from_run_id": "<wandb_run_id>",
    "model_version": "v1",
    "model_alias": "dev",
    "model_registry_path": "tdspds/Churn Prediction/ChurnModelTesting",
    "trained_in_env": "dev",
    "hyperparameters": {...},
    "train_score": 0.95,
    "test_score": 0.92,
    "schema_version": "1.0.0"
}
```

Inference repos should:
1. Load model from W&B artifact (or read metadata from artifact)
2. Include `trained_from_run_id` and `model_registry_path` in their inference run config
3. Use `build_inference_config()` from the schema module

---

## Inference Repos: How to Adopt the Schema

Copy `src/wandb_tracking/` to your inference repo, then:

### API Inference Example

```python
from wandb_tracking import WandbLogger, build_inference_config

config = build_inference_config(
    env="prod",
    service_name="churn-api",
    inference_kind="api",
    model_name="ChurnModelTesting",
    model_version="v1",
    model_alias="prod",
    trained_from_run_id="abc123",  # from model artifact metadata
    model_registry_path="tdspds/Churn Prediction/ChurnModelTesting",
    endpoint="churn-prediction-endpoint",
)

with WandbLogger(
    project="Churn Prediction",
    entity="tdspds",
    job_type="inference",
    tags=["prod", "api", "v1"],
    config=config,
) as wb:
    # ... serve requests ...
    wb.set_summary("n_requests", 1000)
    wb.set_summary("latency_ms_p50", 45.2)
    wb.set_summary("latency_ms_p95", 120.5)
    wb.set_summary("error_rate", 0.001)
    wb.set_summary("status", "success")
```

### Batch Inference Example

```python
from wandb_tracking import WandbLogger, build_inference_config

config = build_inference_config(
    env="prod",
    service_name="churn-batch",
    inference_kind="batch",
    model_name="ChurnModelTesting",
    model_version="v1",
    trained_from_run_id="abc123",
    dataset_uri="s3://bucket/batch/input/20240115/",
    batch_id="batch-20240115-001",
)

with WandbLogger(
    project="Churn Prediction",
    entity="tdspds",
    job_type="inference",
    tags=["prod", "batch", "v1"],
    config=config,
) as wb:
    # ... process batch ...
    wb.set_summary("n_records", 50000)
    wb.set_summary("n_predictions", 49950)
    wb.set_summary("n_errors", 50)
    wb.set_summary("status", "success")
```

---

## Creating the Run-Tracker Table in W&B

### Recommended: W&B Report with Runs Table

1. Go to W&B project → **Reports** → **Create Report**
2. Add a **Runs Table** panel
3. Filter: `job_type = "inference"` (or `"training"` for training runs)
4. Add columns: `env`, `model_version`, `endpoint`, `batch_id`, `status`, `duration_ms`, etc.
5. Group by `model_name` or `env` as needed

This gives you a "one row per inference run" view automatically.

### Alternative: Table Artifact (Not Recommended)

You could create a consolidated W&B Table artifact updated by each run, but this has race condition issues when multiple inference runs execute concurrently. Prefer the Runs Table view.

---

## Safety and PII

- **No PII**: The schema logs only version identifiers, URIs, metrics, and run IDs
- **Graceful failure**: If W&B is unavailable, logging is skipped and the job continues
- **Offline mode**: Set `WANDB_MODE=offline` to log locally without network access

---

## File Structure

```
src/
├── train.py                    # Training entrypoint (uses WandbLogger)
├── main_script.sh              # SageMaker entrypoint script
├── utilities/
│   └── fetch_features.py       # Feature store utilities
└── wandb_tracking/             # W&B logging utilities (vendor to inference repos)
    ├── __init__.py
    ├── schema.py               # Shared run-tracker schema
    └── wandb_logger.py         # Safe W&B wrapper
```
