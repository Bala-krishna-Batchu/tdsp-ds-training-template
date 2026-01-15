## Kindly visit confluence page for more details about the template working
    https://toyota.atlassian.net/wiki/spaces/TDSP/pages/519832860/TDSP+MLOps+Template

## How to configure the pipeline
    https://toyota.atlassian.net/wiki/spaces/TDSP/pages/795610110/AWS+Sagemaker+Training+Configurations

## Weights & Biases (W&B) tracking

### Can we implement an inference “run-tracker table” in this repo?

**No**. This repository is **training-only**:
- **Entrypoint**: `src/main_script.sh` calls `python /opt/ml/code/src/train.py`
- There are **no API handlers** (online inference) and **no batch scoring entrypoints** (offline inference)

What we *can* do here is:
- **Standardize a shared run-tracker schema** (for inference repos to reuse)
- **Harden training W&B logging** (toggleable, offline/disabled-safe)
- **Attach training→inference linkage metadata** onto the logged model artifact

### Shared run-tracker schema (for inference repos)

This repo defines a vendorable schema at `src/wandb_tracking/schema.py`.

**Where the “run-tracker table” comes from**:
- In W&B, create a Report with a **Runs Table** filtered by `job_type="inference"` and your tags (env/model/endpoint/batch_id).
- With consistent `wandb.config` + `wandb.run.summary`, the Runs Table becomes the auditable “one row per inference run” tracker.

**Required config fields for inference runs** (put into `wandb.config`):
- `run_tracker_schema_version`
- `run_kind` = `"inference"`
- `env`
- `service_name`
- `inference_kind` = `"api"` or `"batch"`
- `model_name`
- `dataset_name`
- `dataset_version`

**Strongly recommended fields** (for provenance/linkage):
- `model_artifact_ref` (preferred) or `model_registry_path` (+ `model_version` / `model_alias`)
- `trained_from_run_id` (W&B training run id)
- `code_git_sha`
- `image_uri` / `image_digest`
- `endpoint` (API) or `batch_id` (batch)

### Training→Inference linkage

`src/train.py` now logs the model artifact with **artifact metadata** including:
- `trained_from_run_id` (training W&B run id)
- `model_version`, `model_alias`, `env`, `code_git_sha`
- `model_registry_path` (registry link target)
- feature store pointer fields

Inference repos should copy these into inference run config (or read them from the model artifact metadata if loading from W&B).

### Safe W&B behavior / environment variables

W&B behavior is controlled by environment variables:
- `ENABLE_WANDB`: `true|false` (default: `true` to preserve current behavior; logging is best-effort)
- `WANDB_MODE`: `online|offline|disabled` (recommended to set explicitly in CI/CD)
- `WANDB_API_KEY`: provided via CI/CD secret (required for `online`)

The wrapper at `src/wandb_tracking/wandb_logger.py` is **best-effort**:
- No training failure if W&B is unavailable
- Avoids interactive prompts (auto-disables if key is missing and mode isn’t set)
