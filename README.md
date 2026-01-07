## Kindly visit confluence page for more details about the template working
    https://toyota.atlassian.net/wiki/spaces/TDSP/pages/519832860/TDSP+MLOps+Template

## How to configure the pipeline
    https://toyota.atlassian.net/wiki/spaces/TDSP/pages/795610110/AWS+Sagemaker+Training+Configurations

## W&B run tracking (training + inference contract)

### Can we implement the inference “run-tracker table” here?
No — this repository only runs **training** (`src/main_script.sh` → `src/train.py`). There are **no API/batch inference entrypoints** in this codebase, so inference run rows must be produced in the **API inference repo** and the **batch inference repo**.

What we *do* provide here:
- Training W&B logging is made **safer** (env toggle + graceful failure).
- The logged **model artifact** includes metadata needed to link inference runs back to training (`trained_from_run_id`, `model_version`, etc.).
- A small **shared schema** module you can vendor/copy into inference repos: `src/wandb_tracking/schema.py`.

### Required environment variables
- `ENABLE_WANDB`: `true|false` (default: `true`). When `false`, W&B calls are skipped.
- `WANDB_MODE`: optional. Set to `offline` to force offline logging.
- `WANDB_API_KEY`: required for non-offline logging in headless/CI environments.

### Standardized inference run schema (to use in inference repos)
Put these into `wandb.config` (via `build_common_config(...)` in `src/wandb_tracking/schema.py`):
- `run_tracker_schema_version`
- `run_kind` = `inference`
- `env` (dev/qa/prod)
- `service_name` (e.g., `churn-api` / `churn-batch`)
- `inference_kind` = `api` or `batch`
- `model_name`, `model_version`, `model_alias`
- `model_registry_path` and/or `model_artifact_ref`
- `trained_from_run_id` (training run id)
- `dataset_name`, `dataset_version`, `dataset_uri`
- `code_git_sha`, `image_uri`, `image_digest`

Put final audit metrics into `wandb.run.summary` (via `build_inference_summary(...)`):
- `status` (success/failure)
- `n_requests` (API) / `n_records` (batch)
- `latency_ms_p50/p95/p99` (API), `duration_ms`, `error_rate` (if applicable)

### How to build the W&B “run-tracker table” in the UI
In W&B, create a **Report** with a **Runs Table** panel filtered by:
- `job_type` = `inference`
- plus tags like `env:prod`, `inference_kind:api`, `model_name:...`

Then add table columns for the standardized config fields + summary metrics above.
