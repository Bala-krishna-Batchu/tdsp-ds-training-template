## Kindly visit confluence page for more details about the template working
    https://toyota.atlassian.net/wiki/spaces/TDSP/pages/519832860/TDSP+MLOps+Template

## How to configure the pipeline
    https://toyota.atlassian.net/wiki/spaces/TDSP/pages/795610110/AWS+Sagemaker+Training+Configurations

## W&B logging
This repository is a training template. It does not execute inference runs.
For inference run-tracker rows, instrument the API and batch inference repos.

Environment variables:
- ENABLE_WANDB=true|false (default: true)
- WANDB_MODE=online|offline|disabled
- WANDB_API_KEY (required for online logging)

Shared run-tracker schema:
- Defined in `src/wandb_tracking/schema.py`.
- Training runs use `run_kind=training` and publish model artifact metadata that
  inference repos can copy into their run config (trained_from_run_id, model
  version/alias, model_registry_path).

Safety guidance:
- Do not log PII or raw input records.
- Prefer dataset pointers and aggregate metrics only.
