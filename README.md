## Kindly visit confluence page for more details about the template working
    https://toyota.atlassian.net/wiki/spaces/TDSP/pages/519832860/TDSP+MLOps+Template

## How to configure the pipeline
    https://toyota.atlassian.net/wiki/spaces/TDSP/pages/795610110/AWS+Sagemaker+Training+Configurations

## W&B logging
This repo only logs training metadata. Keep inference run tracking in API and batch inference repos.

### Environment toggles
- Set `ENABLE_WANDB=true` to enable logging.
- Set `WANDB_DISABLED=true` to force disable.
- If `WANDB_API_KEY` is missing and `WANDB_MODE` is unset, logging defaults to `WANDB_MODE=disabled`.
- Use `WANDB_MODE=offline` to log locally without uploading.

### Schema contract
The shared run-tracker schema lives in `src/wandb_tracking/schema.py`, and the safe wrapper is in
`src/wandb_tracking/wandb_logger.py`. Training logs standardized config/summary fields and
attaches `trained_from_run_id` and `model_registry_path` to the model artifact metadata so inference
repos can link back to the originating training run.

### Data hygiene
Do not log PII. Log only aggregate metrics and metadata.
