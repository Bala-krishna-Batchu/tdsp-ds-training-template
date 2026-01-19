## Kindly visit confluence page for more details about the template working
    https://toyota.atlassian.net/wiki/spaces/TDSP/pages/519832860/TDSP+MLOps+Template

## How to configure the pipeline
    https://toyota.atlassian.net/wiki/spaces/TDSP/pages/795610110/AWS+Sagemaker+Training+Configurations

## Weights & Biases logging
This repo logs training metrics and the model artifact to W&B. Inference run-tracker
logging should be implemented in the API and batch inference repos using the shared
schema in `src/wandb_tracking/schema.py` and the safe wrapper in
`src/wandb_tracking/wandb_logger.py`.

Environment toggles:
- `ENABLE_WANDB=true|false` to enable or disable logging
- `WANDB_MODE=online|offline|disabled` for runtime mode
- `WANDB_API_KEY` required for online mode
- `WANDB_DISABLED=true|false` (legacy override)

The training job attaches linkage metadata (model version, alias, trained run id, and
registry path) to the logged model artifact so inference runs can reference it.
