## Kindly visit confluence page for more details about the template working
    https://toyota.atlassian.net/wiki/spaces/TDSP/pages/519832860/TDSP+MLOps+Template

## How to configure the pipeline
    https://toyota.atlassian.net/wiki/spaces/TDSP/pages/795610110/AWS+Sagemaker+Training+Configurations

## Weights & Biases Logging

This repository uses a shared logging contract for Weights & Biases.

### Environment Variables

- `ENABLE_WANDB`: Set to `true` (or `1`, `yes`) to enable W&B logging. Defaults to `true` if `WANDB_API_KEY` is present.
- `WANDB_API_KEY`: Your W&B API key.
- `WANDB_MODE`: Can be set to `online`, `offline`, or `disabled`.

### Inference Run Tracking

While this repository is for **training**, it publishes model artifacts with metadata that allows inference runs (API or Batch) to link back to the training run.

Downstream inference repositories should use the shared schema defined in `src/wandb_tracking/schema.py` to ensure consistent logging in the "Run Tracker" table.

**Linkage Metadata:**
Models logged by this repo include:
- `trained_from_run_id`: The W&B Run ID of the training job.
- `model_registry_path`: The W&B Model Registry path.

### Schema Adoption

Inference repositories should vendor `src/wandb_tracking/` and use:
```python
from src.wandb_tracking import WandbLogger, build_run_config

wl = WandbLogger()
config = build_run_config(
    run_kind="inference",
    inference_kind="api", # or "batch"
    ...
)
wl.init(..., config=config)
```
