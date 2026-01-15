"""
W&B Run-Tracker Module

Provides standardized W&B logging for training and inference runs.

This module can be vendored (copied) into inference repositories to
maintain consistent field schemas across training, API inference,
and batch inference repos.

Components:
- schema: Dataclasses defining config and summary field schemas
- wandb_logger: Safe W&B wrapper with env toggles and graceful failure

Usage (Training):
    from wandb_tracking import WandbLogger, TrainingRunConfig, get_run_tags
    
    config = TrainingRunConfig(
        env="dev",
        model_name="ChurnModel",
        ...
    )
    
    logger = WandbLogger()
    logger.init(
        entity="team",
        project="project",
        tags=get_run_tags(config.env, config.run_kind, config.model_name),
        job_type="training",
    )
    logger.config_update(config.to_dict())
    # ... training code ...
    logger.finish()

Usage (Inference - copy this module to inference repo):
    from wandb_tracking import WandbLogger, InferenceRunConfig, get_run_tags
    
    config = InferenceRunConfig(
        env="prod",
        inference_kind="api",
        model_name="ChurnModel",
        trained_from_run_id="abc123",  # From model artifact metadata
        ...
    )
    
    logger = WandbLogger()
    logger.init(
        entity="team",
        project="project",
        tags=get_run_tags(config.env, config.run_kind, config.model_name, config.inference_kind),
        job_type="inference",
    )
    logger.config_update(config.to_dict())
    # ... inference code ...
    logger.summary_update({"status": "success", "n_records": 1000, ...})
    logger.finish()

Environment Variables:
    ENABLE_WANDB: Set to "false" to disable W&B (default: "true")
    WANDB_MODE: "online" (default), "offline", "disabled"
    WANDB_API_KEY: Required for online mode
"""

from .schema import (
    # Version
    RUN_TRACKER_SCHEMA_VERSION,
    # Enums
    RunKind,
    InferenceKind,
    RunStatus,
    # Config dataclasses
    BaseRunConfig,
    TrainingRunConfig,
    InferenceRunConfig,
    # Summary dataclasses
    TrainingRunSummary,
    InferenceRunSummary,
    # Metadata helpers
    MODEL_ARTIFACT_METADATA_KEYS,
    get_model_artifact_metadata,
    get_run_tags,
)

from .wandb_logger import (
    WandbLogger,
    get_logger,
    is_enabled,
)

__all__ = [
    # Version
    "RUN_TRACKER_SCHEMA_VERSION",
    # Enums
    "RunKind",
    "InferenceKind",
    "RunStatus",
    # Config dataclasses
    "BaseRunConfig",
    "TrainingRunConfig",
    "InferenceRunConfig",
    # Summary dataclasses
    "TrainingRunSummary",
    "InferenceRunSummary",
    # Metadata helpers
    "MODEL_ARTIFACT_METADATA_KEYS",
    "get_model_artifact_metadata",
    "get_run_tags",
    # Logger
    "WandbLogger",
    "get_logger",
    "is_enabled",
]
