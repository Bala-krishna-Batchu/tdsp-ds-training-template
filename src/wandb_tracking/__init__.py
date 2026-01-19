"""
W&B tracking utilities for standardized run logging.

This module provides:
- WandbLogger: Safe W&B wrapper with env toggles and graceful failure
- Schema builders: Functions to construct standardized config dicts
- Schema constants: Field definitions for run-tracker table consistency
"""

from .wandb_logger import WandbLogger, wandb_run, is_wandb_enabled
from .schema import (
    RUN_TRACKER_SCHEMA_VERSION,
    REQUIRED_CONFIG_FIELDS,
    RECOMMENDED_CONFIG_FIELDS,
    REQUIRED_SUMMARY_FIELDS,
    TRAINING_SUMMARY_FIELDS,
    API_INFERENCE_SUMMARY_FIELDS,
    BATCH_INFERENCE_SUMMARY_FIELDS,
    build_training_config,
    build_inference_config,
    build_model_artifact_metadata,
)

__all__ = [
    # Logger
    "WandbLogger",
    "wandb_run",
    "is_wandb_enabled",
    # Schema version
    "RUN_TRACKER_SCHEMA_VERSION",
    # Field definitions
    "REQUIRED_CONFIG_FIELDS",
    "RECOMMENDED_CONFIG_FIELDS",
    "REQUIRED_SUMMARY_FIELDS",
    "TRAINING_SUMMARY_FIELDS",
    "API_INFERENCE_SUMMARY_FIELDS",
    "BATCH_INFERENCE_SUMMARY_FIELDS",
    # Config builders
    "build_training_config",
    "build_inference_config",
    "build_model_artifact_metadata",
]
