"""
Shared W&B Run-Tracker Schema for Training and Inference Runs.

This module defines the standardized field schema for W&B logging across:
- Training runs (this repo)
- API inference runs (separate repo)
- Batch inference runs (separate repo)

The schema ensures we can create a unified "run-tracker table" in W&B
to audit and query by model/dataset/version/env/endpoint and key metrics.

Usage:
    Copy this file (or vendor the wandb_tracking package) into inference repos
    to maintain consistent field names and types.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Literal
from enum import Enum


# Schema version - bump when breaking changes are made
RUN_TRACKER_SCHEMA_VERSION = "1.0.0"


class RunKind(str, Enum):
    """Type of W&B run for filtering in the run-tracker table."""
    TRAINING = "training"
    INFERENCE = "inference"


class InferenceKind(str, Enum):
    """Sub-type of inference run (only applicable when run_kind=inference)."""
    API = "api"
    BATCH = "batch"


class RunStatus(str, Enum):
    """Final status of a run."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # e.g., batch completed with some errors


@dataclass
class BaseRunConfig:
    """
    Base configuration fields logged to wandb.config for ALL runs.
    
    These fields enable filtering and grouping in the W&B Runs Table.
    """
    # Schema version for forward compatibility
    run_tracker_schema_version: str = RUN_TRACKER_SCHEMA_VERSION
    
    # Run classification
    run_kind: str = ""  # "training" or "inference"
    
    # Environment and service identity
    env: str = ""  # dev, qa, prod
    service_name: str = ""  # e.g., "churn-model-training", "churn-api", "churn-batch"
    
    # Model identity (link back to training)
    model_name: str = ""
    model_version: str = ""
    model_alias: str = ""  # dev, qa, prod alias
    model_registry_path: str = ""  # W&B registry path: "{entity}/{project}/{model_name}"
    model_artifact_ref: str = ""  # W&B artifact reference: "entity/project/artifact:version"
    trained_from_run_id: str = ""  # W&B run ID of the training run that produced the model
    
    # Code/container identity
    code_git_sha: str = ""
    code_git_branch: str = ""
    image_uri: str = ""
    image_digest: str = ""
    
    # Team identity
    pod_name: str = ""
    project_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for wandb.config.update()."""
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class TrainingRunConfig(BaseRunConfig):
    """
    Configuration fields specific to training runs.
    """
    run_kind: str = field(default=RunKind.TRAINING.value)
    
    # Dataset identity for training
    dataset_name: str = ""
    dataset_version: str = ""
    dataset_uri: str = ""  # S3 path or feature store ARN
    feature_group_arn: str = ""
    
    # Training job identity (SageMaker specific)
    training_job_name: str = ""
    training_job_arn: str = ""
    
    # Hyperparameters (logged separately, but we track if they exist)
    hyperparameters_logged: bool = False


@dataclass
class InferenceRunConfig(BaseRunConfig):
    """
    Configuration fields specific to inference runs.
    
    Inference repos should use this schema to ensure consistent
    fields in the run-tracker table.
    """
    run_kind: str = field(default=RunKind.INFERENCE.value)
    
    # Inference type
    inference_kind: str = ""  # "api" or "batch"
    
    # Dataset identity for inference
    dataset_name: str = ""
    dataset_version: str = ""
    dataset_uri: str = ""  # S3 path for batch, or "realtime" for API
    
    # API-specific fields
    endpoint_name: str = ""
    endpoint_url: str = ""
    
    # Batch-specific fields
    batch_id: str = ""
    batch_job_name: str = ""
    input_manifest_uri: str = ""
    output_uri: str = ""


@dataclass
class TrainingRunSummary:
    """
    Summary metrics logged to wandb.run.summary for training runs.
    """
    status: str = ""
    duration_ms: int = 0
    
    # Training metrics
    train_score: float = 0.0
    test_score: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data stats
    n_train_samples: int = 0
    n_test_samples: int = 0
    n_features: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for wandb.run.summary update."""
        return {k: v for k, v in asdict(self).items() if v or v == 0}


@dataclass
class InferenceRunSummary:
    """
    Summary metrics logged to wandb.run.summary for inference runs.
    
    Inference repos should log these fields to enable consistent
    querying in the run-tracker table.
    """
    status: str = ""
    duration_ms: int = 0
    
    # Common inference metrics
    n_records: int = 0  # Total records processed
    n_predictions: int = 0
    n_errors: int = 0
    error_rate: float = 0.0
    
    # API-specific metrics (leave 0 for batch)
    n_requests: int = 0
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    latency_ms_p99: float = 0.0
    
    # Batch-specific metrics (leave 0 for API)
    n_batches: int = 0
    records_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for wandb.run.summary update."""
        return {k: v for k, v in asdict(self).items() if v or v == 0}


# Model artifact metadata keys
# When logging model artifacts, include these metadata fields
# so inference repos can extract linkage information
MODEL_ARTIFACT_METADATA_KEYS = [
    "trained_from_run_id",
    "model_version",
    "model_alias",
    "model_registry_path",
    "training_job_name",
    "code_git_sha",
    "env",
    "pod_name",
    "project_name",
]


def get_model_artifact_metadata(config: TrainingRunConfig, run_id: str) -> Dict[str, str]:
    """
    Build metadata dict to attach to model artifact.
    
    Inference repos can read this metadata to populate their
    trained_from_run_id and other linkage fields.
    
    Args:
        config: Training run configuration
        run_id: W&B run ID (wandb.run.id)
    
    Returns:
        Dict of metadata to attach to artifact
    """
    return {
        "trained_from_run_id": run_id,
        "model_version": config.model_version,
        "model_alias": config.model_alias,
        "model_registry_path": config.model_registry_path,
        "training_job_name": config.training_job_name,
        "code_git_sha": config.code_git_sha,
        "env": config.env,
        "pod_name": config.pod_name,
        "project_name": config.project_name,
        "schema_version": RUN_TRACKER_SCHEMA_VERSION,
    }


# W&B Tags for easy filtering
# Use consistent tag prefixes across all repos
def get_run_tags(
    env: str,
    run_kind: str,
    model_name: str,
    inference_kind: Optional[str] = None,
    extra_tags: Optional[list] = None,
) -> list:
    """
    Generate consistent W&B tags for filtering runs.
    
    Args:
        env: Environment (dev, qa, prod)
        run_kind: "training" or "inference"
        model_name: Model name
        inference_kind: "api" or "batch" (only for inference runs)
        extra_tags: Additional custom tags
    
    Returns:
        List of tags for wandb.init(tags=...)
    """
    tags = [
        f"env:{env}",
        f"kind:{run_kind}",
        f"model:{model_name}",
    ]
    if inference_kind:
        tags.append(f"inference:{inference_kind}")
    if extra_tags:
        tags.extend(extra_tags)
    return tags
