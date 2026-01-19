"""
Shared W&B run-tracker schema for standardized logging across training and inference repos.

This module defines the required and recommended fields for wandb.config and wandb.run.summary
to enable a consistent "run-tracker table" view in W&B UI (Runs Table / Reports).

Copy or vendor this file into inference repos to maintain field consistency.
"""

from typing import Any, Dict, List, Literal, Optional

# Schema version for forward compatibility
RUN_TRACKER_SCHEMA_VERSION = "1.0.0"

# -----------------------------------------------------------------------------
# Required config fields (wandb.config)
# -----------------------------------------------------------------------------
REQUIRED_CONFIG_FIELDS = [
    "run_tracker_schema_version",  # str: schema version for compatibility
    "run_kind",                    # str: "training" | "inference"
    "env",                         # str: environment (dev|qa|prod)
    "service_name",                # str: name of service/job
    "model_name",                  # str: model identifier
    "model_version",               # str: model version tag
]

# -----------------------------------------------------------------------------
# Recommended config fields (wandb.config)
# -----------------------------------------------------------------------------
RECOMMENDED_CONFIG_FIELDS = [
    "model_alias",                 # str: model alias (dev|staging|prod)
    "model_registry_path",         # str: W&B registry path (entity/project/model)
    "model_artifact_ref",          # str: W&B artifact reference (entity/project/artifact:version)
    "trained_from_run_id",         # str: W&B run ID of the training run (for inference linkage)
    "inference_kind",              # str: "api" | "batch" (for inference runs)
    "dataset_name",                # str: dataset identifier
    "dataset_version",             # str: dataset version/snapshot
    "dataset_uri",                 # str: S3/GCS/etc. path to dataset
    "feature_group_arn",           # str: SageMaker Feature Store ARN
    "endpoint",                    # str: API endpoint name (for API inference)
    "batch_id",                    # str: batch job ID (for batch inference)
    "code_git_sha",                # str: git commit SHA
    "image_uri",                   # str: container image URI
    "image_digest",                # str: container image digest (sha256:...)
    "hyperparameters",             # dict: hyperparameters used
]

# -----------------------------------------------------------------------------
# Required summary metrics (wandb.run.summary)
# -----------------------------------------------------------------------------
REQUIRED_SUMMARY_FIELDS = [
    "status",                      # str: "success" | "failed" | "partial"
    "duration_ms",                 # int: total run duration in milliseconds
]

# Training-specific summary fields
TRAINING_SUMMARY_FIELDS = [
    "train_score",                 # float: training accuracy/metric
    "test_score",                  # float: test/validation accuracy/metric
    "best_hyperparameters",        # dict: best hyperparams from tuning
]

# API inference summary fields
API_INFERENCE_SUMMARY_FIELDS = [
    "n_requests",                  # int: total number of requests processed
    "latency_ms_p50",              # float: 50th percentile latency
    "latency_ms_p95",              # float: 95th percentile latency
    "latency_ms_p99",              # float: 99th percentile latency
    "error_rate",                  # float: error rate (0.0-1.0)
]

# Batch inference summary fields
BATCH_INFERENCE_SUMMARY_FIELDS = [
    "n_records",                   # int: total records processed
    "n_errors",                    # int: number of failed records
    "n_predictions",               # int: number of successful predictions
]


def build_training_config(
    env: str,
    project_name: str,
    model_name: str,
    model_version: str,
    model_alias: Optional[str] = None,
    pod_name: Optional[str] = None,
    dataset_uri: Optional[str] = None,
    feature_group_arn: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    code_git_sha: Optional[str] = None,
    image_uri: Optional[str] = None,
    **extra_fields: Any,
) -> Dict[str, Any]:
    """
    Build a standardized config dict for training runs.
    
    Args:
        env: Environment (dev|qa|prod)
        project_name: W&B project name
        model_name: Model identifier
        model_version: Model version tag
        model_alias: Model alias (optional)
        pod_name: Team/pod name for entity (optional)
        dataset_uri: S3/GCS path to training data (optional)
        feature_group_arn: SageMaker Feature Store ARN (optional)
        hyperparameters: Hyperparameters dict (optional)
        code_git_sha: Git commit SHA (optional)
        image_uri: Container image URI (optional)
        **extra_fields: Additional custom fields
    
    Returns:
        Dict suitable for wandb.config.update()
    """
    config = {
        "run_tracker_schema_version": RUN_TRACKER_SCHEMA_VERSION,
        "run_kind": "training",
        "env": env,
        "service_name": f"{project_name}-training",
        "model_name": model_name,
        "model_version": model_version,
    }
    
    # Add optional fields if provided
    if model_alias:
        config["model_alias"] = model_alias
    if pod_name:
        config["model_registry_path"] = f"{pod_name}/{project_name}/{model_name}"
    if dataset_uri:
        config["dataset_uri"] = dataset_uri
    if feature_group_arn:
        config["feature_group_arn"] = feature_group_arn
    if hyperparameters:
        config["hyperparameters"] = hyperparameters
    if code_git_sha:
        config["code_git_sha"] = code_git_sha
    if image_uri:
        config["image_uri"] = image_uri
    
    # Merge any extra fields
    config.update(extra_fields)
    
    return config


def build_inference_config(
    env: str,
    service_name: str,
    inference_kind: Literal["api", "batch"],
    model_name: str,
    model_version: str,
    model_alias: Optional[str] = None,
    model_registry_path: Optional[str] = None,
    model_artifact_ref: Optional[str] = None,
    trained_from_run_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_version: Optional[str] = None,
    dataset_uri: Optional[str] = None,
    endpoint: Optional[str] = None,
    batch_id: Optional[str] = None,
    code_git_sha: Optional[str] = None,
    image_uri: Optional[str] = None,
    image_digest: Optional[str] = None,
    **extra_fields: Any,
) -> Dict[str, Any]:
    """
    Build a standardized config dict for inference runs.
    
    This function is intended for use in API and batch inference repos.
    Copy this module to those repos and call this function.
    
    Args:
        env: Environment (dev|qa|prod)
        service_name: Name of the inference service/job
        inference_kind: Type of inference ("api" | "batch")
        model_name: Model identifier
        model_version: Model version tag
        model_alias: Model alias (optional)
        model_registry_path: W&B registry path (optional)
        model_artifact_ref: W&B artifact reference (optional)
        trained_from_run_id: W&B run ID of training run (optional, for linkage)
        dataset_name: Dataset identifier (optional)
        dataset_version: Dataset version (optional)
        dataset_uri: Path to input data (optional)
        endpoint: API endpoint name (optional, for API inference)
        batch_id: Batch job ID (optional, for batch inference)
        code_git_sha: Git commit SHA (optional)
        image_uri: Container image URI (optional)
        image_digest: Container image digest (optional)
        **extra_fields: Additional custom fields
    
    Returns:
        Dict suitable for wandb.config.update()
    """
    config = {
        "run_tracker_schema_version": RUN_TRACKER_SCHEMA_VERSION,
        "run_kind": "inference",
        "env": env,
        "service_name": service_name,
        "inference_kind": inference_kind,
        "model_name": model_name,
        "model_version": model_version,
    }
    
    # Add optional fields if provided
    if model_alias:
        config["model_alias"] = model_alias
    if model_registry_path:
        config["model_registry_path"] = model_registry_path
    if model_artifact_ref:
        config["model_artifact_ref"] = model_artifact_ref
    if trained_from_run_id:
        config["trained_from_run_id"] = trained_from_run_id
    if dataset_name:
        config["dataset_name"] = dataset_name
    if dataset_version:
        config["dataset_version"] = dataset_version
    if dataset_uri:
        config["dataset_uri"] = dataset_uri
    if endpoint:
        config["endpoint"] = endpoint
    if batch_id:
        config["batch_id"] = batch_id
    if code_git_sha:
        config["code_git_sha"] = code_git_sha
    if image_uri:
        config["image_uri"] = image_uri
    if image_digest:
        config["image_digest"] = image_digest
    
    # Merge any extra fields
    config.update(extra_fields)
    
    return config


def build_model_artifact_metadata(
    trained_from_run_id: str,
    model_version: str,
    model_alias: Optional[str] = None,
    model_registry_path: Optional[str] = None,
    env: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    train_score: Optional[float] = None,
    test_score: Optional[float] = None,
    **extra_fields: Any,
) -> Dict[str, Any]:
    """
    Build metadata dict to attach to model artifacts for inference linkage.
    
    Inference repos should read this metadata when loading the model artifact
    and include trained_from_run_id and model_registry_path in their config.
    
    Args:
        trained_from_run_id: W&B run ID of this training run
        model_version: Model version tag
        model_alias: Model alias (optional)
        model_registry_path: W&B registry path (optional)
        env: Environment where model was trained (optional)
        hyperparameters: Best hyperparameters (optional)
        train_score: Training accuracy/metric (optional)
        test_score: Test accuracy/metric (optional)
        **extra_fields: Additional custom metadata
    
    Returns:
        Dict suitable for artifact.metadata
    """
    metadata = {
        "trained_from_run_id": trained_from_run_id,
        "model_version": model_version,
        "schema_version": RUN_TRACKER_SCHEMA_VERSION,
    }
    
    if model_alias:
        metadata["model_alias"] = model_alias
    if model_registry_path:
        metadata["model_registry_path"] = model_registry_path
    if env:
        metadata["trained_in_env"] = env
    if hyperparameters:
        metadata["hyperparameters"] = hyperparameters
    if train_score is not None:
        metadata["train_score"] = train_score
    if test_score is not None:
        metadata["test_score"] = test_score
    
    metadata.update(extra_fields)
    
    return metadata
