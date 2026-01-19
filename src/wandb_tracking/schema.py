from typing import Any, Dict, Optional

# Schema version to handle evolution
RUN_TRACKER_SCHEMA_VERSION = "1.0.0"

# Standard keys for W&B config (Run Tracker Table)
class RunTrackerConfig:
    SCHEMA_VERSION = "run_tracker_schema_version"
    RUN_KIND = "run_kind"  # 'training' or 'inference'
    ENV = "env"
    SERVICE_NAME = "service_name"
    INFERENCE_KIND = "inference_kind"  # 'api' or 'batch' or None for training
    
    # Model Identity
    MODEL_NAME = "model_name"
    MODEL_VERSION = "model_version"
    MODEL_ALIAS = "model_alias"
    MODEL_REGISTRY_PATH = "model_registry_path"
    MODEL_ARTIFACT_REF = "model_artifact_ref"
    TRAINED_FROM_RUN_ID = "trained_from_run_id"
    
    # Dataset Identity
    DATASET_NAME = "dataset_name"
    DATASET_VERSION = "dataset_version"
    DATASET_URI = "dataset_uri"
    
    # Execution Context
    ENDPOINT = "endpoint"
    BATCH_ID = "batch_id"
    CODE_GIT_SHA = "code_git_sha"
    IMAGE_URI = "image_uri"
    IMAGE_DIGEST = "image_digest"

# Standard keys for W&B Summary Metrics
class RunTrackerMetrics:
    STATUS = "status"  # 'success', 'failed'
    DURATION_MS = "duration_ms"
    
    # API Specific
    REQUEST_COUNT = "n_requests"
    LATENCY_P50 = "latency_ms_p50"
    LATENCY_P95 = "latency_ms_p95"
    LATENCY_P99 = "latency_ms_p99"
    ERROR_RATE = "error_rate"
    
    # Batch Specific
    RECORD_COUNT = "n_records"
    ERROR_COUNT = "n_errors"

def build_run_config(
    run_kind: str,
    env: str,
    service_name: str,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    model_alias: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Helper to build a standardized config dictionary for W&B runs."""
    config = {
        RunTrackerConfig.SCHEMA_VERSION: RUN_TRACKER_SCHEMA_VERSION,
        RunTrackerConfig.RUN_KIND: run_kind,
        RunTrackerConfig.ENV: env,
        RunTrackerConfig.SERVICE_NAME: service_name,
        RunTrackerConfig.MODEL_NAME: model_name,
        RunTrackerConfig.MODEL_VERSION: model_version,
        RunTrackerConfig.MODEL_ALIAS: model_alias,
    }
    # Add any extra valid fields provided in kwargs
    config.update(kwargs)
    return config

def get_linkage_metadata(
    run_id: str,
    model_registry_path: str,
    model_version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Returns metadata dict to attach to a model artifact so downstream 
    inference runs can link back to this training run.
    """
    return {
        RunTrackerConfig.TRAINED_FROM_RUN_ID: run_id,
        RunTrackerConfig.MODEL_REGISTRY_PATH: model_registry_path,
        RunTrackerConfig.MODEL_VERSION: model_version,
    }
