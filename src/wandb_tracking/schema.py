"""Shared run-tracker schema contract for training and inference repos."""

RUN_TRACKER_SCHEMA_VERSION = "1"

RUN_KIND_TRAINING = "training"
RUN_KIND_INFERENCE = "inference"

INFERENCE_KIND_API = "api"
INFERENCE_KIND_BATCH = "batch"

CONFIG_REQUIRED_FIELDS = (
    "run_tracker_schema_version",
    "run_kind",
    "env",
    "service_name",
    "model_name",
    "model_version",
    "model_alias",
)

CONFIG_RECOMMENDED_FIELDS = (
    "inference_kind",
    "model_registry_path",
    "model_artifact_ref",
    "trained_from_run_id",
    "dataset_name",
    "dataset_version",
    "dataset_uri",
    "endpoint",
    "batch_id",
    "code_git_sha",
    "image_uri",
    "image_digest",
)

SUMMARY_REQUIRED_FIELDS = ("status", "duration_ms")
SUMMARY_API_FIELDS = ("n_requests", "latency_ms_p50", "latency_ms_p95", "latency_ms_p99", "error_rate")
SUMMARY_BATCH_FIELDS = ("n_records", "n_errors")

LINKAGE_METADATA_FIELDS = ("trained_from_run_id", "model_registry_path")


def build_run_config(
    *,
    run_kind,
    env,
    service_name,
    model_name,
    model_version,
    model_alias,
    inference_kind=None,
    model_registry_path=None,
    model_artifact_ref=None,
    trained_from_run_id=None,
    dataset_name=None,
    dataset_version=None,
    dataset_uri=None,
    endpoint=None,
    batch_id=None,
    code_git_sha=None,
    image_uri=None,
    image_digest=None,
):
    config = {
        "run_tracker_schema_version": RUN_TRACKER_SCHEMA_VERSION,
        "run_kind": run_kind,
        "env": env,
        "service_name": service_name,
        "model_name": model_name,
        "model_version": model_version,
        "model_alias": model_alias,
        "inference_kind": inference_kind,
        "model_registry_path": model_registry_path,
        "model_artifact_ref": model_artifact_ref,
        "trained_from_run_id": trained_from_run_id,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_uri": dataset_uri,
        "endpoint": endpoint,
        "batch_id": batch_id,
        "code_git_sha": code_git_sha,
        "image_uri": image_uri,
        "image_digest": image_digest,
    }
    return {key: value for key, value in config.items() if value is not None}


def build_linkage_metadata(*, trained_from_run_id=None, model_registry_path=None):
    metadata = {
        "trained_from_run_id": trained_from_run_id,
        "model_registry_path": model_registry_path,
    }
    return {key: value for key, value in metadata.items() if value is not None}
