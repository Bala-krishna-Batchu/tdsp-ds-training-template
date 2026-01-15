SCHEMA_VERSION = "1.0"

TRAINING_RUN_KIND = "training"
INFERENCE_RUN_KIND = "inference"

INFERENCE_REQUIRED_CONFIG_FIELDS = (
    "run_tracker_schema_version",
    "run_kind",
    "env",
    "service_name",
    "inference_kind",
    "model_name",
    "model_version",
    "model_alias",
)

INFERENCE_RECOMMENDED_CONFIG_FIELDS = (
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

INFERENCE_REQUIRED_SUMMARY_FIELDS = (
    "status",
    "duration_ms",
)

INFERENCE_API_SUMMARY_FIELDS = (
    "n_requests",
    "latency_ms_p50",
    "latency_ms_p95",
    "latency_ms_p99",
    "error_rate",
)

INFERENCE_BATCH_SUMMARY_FIELDS = (
    "n_records",
    "n_errors",
)


def build_training_link_metadata(
    run_id,
    model_name,
    model_version,
    model_alias,
    model_registry_path,
):
    metadata = {
        "run_tracker_schema_version": SCHEMA_VERSION,
        "trained_from_run_id": run_id,
        "model_name": model_name,
        "model_version": model_version,
        "model_alias": model_alias,
        "model_registry_path": model_registry_path,
    }
    return {key: value for key, value in metadata.items() if value is not None}
