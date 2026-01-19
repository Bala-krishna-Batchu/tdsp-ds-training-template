RUN_TRACKER_SCHEMA_VERSION = "1.0"

RUN_KIND_INFERENCE = "inference"
RUN_KIND_TRAINING = "training"

REQUIRED_INFERENCE_CONFIG_FIELDS = [
    "run_tracker_schema_version",
    "run_kind",
    "env",
    "service_name",
    "inference_kind",
    "model_name",
    "model_version",
    "model_alias",
]

RECOMMENDED_INFERENCE_CONFIG_FIELDS = [
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
]

REQUIRED_SUMMARY_FIELDS = [
    "status",
    "duration_ms",
]

API_SUMMARY_FIELDS = [
    "n_requests",
    "latency_ms_p50",
    "latency_ms_p95",
    "latency_ms_p99",
    "error_rate",
]

BATCH_SUMMARY_FIELDS = [
    "n_records",
    "n_errors",
]


def build_model_artifact_metadata(
    model_name,
    model_version,
    model_alias,
    model_registry_path,
    trained_from_run_id,
    project_name=None,
    pod_name=None,
):
    metadata = {
        "run_tracker_schema_version": RUN_TRACKER_SCHEMA_VERSION,
        "model_name": model_name,
        "model_version": model_version,
        "model_alias": model_alias,
        "model_registry_path": model_registry_path,
        "trained_from_run_id": trained_from_run_id,
    }

    if project_name:
        metadata["project_name"] = project_name
    if pod_name:
        metadata["pod_name"] = pod_name

    return {key: value for key, value in metadata.items() if value is not None}
