"""
Shared W&B run-tracker schema.

This repo is training-only, but inference repos (API + batch) should reuse this schema
so a W&B "Runs Table" can act as a standardized run-tracker table (one row per run).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional

RUN_TRACKER_SCHEMA_VERSION = "1.0"

RunKind = Literal["training", "inference"]
InferenceKind = Literal["api", "batch"]


@dataclass(frozen=True)
class RunTrackerConfig:
    """
    Canonical config fields to put into wandb.config.

    Inference repos should populate ALL required fields for filtering/auditing.
    Training repos should populate the overlapping subset plus linkage fields.
    """

    # Versioning
    run_tracker_schema_version: str = RUN_TRACKER_SCHEMA_VERSION

    # Identity
    run_kind: RunKind = "inference"
    env: Optional[str] = None
    service_name: Optional[str] = None
    inference_kind: Optional[InferenceKind] = None  # required when run_kind == "inference"

    # Model identity / provenance
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_alias: Optional[str] = None
    model_registry_path: Optional[str] = None  # e.g. "{entity}/{project}/{model_name}"
    model_artifact_ref: Optional[str] = None  # e.g. "entity/project/model_name:v0"
    model_digest: Optional[str] = None  # e.g. sha256 of model file / image digest, if available
    trained_from_run_id: Optional[str] = None  # training run id (W&B) if known

    # Data identity
    dataset_name: Optional[str] = None
    dataset_version: Optional[str] = None
    dataset_uri: Optional[str] = None
    data_manifest_uri: Optional[str] = None

    # Deployment/runtime identity
    endpoint: Optional[str] = None  # API route or logical endpoint name
    batch_id: Optional[str] = None  # batch execution id / sagemaker processing job id / etc.
    code_git_sha: Optional[str] = None
    image_uri: Optional[str] = None
    image_digest: Optional[str] = None

    # Grouping & labels
    tags: Optional[Iterable[str]] = None
    group: Optional[str] = None

    def to_wandb_config(self) -> Dict[str, Any]:
        """
        Convert to a dict suitable for wandb.config.update(...).
        Omits None values.
        """

        raw: Dict[str, Any] = {
            "run_tracker_schema_version": self.run_tracker_schema_version,
            "run_kind": self.run_kind,
            "env": self.env,
            "service_name": self.service_name,
            "inference_kind": self.inference_kind,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_alias": self.model_alias,
            "model_registry_path": self.model_registry_path,
            "model_artifact_ref": self.model_artifact_ref,
            "model_digest": self.model_digest,
            "trained_from_run_id": self.trained_from_run_id,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_uri": self.dataset_uri,
            "data_manifest_uri": self.data_manifest_uri,
            "endpoint": self.endpoint,
            "batch_id": self.batch_id,
            "code_git_sha": self.code_git_sha,
            "image_uri": self.image_uri,
            "image_digest": self.image_digest,
        }

        return {k: v for k, v in raw.items() if v is not None}


INFERENCE_REQUIRED_CONFIG_FIELDS = (
    "run_tracker_schema_version",
    "run_kind",
    "env",
    "service_name",
    "inference_kind",
    "model_name",
    # strongly recommended for provenance:
    # - model_artifact_ref OR model_registry_path (+ model_version/alias)
    # - trained_from_run_id if discoverable
    "dataset_name",
    "dataset_version",
)


def build_registry_path(entity: Optional[str], project: Optional[str], model_name: Optional[str]) -> Optional[str]:
    if not entity or not project or not model_name:
        return None
    return f"{entity}/{project}/{model_name}"

