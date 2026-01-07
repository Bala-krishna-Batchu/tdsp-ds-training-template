"""
Standardized W&B run-tracker schema (shared contract).

Goal: make inference runs queryable/auditable via a consistent set of config
fields + summary metrics across API and batch inference repos.

This module is intentionally dependency-free so it can be copied across repos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional


SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class RunKinds:
    TRAINING: str = "training"
    INFERENCE: str = "inference"


@dataclass(frozen=True)
class InferenceKinds:
    API: str = "api"
    BATCH: str = "batch"


def normalize_tags(tags: Optional[Iterable[str]]) -> list[str]:
    if not tags:
        return []
    out: list[str] = []
    for t in tags:
        if t is None:
            continue
        s = str(t).strip()
        if not s:
            continue
        out.append(s)
    return out


def build_common_config(
    *,
    run_kind: str,
    env: Optional[str],
    service_name: Optional[str] = None,
    inference_kind: Optional[str] = None,  # api|batch
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    model_alias: Optional[str] = None,
    model_registry_path: Optional[str] = None,
    model_artifact_ref: Optional[str] = None,
    trained_from_run_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_version: Optional[str] = None,
    dataset_uri: Optional[str] = None,
    code_git_sha: Optional[str] = None,
    image_uri: Optional[str] = None,
    image_digest: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Canonical config fields to put into `wandb.config`.
    """
    cfg: Dict[str, Any] = {
        "run_tracker_schema_version": SCHEMA_VERSION,
        "run_kind": run_kind,
        "env": env,
        "service_name": service_name,
        "inference_kind": inference_kind,
        "model_name": model_name,
        "model_version": model_version,
        "model_alias": model_alias,
        "model_registry_path": model_registry_path,
        "model_artifact_ref": model_artifact_ref,
        "trained_from_run_id": trained_from_run_id,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_uri": dataset_uri,
        "code_git_sha": code_git_sha,
        "image_uri": image_uri,
        "image_digest": image_digest,
    }
    if extra:
        cfg.update(dict(extra))
    return cfg


def build_inference_summary(
    *,
    status: str,
    n_requests: Optional[int] = None,  # API
    n_records: Optional[int] = None,  # batch
    latency_ms_p50: Optional[float] = None,
    latency_ms_p95: Optional[float] = None,
    latency_ms_p99: Optional[float] = None,
    error_rate: Optional[float] = None,
    duration_ms: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Canonical final summary fields to put into `wandb.run.summary`.
    """
    out: Dict[str, Any] = {
        "status": status,
        "n_requests": n_requests,
        "n_records": n_records,
        "latency_ms_p50": latency_ms_p50,
        "latency_ms_p95": latency_ms_p95,
        "latency_ms_p99": latency_ms_p99,
        "error_rate": error_rate,
        "duration_ms": duration_ms,
    }
    if extra:
        out.update(dict(extra))
    return out

