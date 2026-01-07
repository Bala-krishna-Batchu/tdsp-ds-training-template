"""
Safe W&B logger wrapper.

Requirements:
- No hard dependency at import-time (so unit tests / minimal envs don't break).
- Env toggle: ENABLE_WANDB=true/false
- Offline mode: WANDB_MODE=offline (or disabled if no API key and mode not set)
- Graceful failure: exceptions disable logging instead of failing the job.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from src.wandb_tracking.schema import normalize_tags


def _env_truthy(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def _try_import_wandb():
    try:
        import wandb  # type: ignore

        return wandb
    except Exception:
        return None


def _default_disable_reason() -> Optional[str]:
    if not _env_truthy("ENABLE_WANDB", default=True):
        return "ENABLE_WANDB is false"
    # If user explicitly set WANDB_MODE, we won't infer-disable.
    if os.getenv("WANDB_MODE"):
        return None
    # If no API key is present, W&B can become interactive; disable by default.
    if not os.getenv("WANDB_API_KEY"):
        return "WANDB_API_KEY missing and WANDB_MODE not set"
    return None


@dataclass
class WandbLogger:
    enabled: bool
    run: Any = None
    wandb: Any = None
    start_time_s: float = 0.0

    @classmethod
    def init(
        cls,
        *,
        entity: Optional[str],
        project: Optional[str],
        name: Optional[str],
        job_type: Optional[str],
        group: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        config: Optional[Mapping[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> "WandbLogger":
        disable_reason = _default_disable_reason()
        if disable_reason:
            return cls(enabled=False, run=None, wandb=None, start_time_s=time.time())

        wandb = _try_import_wandb()
        if wandb is None:
            return cls(enabled=False, run=None, wandb=None, start_time_s=time.time())

        try:
            run = wandb.init(
                entity=_safe_str(entity),
                project=_safe_str(project),
                name=_safe_str(name),
                job_type=_safe_str(job_type),
                group=_safe_str(group),
                tags=normalize_tags(tags),
                notes=_safe_str(notes),
                config=dict(config or {}),
            )
            return cls(enabled=True, run=run, wandb=wandb, start_time_s=time.time())
        except Exception:
            # Never break the job due to W&B issues.
            return cls(enabled=False, run=None, wandb=None, start_time_s=time.time())

    @property
    def run_id(self) -> Optional[str]:
        if not self.enabled or not self.run:
            return None
        return getattr(self.run, "id", None)

    def config_update(self, values: Mapping[str, Any]) -> None:
        if not self.enabled or not self.run:
            return
        try:
            self.run.config.update(dict(values), allow_val_change=True)
        except Exception:
            self.enabled = False

    def log(self, metrics: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        if not self.enabled or not self.wandb:
            return
        try:
            if step is None:
                self.wandb.log(dict(metrics))
            else:
                self.wandb.log(dict(metrics), step=step)
        except Exception:
            self.enabled = False

    def summary_update(self, values: Mapping[str, Any]) -> None:
        if not self.enabled or not self.run:
            return
        try:
            for k, v in dict(values).items():
                self.run.summary[k] = v
        except Exception:
            self.enabled = False

    def log_artifact_file(
        self,
        *,
        artifact_name: str,
        artifact_type: str,
        file_path: str,
        aliases: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Any]:
        if not self.enabled or not self.wandb or not self.run:
            return None
        try:
            artifact = self.wandb.Artifact(
                name=artifact_name, type=artifact_type, metadata=dict(metadata or {})
            )
            artifact.add_file(file_path)
            logged = self.run.log_artifact(artifact, aliases=normalize_tags(aliases))
            try:
                logged.wait()
            except Exception:
                pass
            return logged
        except Exception:
            self.enabled = False
            return None

    def link_artifact(self, artifact: Any, target_path: str) -> None:
        if not self.enabled or not self.run:
            return
        try:
            self.run.link_artifact(artifact, target_path)
        except Exception:
            self.enabled = False

    def finish(self) -> None:
        # Always attempt to record duration if possible.
        duration_ms = (time.time() - (self.start_time_s or time.time())) * 1000.0
        self.summary_update({"duration_ms": duration_ms})
        if not self.enabled or not self.run:
            return
        try:
            self.run.finish()
        except Exception:
            self.enabled = False

