from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, Mapping, Optional

from src.wandb_tracking.schema import RunTrackerConfig

logger = logging.getLogger("root")


def _truthy_env(name: str, default: Optional[bool] = None) -> Optional[bool]:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


class WandbLogger:
    """
    Best-effort W&B wrapper that never fails the job.

    Key behavior:
    - Controlled by ENABLE_WANDB (default: enabled to preserve current behavior).
    - Honors WANDB_MODE (online|offline|disabled).
    - If init/log fails, it degrades to a no-op logger.
    """

    def __init__(self) -> None:
        self._enabled: bool = bool(_truthy_env("ENABLE_WANDB", default=True))
        self._initialized: bool = False
        self._run = None
        self._start_time = time.time()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def run_id(self) -> Optional[str]:
        try:
            return getattr(self._run, "id", None)
        except Exception:
            return None

    def _import_wandb(self):
        if not self._enabled:
            return None
        try:
            import wandb  # type: ignore

            return wandb
        except Exception as e:
            logger.warning("W&B import failed; disabling W&B. error=%s", e)
            self._enabled = False
            return None

    def init(
        self,
        *,
        project: str,
        entity: Optional[str] = None,
        job_type: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        group: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self._enabled or self._initialized:
            return

        wandb = self._import_wandb()
        if wandb is None:
            return

        try:
            # Avoid interactive prompts in CI/containers.
            if os.getenv("WANDB_MODE") is None and os.getenv("WANDB_API_KEY") is None:
                # If no explicit mode and no key, disable to prevent hanging/prompting.
                os.environ["WANDB_MODE"] = "disabled"

            self._run = wandb.init(
                entity=entity,
                project=project,
                job_type=job_type,
                name=name,
                notes=notes,
                tags=list(tags) if tags else None,
                group=group,
            )
            self._initialized = True

            if config:
                self.config_update(dict(config))
        except Exception as e:
            logger.warning("W&B init failed; disabling W&B. error=%s", e)
            self._enabled = False
            self._initialized = False
            self._run = None

    def config_update(self, data: Mapping[str, Any]) -> None:
        if not self._enabled or not self._initialized:
            return
        wandb = self._import_wandb()
        if wandb is None:
            return
        try:
            wandb.config.update(dict(data), allow_val_change=True)
        except Exception as e:
            logger.warning("W&B config.update failed (ignored). error=%s", e)

    def config_update_run_tracker(self, cfg: RunTrackerConfig) -> None:
        self.config_update(cfg.to_wandb_config())

    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        if not self._enabled or not self._initialized:
            return
        wandb = self._import_wandb()
        if wandb is None:
            return
        try:
            wandb.log(dict(metrics), step=step)
        except Exception as e:
            logger.warning("W&B log failed (ignored). error=%s", e)

    def summary_update(self, data: Mapping[str, Any]) -> None:
        if not self._enabled or not self._initialized:
            return
        try:
            for k, v in data.items():
                self._run.summary[k] = v
        except Exception as e:
            logger.warning("W&B summary update failed (ignored). error=%s", e)

    def log_json_artifact(
        self,
        *,
        name: str,
        artifact_type: str,
        payload: Mapping[str, Any],
        filename: str = "payload.json",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        """
        Log a small JSON artifact (e.g., data manifest, run metadata).
        Returns the logged artifact ref if available.
        """
        if not self._enabled or not self._initialized:
            return None
        wandb = self._import_wandb()
        if wandb is None:
            return None

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)

            artifact = wandb.Artifact(name, type=artifact_type, metadata=dict(metadata or {}))
            artifact.add_file(filename)
            logged = wandb.log_artifact(artifact)
            logged.wait()
            try:
                return getattr(logged, "name", None)
            except Exception:
                return None
        except Exception as e:
            logger.warning("W&B log_json_artifact failed (ignored). error=%s", e)
            return None

    def log_model_file_artifact(
        self,
        *,
        model_file_path: str,
        artifact_name: str,
        artifact_type: str = "model",
        metadata: Optional[Mapping[str, Any]] = None,
        registry_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Log a model file as a W&B artifact and optionally link it to the model registry path.
        Returns the logged artifact ref/name if available.
        """
        if not self._enabled or not self._initialized:
            return None
        wandb = self._import_wandb()
        if wandb is None:
            return None

        try:
            artifact = wandb.Artifact(artifact_name, type=artifact_type, metadata=dict(metadata or {}))
            artifact.add_file(model_file_path)
            logged = wandb.log_artifact(artifact)
            logged.wait()

            if registry_path:
                try:
                    wandb.run.link_artifact(artifact, registry_path)
                except Exception as e:
                    logger.warning("W&B link_artifact failed (ignored). error=%s", e)

            # Best-effort artifact ref
            try:
                return getattr(logged, "name", None)
            except Exception:
                return None
        except Exception as e:
            logger.warning("W&B log_model_file_artifact failed (ignored). error=%s", e)
            return None

    def finish(self, status: str = "success") -> None:
        if self._enabled and self._initialized:
            try:
                duration_ms = int((time.time() - self._start_time) * 1000)
                self.summary_update({"status": status, "duration_ms": duration_ms})
            except Exception:
                pass

            try:
                self._run.finish()
            except Exception:
                pass

        self._initialized = False
        self._run = None

