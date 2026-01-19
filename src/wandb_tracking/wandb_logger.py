"""
Safe W&B wrapper with environment toggles and graceful failure handling.

This module provides a WandbLogger class that:
- Respects ENABLE_WANDB / WANDB_DISABLED environment variables
- Automatically disables if WANDB_API_KEY is missing (unless offline mode)
- Never fails the training/inference job due to logging errors
- Supports offline mode via WANDB_MODE=offline
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def is_wandb_enabled() -> bool:
    """
    Check if W&B logging should be enabled based on environment variables.
    
    Checks (in order):
    1. ENABLE_WANDB=false -> disabled
    2. WANDB_DISABLED=true -> disabled
    3. WANDB_MODE=disabled -> disabled
    4. No WANDB_API_KEY and WANDB_MODE not in (offline, dryrun) -> disabled
    5. Otherwise -> enabled
    """
    # Explicit disable flags
    if os.getenv("ENABLE_WANDB", "true").lower() == "false":
        logger.info("W&B disabled via ENABLE_WANDB=false")
        return False
    
    if os.getenv("WANDB_DISABLED", "false").lower() == "true":
        logger.info("W&B disabled via WANDB_DISABLED=true")
        return False
    
    wandb_mode = os.getenv("WANDB_MODE", "").lower()
    if wandb_mode == "disabled":
        logger.info("W&B disabled via WANDB_MODE=disabled")
        return False
    
    # Check for API key unless in offline/dryrun mode
    if wandb_mode not in ("offline", "dryrun"):
        if not os.getenv("WANDB_API_KEY"):
            logger.warning(
                "W&B disabled: WANDB_API_KEY not set and WANDB_MODE is not offline/dryrun. "
                "Set WANDB_API_KEY or WANDB_MODE=offline to enable logging."
            )
            return False
    
    return True


class WandbLogger:
    """
    Safe W&B wrapper that handles initialization, logging, and cleanup.
    
    All methods are no-ops if W&B is disabled or initialization failed.
    Exceptions are caught and logged but never propagated.
    
    Usage:
        with WandbLogger(project="my-project", entity="my-team") as wb:
            wb.update_config({"model_name": "churn-model"})
            wb.log({"accuracy": 0.95})
            wb.set_summary("status", "success")
            wb.log_artifact(artifact)
    """
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        job_type: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
        reinit: bool = False,
    ):
        """
        Initialize the W&B logger.
        
        Args:
            project: W&B project name
            entity: W&B entity (team/user)
            name: Run name
            job_type: Job type (training, inference, etc.)
            notes: Run notes
            tags: List of tags for filtering
            config: Initial config dict
            reinit: Allow reinitializing in the same process
        """
        self.enabled = is_wandb_enabled()
        self.run = None
        self._init_params = {
            "project": project,
            "entity": entity,
            "name": name,
            "job_type": job_type,
            "notes": notes,
            "tags": tags,
            "config": config,
            "reinit": reinit,
        }
        self._start_time = None
    
    def __enter__(self) -> "WandbLogger":
        """Initialize W&B run on context entry."""
        self._start_time = time.time()
        if not self.enabled:
            logger.info("W&B logging disabled, skipping initialization")
            return self
        
        try:
            import wandb
            self.run = wandb.init(**self._init_params)
            logger.info(f"W&B run initialized: {self.run.id if self.run else 'None'}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B run: {e}")
            self.enabled = False
            self.run = None
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish W&B run on context exit."""
        if self.run:
            try:
                # Set duration if start time was recorded
                if self._start_time:
                    duration_ms = int((time.time() - self._start_time) * 1000)
                    self.set_summary("duration_ms", duration_ms)
                
                # Set status based on exception
                if exc_type is not None:
                    self.set_summary("status", "failed")
                
                import wandb
                wandb.finish()
                logger.info("W&B run finished successfully")
            except Exception as e:
                logger.error(f"Failed to finish W&B run: {e}")
        
        # Don't suppress exceptions
        return False
    
    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID, or None if not initialized."""
        return self.run.id if self.run else None
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the run config with additional fields.
        
        Args:
            config: Dict of config fields to add/update
        """
        if not self.run:
            return
        
        try:
            import wandb
            wandb.config.update(config)
        except Exception as e:
            logger.error(f"Failed to update W&B config: {e}")
    
    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to W&B.
        
        Args:
            data: Dict of metrics to log
            step: Optional step number
        """
        if not self.run:
            return
        
        try:
            import wandb
            wandb.log(data, step=step)
        except Exception as e:
            logger.error(f"Failed to log to W&B: {e}")
    
    def set_summary(self, key: str, value: Any) -> None:
        """
        Set a summary metric.
        
        Args:
            key: Metric name
            value: Metric value
        """
        if not self.run:
            return
        
        try:
            import wandb
            wandb.run.summary[key] = value
        except Exception as e:
            logger.error(f"Failed to set W&B summary: {e}")
    
    def log_artifact(
        self,
        artifact_or_path: Any,
        name: Optional[str] = None,
        type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Log an artifact to W&B.
        
        Args:
            artifact_or_path: wandb.Artifact object or path to file/directory
            name: Artifact name (required if path is provided)
            type: Artifact type (default: "model")
            metadata: Optional metadata dict
        
        Returns:
            The logged artifact, or None on failure
        """
        if not self.run:
            return None
        
        try:
            import wandb
            
            # Handle path vs artifact object
            if isinstance(artifact_or_path, str):
                if name is None:
                    raise ValueError("name is required when logging a path")
                artifact = wandb.Artifact(name, type=type, metadata=metadata)
                if os.path.isdir(artifact_or_path):
                    artifact.add_dir(artifact_or_path)
                else:
                    artifact.add_file(artifact_or_path)
            else:
                artifact = artifact_or_path
                if metadata:
                    artifact.metadata.update(metadata)
            
            logged = wandb.log_artifact(artifact)
            logged.wait()
            return artifact
        except Exception as e:
            logger.error(f"Failed to log W&B artifact: {e}")
            return None
    
    def link_artifact(self, artifact: Any, target_path: str) -> bool:
        """
        Link an artifact to the model registry.
        
        Args:
            artifact: The artifact to link
            target_path: Registry path (entity/project/model_name)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.run or artifact is None:
            return False
        
        try:
            import wandb
            wandb.run.link_artifact(artifact, target_path)
            logger.info(f"Artifact linked to registry: {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to link artifact to registry: {e}")
            return False
    
    def create_artifact(
        self,
        name: str,
        type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Create a new artifact (without logging it yet).
        
        Args:
            name: Artifact name
            type: Artifact type
            metadata: Optional metadata dict
        
        Returns:
            The created artifact, or None if W&B is disabled
        """
        if not self.run:
            return None
        
        try:
            import wandb
            return wandb.Artifact(name, type=type, metadata=metadata)
        except Exception as e:
            logger.error(f"Failed to create W&B artifact: {e}")
            return None


@contextmanager
def wandb_run(
    project: str,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    job_type: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[list] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for W&B runs with safe initialization and cleanup.
    
    Usage:
        with wandb_run(project="my-project") as wb:
            wb.log({"accuracy": 0.95})
    """
    logger_instance = WandbLogger(
        project=project,
        entity=entity,
        name=name,
        job_type=job_type,
        notes=notes,
        tags=tags,
        config=config,
    )
    with logger_instance as wb:
        yield wb
