"""
Safe W&B Logger Wrapper with Environment Toggles and Graceful Failure.

This module provides a WandbLogger class that:
- Respects ENABLE_WANDB env var (true/false) to toggle W&B on/off
- Supports WANDB_MODE for offline mode
- Gracefully handles missing API key or network failures
- Never fails the main job due to W&B errors (best-effort logging)

Usage:
    from wandb_tracking import WandbLogger
    
    logger = WandbLogger()
    if logger.init(entity="team", project="my-project", ...):
        logger.config_update({...})
        logger.log({...})
        artifact = logger.create_artifact(...)
        logger.finish()
"""

import os
import logging
from typing import Optional, Dict, Any, List
from functools import wraps

logger = logging.getLogger(__name__)


def _safe_wandb_call(method_name: str):
    """
    Decorator to wrap W&B calls with exception handling.
    
    Logs warnings on failure but never raises exceptions
    to prevent W&B issues from breaking the main job.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self._enabled:
                logger.debug(f"W&B disabled, skipping {method_name}")
                return None
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.warning(f"W&B {method_name} failed (non-fatal): {e}")
                return None
        return wrapper
    return decorator


class WandbLogger:
    """
    Safe wrapper for Weights & Biases logging operations.
    
    Features:
    - Environment-based enable/disable (ENABLE_WANDB)
    - Graceful handling of missing credentials
    - Offline mode support (WANDB_MODE)
    - Exception-safe: never fails the main job
    
    Environment Variables:
        ENABLE_WANDB: Set to "false" to disable all W&B operations (default: true)
        WANDB_MODE: Set to "offline" for local-only logging, "disabled" to skip
        WANDB_API_KEY: Required for online mode (unless WANDB_MODE=offline|disabled)
    """
    
    def __init__(self):
        self._enabled = self._check_enabled()
        self._initialized = False
        self._run = None
        self._wandb = None
        
        if self._enabled:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb package not installed, disabling W&B logging")
                self._enabled = False
    
    def _check_enabled(self) -> bool:
        """Check if W&B should be enabled based on environment."""
        # Check explicit toggle
        enable_wandb = os.getenv("ENABLE_WANDB", "true").lower()
        if enable_wandb in ("false", "0", "no", "off"):
            logger.info("W&B disabled via ENABLE_WANDB=false")
            return False
        
        # Check WANDB_MODE
        wandb_mode = os.getenv("WANDB_MODE", "").lower()
        if wandb_mode == "disabled":
            logger.info("W&B disabled via WANDB_MODE=disabled")
            return False
        
        # If online mode (default) and no API key, disable gracefully
        if wandb_mode not in ("offline", "dryrun"):
            api_key = os.getenv("WANDB_API_KEY")
            if not api_key:
                logger.warning(
                    "WANDB_API_KEY not set and WANDB_MODE is not offline. "
                    "Disabling W&B logging. Set WANDB_API_KEY or WANDB_MODE=offline."
                )
                return False
        
        return True
    
    @property
    def enabled(self) -> bool:
        """Check if W&B logging is enabled."""
        return self._enabled
    
    @property
    def initialized(self) -> bool:
        """Check if W&B run is initialized."""
        return self._initialized
    
    @property
    def run(self):
        """Get the current W&B run object (or None if not initialized)."""
        return self._run
    
    @property
    def run_id(self) -> Optional[str]:
        """Get the current W&B run ID (or None if not initialized)."""
        if self._run:
            return self._run.id
        return None
    
    @property
    def run_name(self) -> Optional[str]:
        """Get the current W&B run name (or None if not initialized)."""
        if self._run:
            return self._run.name
        return None
    
    @_safe_wandb_call("init")
    def init(
        self,
        entity: str,
        project: str,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        job_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> bool:
        """
        Initialize a W&B run.
        
        Args:
            entity: W&B entity (team/user)
            project: W&B project name
            name: Run name (optional)
            notes: Run notes (optional)
            tags: List of tags for filtering
            job_type: Job type ("training", "inference", etc.)
            config: Initial config dict
            **kwargs: Additional wandb.init arguments
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        if not self._enabled:
            return False
        
        self._run = self._wandb.init(
            entity=entity,
            project=project,
            name=name,
            notes=notes,
            tags=tags,
            job_type=job_type,
            config=config,
            **kwargs,
        )
        self._initialized = True
        logger.info(f"W&B run initialized: {self._run.id} ({self._run.name})")
        return True
    
    @_safe_wandb_call("config.update")
    def config_update(self, config_dict: Dict[str, Any], allow_val_change: bool = True):
        """
        Update the run config.
        
        Args:
            config_dict: Dictionary of config values
            allow_val_change: Allow changing existing values
        """
        if not self._initialized:
            logger.warning("W&B not initialized, skipping config update")
            return
        
        self._wandb.config.update(config_dict, allow_val_change=allow_val_change)
    
    @_safe_wandb_call("log")
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric values
            step: Optional step number
            commit: Whether to commit the log
        """
        if not self._initialized:
            logger.warning("W&B not initialized, skipping log")
            return
        
        self._wandb.log(metrics, step=step, commit=commit)
    
    @_safe_wandb_call("summary.update")
    def summary_update(self, summary_dict: Dict[str, Any]):
        """
        Update the run summary.
        
        Args:
            summary_dict: Dictionary of summary values
        """
        if not self._initialized:
            logger.warning("W&B not initialized, skipping summary update")
            return
        
        for key, value in summary_dict.items():
            self._wandb.run.summary[key] = value
    
    @_safe_wandb_call("create_artifact")
    def create_artifact(
        self,
        name: str,
        artifact_type: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a W&B artifact.
        
        Args:
            name: Artifact name
            artifact_type: Artifact type (e.g., "model", "dataset")
            description: Optional description
            metadata: Optional metadata dict
        
        Returns:
            W&B Artifact object or None if failed
        """
        if not self._initialized:
            logger.warning("W&B not initialized, skipping artifact creation")
            return None
        
        artifact = self._wandb.Artifact(
            name=name,
            type=artifact_type,
            description=description,
            metadata=metadata,
        )
        return artifact
    
    @_safe_wandb_call("log_artifact")
    def log_artifact(self, artifact, aliases: Optional[List[str]] = None):
        """
        Log an artifact to W&B.
        
        Args:
            artifact: W&B Artifact object
            aliases: Optional list of aliases
        
        Returns:
            Logged artifact reference or None if failed
        """
        if not self._initialized or artifact is None:
            return None
        
        logged = self._wandb.log_artifact(artifact, aliases=aliases)
        if logged:
            logged.wait()
        return logged
    
    @_safe_wandb_call("link_artifact")
    def link_artifact(self, artifact, target_path: str):
        """
        Link an artifact to the model registry.
        
        Args:
            artifact: W&B Artifact object
            target_path: Registry path (e.g., "entity/project/model_name")
        """
        if not self._initialized or artifact is None:
            return
        
        self._wandb.run.link_artifact(artifact, target_path)
    
    @_safe_wandb_call("finish")
    def finish(self, exit_code: int = 0, quiet: bool = False):
        """
        Finish the W&B run.
        
        Args:
            exit_code: Exit code (0 for success)
            quiet: Suppress output
        """
        if not self._initialized:
            return
        
        self._wandb.run.finish(exit_code=exit_code, quiet=quiet)
        self._initialized = False
        self._run = None
        logger.info("W&B run finished")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - finish run on exit."""
        if self._initialized:
            exit_code = 1 if exc_type else 0
            self.finish(exit_code=exit_code)
        return False  # Don't suppress exceptions


# Singleton instance for convenience
_default_logger: Optional[WandbLogger] = None


def get_logger() -> WandbLogger:
    """
    Get the default WandbLogger instance.
    
    Creates a singleton instance on first call.
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = WandbLogger()
    return _default_logger


def is_enabled() -> bool:
    """Check if W&B logging is enabled."""
    return get_logger().enabled
