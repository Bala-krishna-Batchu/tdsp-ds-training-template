"""
Safe W&B utilities with environment toggles and graceful failure handling.

This module provides simple wrapper functions (not classes) that:
- Respect ENABLE_WANDB / WANDB_DISABLED environment variables
- Auto-disable if WANDB_API_KEY is missing (unless offline mode)
- Never fail the training job due to logging errors
- Support offline mode via WANDB_MODE=offline

Usage:
    from src.utilities.wandb_utils import safe_wandb_init, safe_wandb_log, safe_wandb_finish

    run = safe_wandb_init(project="my-project", entity="my-team")
    safe_wandb_log({"accuracy": 0.95})
    safe_wandb_finish()
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Module-level state
_wandb_enabled: Optional[bool] = None
_run = None


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
    global _wandb_enabled
    if _wandb_enabled is not None:
        return _wandb_enabled
    
    # Explicit disable flags
    if os.getenv("ENABLE_WANDB", "true").lower() == "false":
        logger.info("W&B disabled via ENABLE_WANDB=false")
        _wandb_enabled = False
        return False
    
    if os.getenv("WANDB_DISABLED", "false").lower() == "true":
        logger.info("W&B disabled via WANDB_DISABLED=true")
        _wandb_enabled = False
        return False
    
    wandb_mode = os.getenv("WANDB_MODE", "").lower()
    if wandb_mode == "disabled":
        logger.info("W&B disabled via WANDB_MODE=disabled")
        _wandb_enabled = False
        return False
    
    # Check for API key unless in offline/dryrun mode
    if wandb_mode not in ("offline", "dryrun"):
        if not os.getenv("WANDB_API_KEY"):
            logger.warning(
                "W&B disabled: WANDB_API_KEY not set and WANDB_MODE is not offline/dryrun. "
                "Set WANDB_API_KEY or WANDB_MODE=offline to enable logging."
            )
            _wandb_enabled = False
            return False
    
    _wandb_enabled = True
    return True


def safe_wandb_init(
    project: str,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    job_type: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[list] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Optional[Any]:
    """
    Safely initialize a W&B run. Returns None if disabled or on failure.
    
    This is a drop-in replacement for wandb.init() with safety features.
    """
    global _run
    
    if not is_wandb_enabled():
        logger.info("W&B logging disabled, skipping initialization")
        return None
    
    try:
        import wandb
        _run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            job_type=job_type,
            notes=notes,
            tags=tags,
            config=config,
            **kwargs,
        )
        logger.info(f"W&B run initialized: {_run.id if _run else 'None'}")
        return _run
    except Exception as e:
        logger.error(f"Failed to initialize W&B run: {e}")
        return None


def safe_wandb_log(data: Dict[str, Any], step: Optional[int] = None) -> bool:
    """
    Safely log metrics to W&B. Returns True if successful.
    
    This is a drop-in replacement for wandb.log() with safety features.
    """
    if not is_wandb_enabled() or _run is None:
        return False
    
    try:
        import wandb
        wandb.log(data, step=step)
        return True
    except Exception as e:
        logger.error(f"Failed to log to W&B: {e}")
        return False


def safe_wandb_config_update(config: Dict[str, Any]) -> bool:
    """
    Safely update W&B config. Returns True if successful.
    """
    if not is_wandb_enabled() or _run is None:
        return False
    
    try:
        import wandb
        wandb.config.update(config)
        return True
    except Exception as e:
        logger.error(f"Failed to update W&B config: {e}")
        return False


def safe_wandb_summary(key: str, value: Any) -> bool:
    """
    Safely set a W&B summary metric. Returns True if successful.
    """
    if not is_wandb_enabled() or _run is None:
        return False
    
    try:
        import wandb
        wandb.run.summary[key] = value
        return True
    except Exception as e:
        logger.error(f"Failed to set W&B summary: {e}")
        return False


def safe_wandb_artifact(
    name: str,
    type: str = "model",
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Safely create a W&B artifact. Returns None if disabled or on failure.
    """
    if not is_wandb_enabled() or _run is None:
        return None
    
    try:
        import wandb
        return wandb.Artifact(name, type=type, metadata=metadata)
    except Exception as e:
        logger.error(f"Failed to create W&B artifact: {e}")
        return None


def safe_wandb_log_artifact(artifact: Any) -> bool:
    """
    Safely log an artifact to W&B. Returns True if successful.
    """
    if not is_wandb_enabled() or _run is None or artifact is None:
        return False
    
    try:
        import wandb
        logged = wandb.log_artifact(artifact)
        logged.wait()
        return True
    except Exception as e:
        logger.error(f"Failed to log W&B artifact: {e}")
        return False


def safe_wandb_link_artifact(artifact: Any, target_path: str) -> bool:
    """
    Safely link an artifact to W&B model registry. Returns True if successful.
    """
    if not is_wandb_enabled() or _run is None or artifact is None:
        return False
    
    try:
        import wandb
        wandb.run.link_artifact(artifact, target_path)
        logger.info(f"Artifact linked to registry: {target_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to link artifact to registry: {e}")
        return False


def safe_wandb_finish() -> bool:
    """
    Safely finish the W&B run. Returns True if successful.
    """
    global _run
    
    if not is_wandb_enabled() or _run is None:
        return False
    
    try:
        import wandb
        wandb.finish()
        _run = None
        logger.info("W&B run finished successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to finish W&B run: {e}")
        return False


def get_run_id() -> Optional[str]:
    """Get current W&B run ID, or None if not initialized."""
    return _run.id if _run else None
