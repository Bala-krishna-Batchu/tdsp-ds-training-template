import os
import logging
from typing import Any, Dict, Optional, Union

# Try importing wandb, but handle if it's missing or fails
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

class WandbLogger:
    """
    Safe wrapper around Weights & Biases logging.
    Respects ENABLE_WANDB env var and handles missing API keys gracefully.
    """
    
    def __init__(self, enable_wandb: Optional[bool] = None):
        # Determine if W&B should be enabled
        # 1. explicit arg
        # 2. env var ENABLE_WANDB
        # 3. default to True if API key present, else False
        
        if enable_wandb is not None:
            self.enabled = enable_wandb
        else:
            env_enable = os.getenv("ENABLE_WANDB", "").lower()
            if env_enable in ("false", "0", "off", "no"):
                self.enabled = False
            elif env_enable in ("true", "1", "on", "yes"):
                self.enabled = True
            else:
                # auto-detect based on API key
                self.enabled = bool(os.getenv("WANDB_API_KEY"))

        if self.enabled and not WANDB_AVAILABLE:
            logger.warning("ENABLE_WANDB=True but wandb package not installed. Disabling W&B.")
            self.enabled = False

        self._run = None

    def init(self, project: str, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize W&B run safely."""
        if not self.enabled:
            logger.info(f"W&B disabled. Skipping init for project={project} name={name}")
            return

        try:
            # Set mode to disabled if no API key (double safety)
            if not os.getenv("WANDB_API_KEY"):
                logger.warning("No WANDB_API_KEY found. Setting WANDB_MODE=disabled")
                os.environ["WANDB_MODE"] = "disabled"
            
            self._run = wandb.init(
                project=project,
                name=name,
                config=config,
                **kwargs
            )
            logger.info(f"W&B run initialized: {self._run.id} ({self._run.name})")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.enabled = False  # Disable for remainder of run
            self._run = None

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Log metrics safely."""
        if not self.enabled or not self._run:
            return
        
        try:
            self._run.log(data, step=step)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def update_config(self, config: Dict[str, Any]):
        """Update run configuration safely."""
        if not self.enabled or not self._run:
            return
            
        try:
            self._run.config.update(config)
        except Exception as e:
            logger.warning(f"Failed to update W&B config: {e}")

    def set_summary(self, key: str, value: Any):
        """Set a summary metric safely."""
        if not self.enabled or not self._run:
            return
            
        try:
            self._run.summary[key] = value
        except Exception as e:
            logger.warning(f"Failed to set W&B summary {key}: {e}")

    def log_artifact(self, 
                     artifact_name: str, 
                     artifact_type: str, 
                     file_path: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     aliases: Optional[list] = None) -> Optional[Any]:
        """Log an artifact safely."""
        if not self.enabled or not self._run:
            return None
        
        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata
            )
            
            if file_path:
                artifact.add_file(file_path)
            
            logged_artifact = self._run.log_artifact(artifact, aliases=aliases)
            return logged_artifact
        except Exception as e:
            logger.error(f"Failed to log artifact {artifact_name}: {e}")
            return None

    def link_artifact(self, artifact: Any, target_path: str):
        """Link an artifact to a registry path."""
        if not self.enabled or not self._run or not artifact:
            return

        try:
            self._run.link_artifact(artifact, target_path)
        except Exception as e:
            logger.error(f"Failed to link artifact to {target_path}: {e}")

    def finish(self, exit_code: int = 0):
        """Finish the run."""
        if not self.enabled or not self._run:
            return

        try:
            self._run.finish(exit_code=exit_code)
            self._run = None
        except Exception as e:
            logger.error(f"Failed to finish W&B run: {e}")

    @property
    def run_id(self) -> Optional[str]:
        if self.enabled and self._run:
            return self._run.id
        return None
