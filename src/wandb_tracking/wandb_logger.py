import logging
import os

import wandb

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "y", "on"}
_FALSEY = {"0", "false", "no", "n", "off"}


def _parse_bool(value):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSEY:
        return False
    return None


def _resolve_wandb_mode():
    mode = os.getenv("WANDB_MODE")
    if mode:
        return mode
    if not os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "disabled"
        return "disabled"
    return None


def _is_wandb_enabled():
    enabled = _parse_bool(os.getenv("ENABLE_WANDB"))
    if enabled is False:
        return False
    if _parse_bool(os.getenv("WANDB_DISABLED")) is True:
        return False
    mode = _resolve_wandb_mode()
    if mode and mode.strip().lower() == "disabled":
        return False
    return True


class WandbLogger:
    def __init__(self, *, project, entity=None, run_name=None, notes=None, tags=None, config=None):
        self.enabled = _is_wandb_enabled()
        self.run = None
        if not self.enabled:
            logger.info("W&B logging disabled.")
            return
        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                notes=notes,
                tags=tags,
                config=config or {},
            )
        except Exception:
            logger.exception("Failed to initialize W&B. Continuing without logging.")
            self.enabled = False
            self.run = None

    @property
    def run_id(self):
        if not self.run:
            return None
        return self.run.id

    def update_config(self, values):
        if not self.run or not values:
            return
        try:
            self.run.config.update(values, allow_val_change=True)
        except Exception:
            logger.exception("Failed to update W&B config.")

    def log_metrics(self, metrics, step=None):
        if not self.run or not metrics:
            return
        try:
            wandb.log(metrics, step=step)
        except Exception:
            logger.exception("Failed to log W&B metrics.")

    def set_summary(self, summary):
        if not self.run or not summary:
            return
        try:
            self.run.summary.update(summary)
        except Exception:
            logger.exception("Failed to set W&B summary metrics.")

    def log_artifact(self, artifact):
        if not self.run or artifact is None:
            return None
        try:
            return self.run.log_artifact(artifact)
        except Exception:
            logger.exception("Failed to log W&B artifact.")
            return None

    def link_artifact(self, artifact, target_path):
        if not self.run or artifact is None:
            return None
        try:
            return self.run.link_artifact(artifact, target_path)
        except Exception:
            logger.exception("Failed to link W&B artifact.")
            return None

    def finish(self):
        if not self.run:
            return
        try:
            self.run.finish()
        except Exception:
            logger.exception("Failed to finalize W&B run.")
