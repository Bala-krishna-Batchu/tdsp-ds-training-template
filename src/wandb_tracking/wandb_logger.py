import logging
import os

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _parse_bool(value, default=False):
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def _wandb_env_enabled():
    if _parse_bool(os.getenv("WANDB_DISABLED"), default=False):
        return False
    return _parse_bool(os.getenv("ENABLE_WANDB"), default=True)


def _ensure_wandb_mode():
    if os.getenv("WANDB_MODE"):
        return
    if os.getenv("WANDB_API_KEY"):
        return
    os.environ["WANDB_MODE"] = "disabled"
    logger.info("WANDB_API_KEY missing; defaulting WANDB_MODE=disabled")


class WandbLogger:
    def __init__(
        self,
        project,
        entity=None,
        job_type=None,
        name=None,
        notes=None,
        tags=None,
        group=None,
        settings=None,
        init_kwargs=None,
    ):
        self.enabled = False
        self.run = None
        self.wandb = None

        if not _wandb_env_enabled():
            logger.info("W&B logging disabled by environment")
            return

        try:
            import wandb as wandb_module
        except Exception as exc:
            logger.warning("W&B import failed; continuing without logging: %s", exc)
            return

        self.wandb = wandb_module
        _ensure_wandb_mode()

        init_payload = {
            "project": project,
            "entity": entity,
            "job_type": job_type,
            "name": name,
            "notes": notes,
            "tags": tags,
            "group": group,
            "settings": settings,
        }
        init_payload = {key: value for key, value in init_payload.items() if value is not None}
        if init_kwargs:
            init_payload.update(init_kwargs)

        try:
            self.run = wandb_module.init(**init_payload)
            self.enabled = self.run is not None
        except Exception as exc:
            logger.warning("W&B init failed; continuing without logging: %s", exc)
            self.enabled = False
            self.run = None

    @property
    def run_id(self):
        if not self.run:
            return None
        return getattr(self.run, "id", None)

    def update_config(self, values, allow_val_change=True):
        if not self.enabled or not self.run or not values:
            return
        try:
            self.run.config.update(values, allow_val_change=allow_val_change)
        except Exception as exc:
            logger.warning("W&B config update failed: %s", exc)

    def log(self, metrics, step=None):
        if not self.enabled or not metrics:
            return
        try:
            self.wandb.log(metrics, step=step)
        except Exception as exc:
            logger.warning("W&B log failed: %s", exc)

    def set_summary(self, values):
        if not self.enabled or not self.run or not values:
            return
        try:
            self.run.summary.update(values)
        except Exception as exc:
            logger.warning("W&B summary update failed: %s", exc)

    def create_artifact(self, name, artifact_type, metadata=None):
        if not self.enabled:
            return None
        try:
            return self.wandb.Artifact(name, type=artifact_type, metadata=metadata)
        except Exception as exc:
            logger.warning("W&B artifact creation failed: %s", exc)
            return None

    def log_artifact(self, artifact):
        if not self.enabled or artifact is None:
            return None
        try:
            return self.wandb.log_artifact(artifact)
        except Exception as exc:
            logger.warning("W&B artifact log failed: %s", exc)
            return None

    def link_artifact(self, artifact, registry_path):
        if not self.enabled or not self.run or artifact is None or not registry_path:
            return
        try:
            self.run.link_artifact(artifact, registry_path)
        except Exception as exc:
            logger.warning("W&B artifact link failed: %s", exc)

    def finish(self):
        if not self.enabled or not self.run:
            return
        try:
            self.run.finish()
        except Exception as exc:
            logger.warning("W&B finish failed: %s", exc)
