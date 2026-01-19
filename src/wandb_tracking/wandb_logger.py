import logging
import os


def _is_truthy(value):
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class WandbLogger:
    def __init__(self, enabled, wandb_module=None, logger=None):
        self._enabled = enabled
        self._wandb = wandb_module
        self._run = None
        self._logger = logger or logging.getLogger(__name__)

    @classmethod
    def from_env(cls, logger=None):
        logger = logger or logging.getLogger(__name__)

        enable_env = os.getenv("ENABLE_WANDB")
        if enable_env is not None and not _is_truthy(enable_env):
            logger.info("ENABLE_WANDB is false; W&B logging disabled.")
            return cls(False, logger=logger)

        if _is_truthy(os.getenv("WANDB_DISABLED")):
            logger.info("WANDB_DISABLED is true; W&B logging disabled.")
            return cls(False, logger=logger)

        mode = (os.getenv("WANDB_MODE") or "").strip().lower()
        if mode == "disabled":
            logger.info("WANDB_MODE=disabled; W&B logging disabled.")
            return cls(False, logger=logger)

        api_key = os.getenv("WANDB_API_KEY") or os.getenv("WANDB_APIKEY")
        if not api_key and not mode:
            logger.info("WANDB_API_KEY missing and WANDB_MODE unset; W&B disabled.")
            return cls(False, logger=logger)

        try:
            import wandb  # pylint: disable=import-error
        except Exception as exc:
            logger.warning("W&B import failed; disabling logging: %s", exc)
            return cls(False, logger=logger)

        return cls(True, wandb_module=wandb, logger=logger)

    @property
    def enabled(self):
        return self._enabled

    @property
    def run(self):
        return self._run

    @property
    def run_id(self):
        if self._run is None:
            return None
        return getattr(self._run, "id", None)

    def init(self, **kwargs):
        if not self._enabled:
            return None
        try:
            self._run = self._wandb.init(**kwargs)
            return self._run
        except Exception as exc:
            self._logger.warning("W&B init failed; disabling logging: %s", exc)
            self._enabled = False
            return None

    def config_update(self, config, allow_val_change=True):
        if not self._enabled or self._run is None:
            return
        try:
            self._run.config.update(config, allow_val_change=allow_val_change)
        except Exception as exc:
            self._logger.warning("W&B config update failed: %s", exc)

    def log(self, data, step=None):
        if not self._enabled or self._run is None:
            return
        try:
            if step is None:
                self._run.log(data)
            else:
                self._run.log(data, step=step)
        except Exception as exc:
            self._logger.warning("W&B log failed: %s", exc)

    def set_summary(self, summary):
        if not self._enabled or self._run is None:
            return
        try:
            for key, value in summary.items():
                self._run.summary[key] = value
        except Exception as exc:
            self._logger.warning("W&B summary update failed: %s", exc)

    def log_artifact(self, name, artifact_type, files, metadata=None, registry_path=None):
        if not self._enabled or self._run is None:
            return None
        try:
            artifact = self._wandb.Artifact(name, type=artifact_type, metadata=metadata)
            for file_path in files:
                artifact.add_file(file_path)
            logged_artifact = self._run.log_artifact(artifact)
            logged_artifact.wait()
            if registry_path:
                self._run.link_artifact(logged_artifact, registry_path)
            return logged_artifact
        except Exception as exc:
            self._logger.warning("W&B artifact logging failed: %s", exc)
            return None

    def finish(self):
        if not self._enabled or self._run is None:
            return
        try:
            self._run.finish()
        except Exception as exc:
            self._logger.warning("W&B finish failed: %s", exc)
