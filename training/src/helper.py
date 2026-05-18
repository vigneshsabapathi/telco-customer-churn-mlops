"""MLflow logging wrapper for training.

Centralizes the local-vs-remote tracking decision so train_model.py and
evaluate_model.py don't each need to know about DagsHub or env vars.

Hardening vs. chapter 15: credentials NEVER come from YAML. When
cfg.tracking.remote is True, MLFLOW_TRACKING_USERNAME and
MLFLOW_TRACKING_PASSWORD must be present in the environment or
BaseLogger.start() raises RuntimeError.
"""

from __future__ import annotations

import os

import mlflow
from omegaconf import DictConfig


class BaseLogger:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._started = False

    def start(self) -> "BaseLogger":
        tracking = self.cfg.tracking
        if tracking.remote:
            for var in (
                "MLFLOW_TRACKING_USERNAME",
                "MLFLOW_TRACKING_PASSWORD",
            ):
                if not os.environ.get(var):
                    raise RuntimeError(
                        f"Remote tracking requires env var {var}. "
                        "Credentials are intentionally NOT read from YAML config."
                    )
            mlflow.set_tracking_uri(tracking.remote_uri)
        mlflow.set_experiment(tracking.experiment_name)
        self._started = True
        return self

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict) -> None:
        mlflow.log_metrics(metrics)

    def log_model(self, model, name: str) -> None:
        mlflow.sklearn.log_model(model, name)
