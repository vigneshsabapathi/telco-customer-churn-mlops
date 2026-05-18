"""Tests for training.src.helper.BaseLogger.

Contract under test:
- Local mode (cfg.tracking.remote=False): only MLflow is initialized; no DagsHub.
- Remote mode (cfg.tracking.remote=True): MLflow tracking URI is set to
  cfg.tracking.remote_uri AND env vars MLFLOW_TRACKING_USERNAME / _PASSWORD
  must be present (the helper does NOT pull credentials from YAML).
- log_params(dict) / log_metrics(dict) / log_model(model, name) all delegate
  to mlflow; in remote mode they additionally hit DagsHub.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf


def _local_cfg():
    return OmegaConf.create(
        {
            "tracking": {
                "experiment_name": "telco-churn",
                "remote": False,
                "remote_uri": "",
            }
        }
    )


def _remote_cfg():
    return OmegaConf.create(
        {
            "tracking": {
                "experiment_name": "telco-churn",
                "remote": True,
                "remote_uri": "https://dagshub.example/repo.mlflow",
            }
        }
    )


@patch("training.src.helper.mlflow")
def test_local_mode_sets_only_mlflow(mock_mlflow):
    from training.src.helper import BaseLogger

    logger = BaseLogger(_local_cfg())
    logger.start()

    mock_mlflow.set_experiment.assert_called_once_with("telco-churn")
    # Local mode must NOT set a tracking URI (defaults to local ./mlruns)
    mock_mlflow.set_tracking_uri.assert_not_called()


@patch("training.src.helper.mlflow")
def test_remote_mode_sets_tracking_uri_when_creds_present(
    mock_mlflow, monkeypatch
):
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret-token")

    from training.src.helper import BaseLogger

    logger = BaseLogger(_remote_cfg())
    logger.start()

    mock_mlflow.set_tracking_uri.assert_called_once_with(
        "https://dagshub.example/repo.mlflow"
    )
    mock_mlflow.set_experiment.assert_called_once_with("telco-churn")


def test_remote_mode_raises_when_creds_missing(monkeypatch):
    """Remote tracking without env creds must FAIL FAST, not silently downgrade."""
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

    from training.src.helper import BaseLogger

    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_USERNAME"):
        BaseLogger(_remote_cfg()).start()


@patch("training.src.helper.mlflow")
def test_log_params_delegates_to_mlflow(mock_mlflow):
    from training.src.helper import BaseLogger

    logger = BaseLogger(_local_cfg())
    logger.start()
    logger.log_params({"max_depth": 5, "lr": 0.1})

    mock_mlflow.log_params.assert_called_once_with({"max_depth": 5, "lr": 0.1})


@patch("training.src.helper.mlflow")
def test_log_metrics_delegates_to_mlflow(mock_mlflow):
    from training.src.helper import BaseLogger

    logger = BaseLogger(_local_cfg())
    logger.start()
    logger.log_metrics({"roc_auc": 0.83, "pr_auc": 0.62})

    mock_mlflow.log_metrics.assert_called_once_with(
        {"roc_auc": 0.83, "pr_auc": 0.62}
    )


@patch("training.src.helper.mlflow")
def test_log_model_uses_sklearn_flavor(mock_mlflow):
    from training.src.helper import BaseLogger

    fake_model = MagicMock()
    logger = BaseLogger(_local_cfg())
    logger.start()
    logger.log_model(fake_model, "xgboost")

    mock_mlflow.sklearn.log_model.assert_called_once_with(
        fake_model, "xgboost"
    )


def test_credentials_never_read_from_config():
    """Defensive: even if the cfg somehow carries creds, the logger must ignore them.

    This is the chapter-15 hardening — config files must not be a credential channel.
    """
    cfg = _remote_cfg()
    # Smuggle creds into config to simulate a misconfigured YAML.
    cfg.tracking.MLFLOW_TRACKING_USERNAME = "should_be_ignored"
    cfg.tracking.MLFLOW_TRACKING_PASSWORD = "should_be_ignored"

    # No env vars set:
    os.environ.pop("MLFLOW_TRACKING_USERNAME", None)
    os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)

    from training.src.helper import BaseLogger

    # Must still raise — config creds are not honored.
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_USERNAME"):
        BaseLogger(cfg).start()
