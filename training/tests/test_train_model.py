"""Tests for training.src.train_model.train(cfg).

Contract under test:
- After train(cfg), the file at cfg.model.path exists and joblib.load() returns
  something with .predict and .predict_proba.
- The trained model achieves ROC-AUC >= 0.78 on the held-out test set (a low
  bar but real — sanity-checks that we're not training a degenerate model).
- The model honors cfg.model.max_evals as the hyperopt iteration budget
  (passed via Hydra override in tests for speed).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from hydra import compose, initialize
from sklearn.metrics import roc_auc_score

# Use a small hyperopt budget so the test runs in under a minute.
TEST_MAX_EVALS = 3


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    """Hydra-composed config with model output redirected to a temp dir and
    a small max_evals override."""
    tmp_models = tmp_path_factory.mktemp("models")
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(
            config_name="main",
            overrides=[
                f"model.dir={tmp_models.as_posix()}",
                f"model.max_evals={TEST_MAX_EVALS}",
            ],
        )
    return cfg


@pytest.fixture(scope="module")
def _processed_csvs_exist(cfg):
    """Sanity guard — train_model.py reads processed CSVs produced by Phase 2."""
    for key in ("X_train", "X_test", "y_train", "y_test"):
        path = Path(cfg.processed[key].path)
        if not path.exists():
            pytest.skip(
                f"Processed CSV {path} missing. Run `python training/src/process.py` first."
            )
    return True


@pytest.fixture(scope="module")
def trained_model_path(cfg, _processed_csvs_exist):
    """Trains once for the whole test module — hyperopt is slow even at max_evals=3."""
    from training.src.train_model import train

    train(cfg)
    return Path(cfg.model.path)


def test_train_produces_model_file(trained_model_path):
    assert (
        trained_model_path.exists()
    ), f"train(cfg) did not write a model to {trained_model_path}"


def test_trained_model_loads_and_predicts(trained_model_path, cfg):
    model = joblib.load(trained_model_path)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    X_test = pd.read_csv(cfg.processed.X_test.path)
    preds = model.predict(X_test.head(5))
    assert len(preds) == 5
    assert set(preds.tolist()).issubset({0, 1})


def test_trained_model_meets_roc_auc_baseline(trained_model_path, cfg):
    model = joblib.load(trained_model_path)
    X_test = pd.read_csv(cfg.processed.X_test.path)
    y_test = pd.read_csv(cfg.processed.y_test.path).squeeze()

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    # Even at max_evals=3 we should easily clear 0.78 on Telco churn.
    # Full max_evals=100 targets 0.82+.
    assert auc >= 0.78, f"ROC-AUC {auc:.3f} below baseline 0.78"
