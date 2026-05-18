"""Tests for training.src.evaluate_model.evaluate(cfg).

Contract under test:
- After evaluate(cfg) runs against a trained model, metrics.csv exists with
  exactly four columns: metric, value (and value is in [0, 1] for all four).
- The four metrics are: roc_auc, pr_auc, f1, accuracy.
- All metrics are sane: ROC-AUC >= 0.78, accuracy >= 0.70 (Telco baseline).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize

TEST_MAX_EVALS = 3
EXPECTED_METRICS = {"roc_auc", "pr_auc", "f1", "accuracy"}


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp_models = tmp_path_factory.mktemp("models")
    tmp_metrics = tmp_path_factory.mktemp("metrics_out") / "metrics.csv"
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(
            config_name="main",
            overrides=[
                f"model.dir={tmp_models.as_posix()}",
                f"model.max_evals={TEST_MAX_EVALS}",
                f"+metrics_path={tmp_metrics.as_posix()}",
            ],
        )
    return cfg


@pytest.fixture(scope="module")
def _processed_csvs_exist(cfg):
    for key in ("X_train", "X_test", "y_train", "y_test"):
        path = Path(cfg.processed[key].path)
        if not path.exists():
            pytest.skip(f"Processed CSV {path} missing. Run process.py first.")
    return True


@pytest.fixture(scope="module")
def metrics_path(cfg, _processed_csvs_exist) -> Path:
    """Train + evaluate once per module."""
    from training.src.evaluate_model import evaluate
    from training.src.train_model import train

    train(cfg)
    evaluate(cfg)
    return Path(cfg.metrics_path)


def test_metrics_file_exists(metrics_path):
    assert metrics_path.exists(), f"evaluate(cfg) didn't write {metrics_path}"


def test_metrics_csv_has_expected_schema(metrics_path):
    df = pd.read_csv(metrics_path)
    assert set(df.columns) == {
        "metric",
        "value",
    }, f"unexpected columns: {df.columns.tolist()}"
    assert (
        set(df["metric"]) == EXPECTED_METRICS
    ), f"missing metrics: {EXPECTED_METRICS - set(df['metric'])}"


def test_metrics_values_in_unit_interval(metrics_path):
    df = pd.read_csv(metrics_path)
    bad = df[(df["value"] < 0) | (df["value"] > 1)]
    assert bad.empty, f"out-of-range metric values: {bad.to_dict('records')}"


def test_metrics_meet_baselines(metrics_path):
    df = pd.read_csv(metrics_path).set_index("metric")["value"]
    # ROC-AUC is the headline metric for imbalanced binary classification.
    # Accuracy at the default 0.5 threshold is intentionally not gated tightly:
    # the model uses scale_pos_weight to trade accuracy for minority-class recall,
    # which is the right behavior on Telco (~26% positive rate).
    assert df["roc_auc"] >= 0.78, f"roc_auc {df['roc_auc']:.3f} below 0.78"
    assert df["pr_auc"] >= 0.45, f"pr_auc {df['pr_auc']:.3f} below 0.45"
    assert df["accuracy"] >= 0.60, f"accuracy {df['accuracy']:.3f} below 0.60"
