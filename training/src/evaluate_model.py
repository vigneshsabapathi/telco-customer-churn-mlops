"""Evaluate the trained model and write metrics.csv."""

from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import mlflow
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from training.src.helper import BaseLogger


def _compute_metrics(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds)),
        "accuracy": float(accuracy_score(y_test, preds)),
    }


def evaluate(cfg: DictConfig) -> dict:
    model = joblib.load(cfg.model.path)
    X_test = pd.read_csv(cfg.processed.X_test.path)
    y_test = pd.read_csv(cfg.processed.y_test.path).squeeze()

    metrics = _compute_metrics(model, X_test, y_test)

    # metrics_path is optional (allows tests to redirect); defaults to repo-root metrics.csv.
    out_path = Path(getattr(cfg, "metrics_path", "metrics.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items()]
    ).to_csv(out_path, index=False)

    logger = BaseLogger(cfg).start()
    with mlflow.start_run():
        logger.log_metrics(metrics)

    print("evaluate: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
