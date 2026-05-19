"""Evaluate the trained model on the held-out test set and write metrics.csv.

Logs metrics to the SAME MLflow run that train_model.py opened, by reading
the run-id from `cfg.model.dir/mlflow_run_id.txt`. Falls back to a new run
if the id file is missing (e.g. test fixtures that skip the train step).
"""

from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import mlflow
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
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


def _metrics_output_path(cfg: DictConfig) -> Path:
    # OmegaConf.select returns None for missing keys (works in struct mode).
    # The default "metrics.csv" matches the path declared in dvc.yaml.
    path = OmegaConf.select(cfg, "metrics_path", default=None) or "metrics.csv"
    return Path(to_absolute_path(path))


def _resolve_run_id(cfg: DictConfig) -> str | None:
    """Read the train-stage run id so evaluate metrics attach to the same run."""
    candidate = Path(to_absolute_path(cfg.model.dir)) / "mlflow_run_id.txt"
    if candidate.exists():
        return candidate.read_text().strip() or None
    return None


def evaluate(cfg: DictConfig) -> dict:
    model = joblib.load(to_absolute_path(cfg.model.path))
    X_test = pd.read_csv(to_absolute_path(cfg.processed.X_test.path))
    y_test = pd.read_csv(to_absolute_path(cfg.processed.y_test.path)).squeeze()

    metrics = _compute_metrics(model, X_test, y_test)

    out_path = _metrics_output_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items()]
    ).to_csv(out_path, index=False)

    logger = BaseLogger(cfg).start()
    run_id = _resolve_run_id(cfg)
    with mlflow.start_run(
        run_id=run_id,
        run_name=f"{cfg.model.name}-eval" if not run_id else None,
    ):
        logger.log_metrics(metrics)

    print("evaluate: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
