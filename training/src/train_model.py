"""Train an XGBoost classifier on processed Telco data via hyperopt search."""

from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from training.src.helper import BaseLogger


def _build_space(model_cfg: DictConfig) -> dict:
    """Translate the YAML hyperopt config into a hyperopt search space.

    The YAML uses `{low, high, q}` triples for quniform and `{low, high}` for uniform.
    """

    def _one(spec, name):
        low, high = float(spec.low), float(spec.high)
        if "q" in spec:
            return hp.quniform(name, low, high, float(spec.q))
        return hp.uniform(name, low, high)

    return {
        "max_depth": _one(model_cfg.max_depth, "max_depth"),
        "gamma": _one(model_cfg.gamma, "gamma"),
        "reg_alpha": _one(model_cfg.reg_alpha, "reg_alpha"),
        "reg_lambda": _one(model_cfg.reg_lambda, "reg_lambda"),
        "colsample_bytree": _one(
            model_cfg.colsample_bytree, "colsample_bytree"
        ),
        "min_child_weight": _one(
            model_cfg.min_child_weight, "min_child_weight"
        ),
        "scale_pos_weight": _one(
            model_cfg.scale_pos_weight, "scale_pos_weight"
        ),
    }


def _make_objective(X_train, y_train, X_test, y_test, fixed: dict):
    """Closure that hyperopt minimizes — returns negative ROC-AUC on test set."""

    def objective(params):
        # hp.quniform yields floats; cast int-valued params back.
        for k in ("max_depth", "reg_alpha", "min_child_weight"):
            params[k] = int(params[k])

        model = XGBClassifier(
            **params,
            n_estimators=fixed["n_estimators"],
            objective=fixed["objective"],
            eval_metric=fixed["eval_metric"],
            early_stopping_rounds=fixed["early_stopping_rounds"],
            random_state=fixed["seed"],
            use_label_encoder=False,
            verbosity=0,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        return {"loss": -auc, "status": STATUS_OK, "model": model, "auc": auc}

    return objective


def train(cfg: DictConfig) -> None:
    X_train = pd.read_csv(cfg.processed.X_train.path)
    X_test = pd.read_csv(cfg.processed.X_test.path)
    y_train = pd.read_csv(cfg.processed.y_train.path).squeeze()
    y_test = pd.read_csv(cfg.processed.y_test.path).squeeze()

    fixed = {
        "n_estimators": int(cfg.model.n_estimators),
        "objective": cfg.model.objective,
        "eval_metric": cfg.model.eval_metric,
        "early_stopping_rounds": int(cfg.model.early_stopping_rounds),
        "seed": int(cfg.model.seed),
    }
    space = _build_space(cfg.model)
    objective = _make_objective(X_train, y_train, X_test, y_test, fixed)

    logger = BaseLogger(cfg).start()
    with mlflow.start_run():
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=int(cfg.model.max_evals),
            trials=trials,
            rstate=np.random.default_rng(fixed["seed"]),
            show_progressbar=False,
        )

        losses = [r["loss"] for r in trials.results]
        best_trial = trials.results[int(np.argmin(losses))]
        best_model = best_trial["model"]

        logger.log_params({**best_params, **fixed})
        logger.log_metrics({"best_roc_auc": best_trial["auc"]})

        Path(cfg.model.dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, cfg.model.path)
        logger.log_model(best_model, cfg.model.name)

    print(
        f"train: best_auc={best_trial['auc']:.4f}, "
        f"evals={len(trials.results)}, "
        f"saved={cfg.model.path}"
    )


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
