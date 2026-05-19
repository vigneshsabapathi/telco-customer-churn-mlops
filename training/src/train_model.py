"""Train an XGBoost classifier on processed Telco data via hyperopt search.

Cleanliness contract: the held-out test set in `cfg.processed.X_test` is
NEVER touched here — neither for model selection nor for early stopping.
Hyperopt searches on (X_tr, y_tr) with early-stopping against a validation
split carved out of X_train. evaluate_model.py is the only consumer of the
test set.

This module also writes the MLflow run-id to `cfg.model.dir/mlflow_run_id.txt`
so evaluate_model.py can attach its metrics to the same run (rather than
opening a separate orphan run).
"""

from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import mlflow
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from training.src.helper import BaseLogger

# Carve a hyperopt validation set out of X_train. Keeps test untouched.
_HYPEROPT_VAL_FRACTION = 0.2


def _build_space(model_cfg: DictConfig) -> dict:
    """Translate the YAML hyperopt config into a hyperopt search space."""

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


def _make_objective(X_tr, y_tr, X_val, y_val, fixed: dict):
    """Closure hyperopt minimizes — fits on tr, early-stops on val.

    Note: test set is NOT passed. Model selection and early stopping both
    use the validation split. Test metrics happen in evaluate_model.
    """

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
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        return {"loss": -auc, "status": STATUS_OK, "model": model, "auc": auc}

    return objective


def train(cfg: DictConfig) -> None:
    X_train = pd.read_csv(to_absolute_path(cfg.processed.X_train.path))
    y_train = pd.read_csv(
        to_absolute_path(cfg.processed.y_train.path)
    ).squeeze()

    # Carve out validation set for hyperopt. Stratified so class balance
    # is preserved (~26% positive on Telco). Test set is untouched here.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=_HYPEROPT_VAL_FRACTION,
        random_state=int(cfg.split.random_state),
        stratify=y_train,
    )

    fixed = {
        "n_estimators": int(cfg.model.n_estimators),
        "objective": cfg.model.objective,
        "eval_metric": cfg.model.eval_metric,
        "early_stopping_rounds": int(cfg.model.early_stopping_rounds),
        "seed": int(cfg.model.seed),
    }
    space = _build_space(cfg.model)
    objective = _make_objective(X_tr, y_tr, X_val, y_val, fixed)

    logger = BaseLogger(cfg).start()
    with mlflow.start_run(run_name=f"{cfg.model.name}-train") as run:
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
        # NB: best_val_roc_auc is on the *internal* validation split. The
        # honest test-set numbers land later in evaluate_model.py.
        logger.log_metrics({"best_val_roc_auc": best_trial["auc"]})

        model_dir = Path(to_absolute_path(cfg.model.dir))
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, to_absolute_path(cfg.model.path))
        logger.log_model(best_model, cfg.model.name)

        # Persist run id so evaluate_model.py can attach to the same run.
        (model_dir / "mlflow_run_id.txt").write_text(run.info.run_id)

    print(
        f"train: best_val_auc={best_trial['auc']:.4f}, "
        f"evals={len(trials.results)}, "
        f"saved={cfg.model.path}"
    )


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
