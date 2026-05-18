"""Register a trained joblib model into the BentoML local store.

Run once after `dvc repro` (which writes models/xgboost) and before
`bentoml serve` (which loads `xgboost:latest` from the store).
"""

from __future__ import annotations

import bentoml
import hydra
import joblib
from omegaconf import DictConfig


def save_model(cfg: DictConfig) -> bentoml.Model:
    model = joblib.load(cfg.model.path)
    # signatures={"predict": ...} exposes runner.predict.run(X) on the loaded
    # runner. Without this, BentoML tries to call the model directly which
    # XGBClassifier doesn't support.
    return bentoml.picklable_model.save_model(
        cfg.model.name,
        model,
        signatures={"predict": {"batchable": False}},
    )


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg: DictConfig) -> None:
    saved = save_model(cfg)
    print(f"saved: {saved.tag}")


if __name__ == "__main__":
    main()
