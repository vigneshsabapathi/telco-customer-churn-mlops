"""Tests for application.src.save_model_to_bentoml.

Uses an isolated BENTOML_HOME tmp dir so the test doesn't pollute the
user's real bentoml store. Trains a fast model, runs save, and verifies
the saved model can be loaded back by name from the temp store.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from hydra import compose, initialize


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp_models = tmp_path_factory.mktemp("models")
    tmp_bento_home = tmp_path_factory.mktemp("bentoml_home")
    os.environ["BENTOML_HOME"] = str(tmp_bento_home)

    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(
            config_name="main",
            overrides=[
                f"model.dir={tmp_models.as_posix()}",
                "model.max_evals=3",
            ],
        )
    return cfg


@pytest.fixture(scope="module")
def trained(cfg):
    """Train a model to disk so save_model_to_bentoml has something to register."""
    from training.src.train_model import train

    for key in ("X_train", "X_test", "y_train", "y_test"):
        if not Path(cfg.processed[key].path).exists():
            pytest.skip("Processed CSVs missing — run process.py first.")

    train(cfg)
    return Path(cfg.model.path)


def test_save_registers_model_in_isolated_store(trained, cfg):
    """After save_model(cfg), `bentoml.picklable_model.get(name)` must succeed."""
    import bentoml

    from application.src.save_model_to_bentoml import save_model

    save_model(cfg)

    # If save worked, this lookup returns a non-None Model object.
    found = bentoml.picklable_model.get(f"{cfg.model.name}:latest")
    assert found is not None
    assert found.tag.name == cfg.model.name


def test_saved_model_loads_and_predicts(trained, cfg):
    """End-to-end: save → load runner → predict on test row."""
    import bentoml
    import pandas as pd

    from application.src.save_model_to_bentoml import save_model

    save_model(cfg)

    X_test = pd.read_csv(cfg.processed.X_test.path).head(1)
    # BentoML 1.2: get(tag).to_runner() + runner.predict.run(X) (the
    # signatures={"predict": ...} from save_model exposes this method).
    runner = bentoml.picklable_model.get(
        f"{cfg.model.name}:latest"
    ).to_runner()
    runner.init_local()
    preds = runner.predict.run(X_test.values)
    assert len(preds) == 1
    assert int(preds[0]) in {0, 1}
