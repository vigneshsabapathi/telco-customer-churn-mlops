"""Tests for application.src.save_model_to_bentoml.

Uses an isolated BENTOML_HOME tmp dir so the test doesn't pollute the
user's real bentoml store. Trains a fast model, runs save, and verifies
the saved model can be loaded back by name from the temp store.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize


@pytest.fixture(scope="module")
def _bento_home(tmp_path_factory):
    """Module-scoped, properly-reverted BENTOML_HOME override.

    Uses MonkeyPatch.context() so the env mutation is undone when the
    module finishes — avoids the os.environ mutation footgun where a
    subsequent test module silently uses the deleted tmp store.
    """
    tmp_bento_home = tmp_path_factory.mktemp("bentoml_home")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BENTOML_HOME", str(tmp_bento_home))
        yield tmp_bento_home


@pytest.fixture(scope="module")
def cfg(tmp_path_factory, _bento_home):
    tmp_models = tmp_path_factory.mktemp("models")
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(
            config_name="main",
            overrides=[
                f"model.dir={tmp_models.as_posix()}",
                "model.max_evals=3",
            ],
        )
    return cfg


def _repo_root() -> Path:
    """Anchor path resolution to the repo root regardless of pytest cwd."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def trained(cfg):
    """Train a model to disk so save_model_to_bentoml has something to register."""
    from training.src.train_model import train

    # Resolve processed CSVs relative to the repo root, not the current cwd,
    # so the existence check doesn't false-skip when invoked from elsewhere.
    repo_root = _repo_root()
    for key in ("X_train", "X_test", "y_train", "y_test"):
        rel = cfg.processed[key].path
        if not (repo_root / rel).exists():
            pytest.skip(
                f"Processed CSV missing at {repo_root / rel} "
                "— run process.py first."
            )

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

    repo_root = _repo_root()
    X_test = pd.read_csv(repo_root / cfg.processed.X_test.path).head(1)
    # BentoML 1.2: get(tag).to_runner() + runner.predict.run(X) (the
    # signatures={"predict": ...} from save_model exposes this method).
    runner = bentoml.picklable_model.get(
        f"{cfg.model.name}:latest"
    ).to_runner()
    runner.init_local()
    preds = runner.predict.run(X_test.values)
    assert len(preds) == 1
    assert int(preds[0]) in {0, 1}
