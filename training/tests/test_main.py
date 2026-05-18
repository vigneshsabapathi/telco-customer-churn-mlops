"""Tests for training.src.main.run_pipeline(cfg).

Contract under test:
- run_pipeline(cfg) calls process_data, train, and evaluate exactly once each,
  in that order. (The actual implementations are tested in their own modules;
  here we only verify the orchestration contract.)
"""

from __future__ import annotations

from omegaconf import OmegaConf


def _stub_cfg():
    return OmegaConf.create({"_stub": True})


def test_main_runs_three_stages_in_order(monkeypatch):
    calls = []

    def fake_process(cfg):
        calls.append("process")

    def fake_train(cfg):
        calls.append("train")

    def fake_evaluate(cfg):
        calls.append("evaluate")
        return {}

    monkeypatch.setattr("training.src.process.process_data", fake_process)
    monkeypatch.setattr("training.src.train_model.train", fake_train)
    monkeypatch.setattr("training.src.evaluate_model.evaluate", fake_evaluate)

    from training.src.main import run_pipeline

    run_pipeline(_stub_cfg())

    assert calls == [
        "process",
        "train",
        "evaluate",
    ], f"main must orchestrate in order; got {calls}"


def test_main_passes_cfg_through_unchanged(monkeypatch):
    captured = []

    monkeypatch.setattr(
        "training.src.process.process_data",
        lambda cfg: captured.append(("process", cfg)),
    )
    monkeypatch.setattr(
        "training.src.train_model.train",
        lambda cfg: captured.append(("train", cfg)),
    )
    monkeypatch.setattr(
        "training.src.evaluate_model.evaluate",
        lambda cfg: captured.append(("evaluate", cfg)) or {},
    )

    cfg = _stub_cfg()
    from training.src.main import run_pipeline

    run_pipeline(cfg)

    # All three stages receive the same cfg object — no mutation/wrapping.
    assert all(c[1] is cfg for c in captured), "cfg identity must be preserved"
