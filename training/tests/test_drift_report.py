"""Phase 9 smoke test for the Evidently drift script.

End-to-end: load the raw CSV via the same Hydra config training uses, run the
drift script, and assert that the output artifacts exist *and* that the
synthetic shifts actually register as drift in the JSON payload. This catches
both "the script crashed" and "the report ran but found no drift" — the
latter would mean the synthetic shifter or the column mapping is broken.

Also pins the invariant that the reference frame here matches the X_train
split produced by `process.py` (same `cfg.split.*` params, same row order) —
without that, the drift report compares a *different* split against
production and silently understates real-world drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from sklearn.model_selection import train_test_split

from monitoring.drift_report import (
    _DATASET_METRIC,
    _TABLE_METRIC,
    build_reference_and_production,
    generate_drift_report,
)
from training.src.process import clean, load_raw

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
RAW_CSV = REPO_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


@pytest.fixture(scope="module")
def cfg():
    if not RAW_CSV.exists():
        pytest.skip(
            f"raw CSV not present at {RAW_CSV}; run `python -m training.src.process` first"
        )
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="main")


@pytest.fixture(scope="module")
def cleaned(cfg):
    raw = load_raw(str(RAW_CSV))
    return clean(raw, list(cfg.process.drop), cfg.process.target)


def _metrics_by_name(payload):
    return {m["metric"]: m["result"] for m in payload["metrics"]}


def test_build_production_sample_applies_three_shifts(cleaned, cfg):
    reference, production = build_reference_and_production(cleaned, cfg)

    assert len(reference) > 0 and len(production) > 0
    assert set(reference.columns) == set(production.columns)

    ref_monthly_mean = reference["MonthlyCharges"].mean()
    prod_monthly_mean = production["MonthlyCharges"].mean()
    assert (
        prod_monthly_mean > ref_monthly_mean * 1.10
    ), "MonthlyCharges shift did not register"

    ref_m2m = (reference["Contract"] == "Month-to-month").mean()
    prod_m2m = (production["Contract"] == "Month-to-month").mean()
    assert (
        prod_m2m > ref_m2m + 0.10
    ), f"Contract skew did not register (ref={ref_m2m:.2f}, prod={prod_m2m:.2f})"

    assert (
        production["tenure"].mean() < reference["tenure"].mean() * 0.9
    ), "tenure shift did not register a clear downward compression"
    # Multiplicative compression keeps every row (no filter, no clip) and stays
    # non-negative — no degenerate mass spike at zero.
    assert (production["tenure"] >= 0).all(), "negative tenure leaked"
    _, test_df = train_test_split(
        cleaned,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        stratify=cleaned[cfg.process.target] if cfg.split.stratify else None,
    )
    assert len(production) == len(
        test_df
    ), "multiplicative tenure shift must not drop any rows"


def test_reference_matches_process_train_split(cleaned, cfg):
    """Guard against drift between cfg.split.* and any future per-process
    split override. The drift report's reference must equal what process.py
    actually trained on, otherwise the comparison is meaningless."""
    target = cfg.process.target
    train_df, _ = train_test_split(
        cleaned,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        stratify=cleaned[target] if cfg.split.stratify else None,
    )
    expected_index = train_df.index.tolist()

    reference, _ = build_reference_and_production(cleaned, cfg)
    # build_reference_and_production resets the index to 0..N-1, so compare
    # by row order against the same reset on the canonical training split.
    canonical = train_df.reset_index(drop=True)
    assert len(reference) == len(canonical)
    # Compare a column that uniquely fingerprints a row across the dataset.
    assert (
        reference["TotalCharges"].values == canonical["TotalCharges"].values
    ).all(), (
        "drift report's reference does not match process.py's training split"
    )
    assert len(expected_index) == len(reference)


def test_drift_report_writes_html_and_json(tmp_path, cleaned, cfg):
    reference, production = build_reference_and_production(cleaned, cfg)
    html_path = tmp_path / "drift.html"
    json_path = tmp_path / "drift.json"

    payload = generate_drift_report(
        reference,
        production,
        cfg.process.target,
        html_path,
        json_path,
    )

    assert (
        html_path.exists() and html_path.stat().st_size > 50_000
    ), f"drift.html missing or implausibly small: {html_path.stat().st_size if html_path.exists() else 'absent'}"
    assert json_path.exists() and json_path.stat().st_size > 1_000
    on_disk = json.loads(json_path.read_text())
    assert on_disk == payload


def test_drift_report_excludes_target(tmp_path, cleaned, cfg):
    """The target column must not appear in the drift report's column table —
    a production monitor has no ground-truth label, and including Churn
    distorts the dataset-level drift share denominator."""
    target = cfg.process.target
    reference, production = build_reference_and_production(cleaned, cfg)
    payload = generate_drift_report(
        reference,
        production,
        target,
        tmp_path / "drift.html",
        tmp_path / "drift.json",
    )
    by_name = _metrics_by_name(payload)
    by_column = by_name[_TABLE_METRIC]["drift_by_columns"]
    assert (
        target not in by_column
    ), f"target {target!r} leaked into drift table — should have been dropped"
    overall = by_name[_DATASET_METRIC]
    assert overall["number_of_columns"] == len(reference.columns) - 1


def test_drift_report_detects_shifted_columns(tmp_path, cleaned, cfg):
    reference, production = build_reference_and_production(cleaned, cfg)
    payload = generate_drift_report(
        reference,
        production,
        cfg.process.target,
        tmp_path / "drift.html",
        tmp_path / "drift.json",
    )

    by_name = _metrics_by_name(payload)
    by_column = by_name[_TABLE_METRIC]["drift_by_columns"]
    overall = by_name[_DATASET_METRIC]

    assert (
        overall["number_of_drifted_columns"] > 0
    ), "overall drift count is zero on a deliberately shifted sample"
    for col in ("MonthlyCharges", "Contract", "tenure"):
        assert by_column[col]["drift_detected"], (
            f"expected drift on {col}; got "
            f"{by_column[col].get('drift_score')} via {by_column[col].get('stattest_name')}"
        )
