"""Evidently data drift report — Phase 9.

Compares a *reference* sample (the training distribution) against a *production*
sample that has been synthetically shifted to simulate three plausible
deteriorations of the live population:

* `MonthlyCharges` inflated by ``shift_monthly_pct`` (default +15%) — repricing
  or a market-wide pricing change.
* `Contract` distribution skewed toward Month-to-month at
  ``month_to_month_share`` (default 0.70) — a marketing/acquisition shift
  toward short-commitment customers.
* `tenure` multiplicatively compressed by ``tenure_shrink_factor`` (default
  0.70) — simulates a younger cohort skewed toward newer, shorter-tenured
  customers. Scaling (rather than subtract-and-clip) keeps the shift strictly
  downward, never produces negative tenure or a mass spike at zero, and drops
  no rows.

The script writes ``reports/drift.html`` (human-readable) and
``reports/drift.json`` (machine-readable, consumed by the tests and intended
for a future CI drift gate — not yet wired into any workflow).

The target column is intentionally **dropped before the report runs** —
production monitoring has no ground-truth label at scoring time, so including
`Churn` in the drift comparison would be both leakage-flavored and would
distort the dataset-level drift share denominator.

Run from the repo root::

    python -m monitoring.drift_report

Override the synthetic shift magnitudes via Hydra::

    python -m monitoring.drift_report monitoring.shift_monthly_pct=0.30
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from training.src.process import clean, load_raw

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Lookup by Evidently metric class name rather than positional index — the
# DataDriftPreset has expanded into two metrics on 0.4.x but the ordering is
# not part of the documented API and could shift on patch bumps.
_DATASET_METRIC = "DatasetDriftMetric"
_TABLE_METRIC = "DataDriftTable"


def build_reference_and_production(
    df_clean: pd.DataFrame,
    cfg: DictConfig,
    shift_monthly_pct: float = 0.15,
    month_to_month_share: float = 0.70,
    tenure_shrink_factor: float = 0.70,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a reference + shifted-production pair from the cleaned frame.

    The reference is the *training* split (same `cfg.split` params as
    `process.py`) so the drift report compares production-shaped data against
    what the model actually saw. The production sample is the held-out test
    split with the three synthetic shifts applied.

    The tenure shift multiplies tenure by ``tenure_shrink_factor`` (<1) rather
    than subtracting a constant and clipping/filtering. Scaling moves the whole
    distribution down monotonically — the correct "newer cohort" direction —
    with no mass spike at zero, no negative values, and no rows dropped.
    (Subtract-and-clip would pile a spike on zero; subtract-and-filter would
    drop the low-tenure rows and perversely raise the mean.)
    """
    target = cfg.process.target
    train_df, test_df = train_test_split(
        df_clean,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        stratify=df_clean[target] if cfg.split.stratify else None,
    )
    reference = train_df.reset_index(drop=True).copy()
    production = test_df.reset_index(drop=True).copy()

    production["MonthlyCharges"] = production["MonthlyCharges"] * (
        1.0 + shift_monthly_pct
    )

    rng = np.random.default_rng(random_state)
    other_share = (1.0 - month_to_month_share) / 2.0
    production["Contract"] = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=len(production),
        p=[month_to_month_share, other_share, other_share],
    )

    production["tenure"] = (
        (production["tenure"] * tenure_shrink_factor).round().astype(int)
    )
    return reference, production


def generate_drift_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    target: str,
    html_path: Path,
    json_path: Path,
) -> dict:
    """Run Evidently DataDriftPreset and persist HTML + JSON outputs.

    The target column is dropped from both frames before the report runs (see
    module docstring). Returns the in-memory report dict so callers (tests +
    CI gates) can assert on it without re-parsing the JSON file.
    """
    ref_features = reference.drop(columns=[target])
    prod_features = production.drop(columns=[target])

    categorical_features = [
        c for c in ref_features.columns if c not in NUMERICAL_FEATURES
    ]
    # target=None — we already removed the label column. Telling Evidently
    # `target=target` would have it search for a column we just dropped.
    column_mapping = ColumnMapping(
        target=None,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=categorical_features,
    )
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=ref_features,
        current_data=prod_features,
        column_mapping=column_mapping,
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(html_path))
    payload = report.as_dict()
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    return payload


def _metrics_by_name(report_dict: dict) -> dict:
    """Map Evidently metric class name → result dict. Raises if expected
    metrics are missing — preferable to silently reading the wrong block on
    a future patch bump."""
    by_name = {m["metric"]: m["result"] for m in report_dict["metrics"]}
    missing = {_DATASET_METRIC, _TABLE_METRIC} - by_name.keys()
    if missing:
        raise KeyError(
            f"expected Evidently metrics {sorted(missing)} not in report; "
            f"got {sorted(by_name)}"
        )
    return by_name


def _summary(report_dict: dict) -> str:
    """One-line drift summary — useful in CI logs and as a future CI gate
    parse target. Includes the boolean `dataset_drift` verdict which is the
    actual thing a deploy gate would key on."""
    by_name = _metrics_by_name(report_dict)
    overall = by_name[_DATASET_METRIC]
    by_column = by_name[_TABLE_METRIC]["drift_by_columns"]
    drifted = [
        c for c, info in by_column.items() if info.get("drift_detected")
    ]
    return (
        f"drift: {overall['number_of_drifted_columns']}/"
        f"{overall['number_of_columns']} columns drifted "
        f"(share={overall['share_of_drifted_columns']:.2f}, "
        f"dataset_drift={overall['dataset_drift']}); "
        f"drifted: {drifted}"
    )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    raw = load_raw(to_absolute_path(cfg.raw.path))
    cleaned = clean(raw, list(cfg.process.drop), cfg.process.target)

    # Pull shift magnitudes from cfg if a `monitoring:` group is wired in,
    # otherwise use the function defaults. OmegaConf.select returns None for
    # missing nodes; we filter None so build_*(**kwargs) doesn't override
    # defaults with None.
    overrides = {
        k: OmegaConf.select(cfg, f"monitoring.{k}")
        for k in (
            "shift_monthly_pct",
            "month_to_month_share",
            "tenure_shrink_factor",
            "random_state",
        )
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    reference, production = build_reference_and_production(
        cleaned, cfg, **overrides
    )

    reports_dir = Path(to_absolute_path("reports"))
    html_path = reports_dir / "drift.html"
    json_path = reports_dir / "drift.json"
    payload = generate_drift_report(
        reference,
        production,
        cfg.process.target,
        html_path,
        json_path,
    )
    print(_summary(payload))
    print(f"wrote: {html_path}")
    print(f"wrote: {json_path}")


if __name__ == "__main__":
    main()
