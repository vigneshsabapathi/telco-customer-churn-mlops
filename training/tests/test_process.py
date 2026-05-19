"""Schema + pipeline tests for training.src.process.

Uses pytest-steps to walk load → clean → encode → split and assert on each
intermediate state. The raw CSV is validated against a Pandera schema so a
shape change in the source data fails loudly.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
import pytest
from hydra import compose, initialize
from pytest_steps import test_steps

from training.src.process import (
    clean,
    encode_features,
    load_raw,
    rename_patsy_columns,
)

# --- Pandera schema for the raw CSV (pre-coerce) -------------------------------------

RawTelcoSchema = pa.DataFrameSchema(
    {
        "customerID": pa.Column(
            str, pa.Check.str_matches(r"^\d{4}-[A-Z]{5}$")
        ),
        "gender": pa.Column(str, pa.Check.isin(["Female", "Male"])),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1])),
        "Partner": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "Dependents": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "tenure": pa.Column(int, pa.Check.ge(0)),
        "PhoneService": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        # MultipleLines, InternetService etc. have richer categorical sets;
        # we don't enumerate them to keep the schema maintenance light.
        "MultipleLines": pa.Column(str),
        "InternetService": pa.Column(str),
        "OnlineSecurity": pa.Column(str),
        "OnlineBackup": pa.Column(str),
        "DeviceProtection": pa.Column(str),
        "TechSupport": pa.Column(str),
        "StreamingTV": pa.Column(str),
        "StreamingMovies": pa.Column(str),
        "Contract": pa.Column(
            str, pa.Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "PaymentMethod": pa.Column(str),
        "MonthlyCharges": pa.Column(float, pa.Check.ge(0)),
        # TotalCharges arrives as object dtype because of ~11 whitespace rows.
        # We assert object (str) here — coercion happens in clean().
        "TotalCharges": pa.Column(str),
        "Churn": pa.Column(str, pa.Check.isin(["Yes", "No"])),
    },
    strict=True,
)


# --- Hydra config fixture ------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    with initialize(version_base=None, config_path="../../config"):
        return compose(config_name="main")


# --- pytest-steps pipeline walk ------------------------------------------------------


@test_steps("load", "validate_raw", "clean", "encode", "rename")
def test_process_pipeline_steps(cfg):
    # State persists across yields via the generator's closure — no steps_data
    # holder needed in pytest-steps 1.8 generator mode.

    # Step 1: load — read the raw CSV
    raw = load_raw(cfg.raw.path)
    assert isinstance(raw, pd.DataFrame)
    assert raw.shape == (7043, 21), f"raw shape: {raw.shape}"
    yield  # ----- "load" passed

    # Step 2: validate_raw — Pandera schema check on the as-loaded frame
    validated = RawTelcoSchema.validate(raw)
    assert len(validated) == 7043
    yield  # ----- "validate_raw" passed

    # Step 3: clean — coerce TotalCharges, drop NaN rows, map Churn, cast SeniorCitizen
    cleaned = clean(raw, list(cfg.process.drop), cfg.process.target)
    assert "customerID" not in cleaned.columns, "customerID must be dropped"
    assert (
        cleaned["TotalCharges"].dtype.kind == "f"
    ), "TotalCharges must be float after clean"
    assert cleaned["Churn"].dtype.kind == "i", "Churn must be int after clean"
    assert set(cleaned["Churn"].unique()) == {0, 1}
    assert (
        cleaned["SeniorCitizen"].dtype == object
    ), "SeniorCitizen must be cast to string before patsy"
    assert (
        len(cleaned) == 7032
    ), f"expected 7032 rows after dropping 11 NaN; got {len(cleaned)}"
    yield  # ----- "clean" passed

    # Step 4: encode — patsy emits one-hot columns with `Col[T.value]` format
    X = encode_features(cleaned, list(cfg.process.features))
    assert X.shape[0] == 7032
    # 19 raw features → 31 encoded columns for the with_contract variant
    assert X.shape[1] == 31, f"expected 31 encoded features; got {X.shape[1]}"
    yield  # ----- "encode" passed

    # Step 5: rename — no `[T.` or `]` should survive in column names
    bad = [c for c in X.columns if "[T." in c or "]" in c]
    assert not bad, f"unsanitized patsy columns: {bad}"
    yield  # ----- "rename" passed (pytest-steps needs N yields for N steps)


def test_clean_drops_exactly_eleven_rows(cfg):
    """The Telco-specific 'TotalCharges whitespace' gotcha — must drop 11 rows, not 0 or 50."""
    raw = load_raw(cfg.raw.path)
    cleaned = clean(raw, list(cfg.process.drop), cfg.process.target)
    assert (
        len(raw) - len(cleaned) == 11
    ), f"expected 11 NaN drops, got {len(raw) - len(cleaned)}"


def test_rename_handles_bracketed_patsy_columns():
    """Pure function test — verifies the regex without needing real data.

    Covers both forms patsy emits:
    - `Col[T.value]` (treatment-coded categorical, most common)
    - `Col[value]`   (bare bracketed form, second .replace() branch)
    """
    df = pd.DataFrame(
        {
            "Contract[T.One year]": [1, 0],
            "PaymentMethod[T.Mailed check]": [0, 1],
            "OtherCol[bareform]": [
                1,
                0,
            ],  # bare bracket — exercises the 2nd .replace
            "tenure": [12, 24],  # numeric — unchanged
        }
    )
    out = rename_patsy_columns(df)
    assert list(out.columns) == [
        "Contract_One year",
        "PaymentMethod_Mailed check",
        "OtherCol_bareform",
        "tenure",
    ]


def test_stratification_preserves_class_balance(cfg):
    """The stratify=True flag must keep train/test churn rates within tolerance."""
    from sklearn.model_selection import train_test_split

    raw = load_raw(cfg.raw.path)
    cleaned = clean(raw, list(cfg.process.drop), cfg.process.target)
    X = encode_features(cleaned, list(cfg.process.features))
    y = cleaned[cfg.process.target]

    _, _, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        stratify=y if cfg.split.stratify else None,
    )

    overall_rate = y.mean()
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    # Stratified split must keep both within ±0.5 pp of the overall rate.
    assert (
        abs(train_rate - overall_rate) < 0.005
    ), f"train churn rate {train_rate:.4f} drifted from overall {overall_rate:.4f}"
    assert (
        abs(test_rate - overall_rate) < 0.005
    ), f"test churn rate {test_rate:.4f} drifted from overall {overall_rate:.4f}"
