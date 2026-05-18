"""Unit tests for the BentoML service module.

We test what's testable in-process — the Pydantic input model, the
transform_data column-parity contract, and the prediction logic with a
mock model. The actual BentoML Service + Runner wiring is exercised in
Phase 7 via a live `bentoml serve` smoke test.

Important: this test file imports `Customer`, `transform_data`, and
`predict_logic` from `application.src.create_service`. The module must
keep the BentoML `load_runner` + `Service` construction at the BOTTOM so
that importing for tests doesn't try to load a model from the BentoML
store.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pydantic import ValidationError

# ----- Customer Pydantic model ----------------------------------------------


def test_customer_accepts_default_values():
    from application.src.create_service import Customer

    c = Customer()  # all fields have defaults
    assert c.tenure >= 0
    assert c.Contract in {"Month-to-month", "One year", "Two year"}


def test_customer_accepts_realistic_row():
    from application.src.create_service import Customer

    c = Customer(
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=24,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="Yes",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="Yes",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=70.0,
        TotalCharges=1500.0,
    )
    assert c.gender == "Female"
    assert c.tenure == 24


def test_customer_rejects_wrong_types():
    from application.src.create_service import Customer

    with pytest.raises(ValidationError):
        Customer(tenure="not-an-int")
    with pytest.raises(ValidationError):
        Customer(MonthlyCharges="abc")


# ----- transform_data column parity -----------------------------------------


def _read_training_columns() -> list:
    """The 31 columns that process.py emits — the inference-time transform
    MUST produce the same set in the same order."""
    x_train_path = Path(
        "C:/Data/Project/churn-mlops/data/processed/X_train.csv"
    )
    if not x_train_path.exists():
        pytest.skip("X_train.csv missing — run process.py / dvc repro first.")
    return list(pd.read_csv(x_train_path, nrows=1).columns)


def test_transform_data_produces_31_columns():
    from application.src.create_service import Customer, transform_data

    c = Customer()
    df = pd.DataFrame([c.dict()])
    X = transform_data(df)

    # Numpy ndarray of shape (1, 31)
    assert X.shape == (1, 31), f"expected (1, 31); got {X.shape}"


def test_transform_data_column_names_match_training():
    """Critical parity guarantee — column names match X_train.csv byte for byte."""
    from application.src.create_service import (
        Customer,
        transform_data_with_columns,
    )

    expected = _read_training_columns()
    c = Customer()
    df = pd.DataFrame([c.dict()])
    X_df = transform_data_with_columns(df)

    assert list(X_df.columns) == expected, (
        "column mismatch:\n"
        f"missing from inference: {set(expected) - set(X_df.columns)}\n"
        f"extra at inference: {set(X_df.columns) - set(expected)}"
    )


def test_transform_data_handles_multiple_categorical_levels():
    """A row using different category values for every categorical must
    still produce the same column set (dummy_df ensures all levels present)."""
    from application.src.create_service import (
        Customer,
        transform_data_with_columns,
    )

    rare_combo = Customer(
        gender="Male",
        SeniorCitizen=1,
        Partner="No",
        Dependents="Yes",
        MultipleLines="No phone service",
        InternetService="DSL",
        OnlineSecurity="No internet service",
        Contract="Two year",
        PaymentMethod="Mailed check",
    )
    df = pd.DataFrame([rare_combo.dict()])
    X_df = transform_data_with_columns(df)

    expected = _read_training_columns()
    assert list(X_df.columns) == expected, (
        f"unusual-input column drift: "
        f"{set(expected).symmetric_difference(set(X_df.columns))}"
    )


# ----- predict_logic with mock model ----------------------------------------


def test_predict_logic_returns_class_label():
    """predict_logic should return a numpy array containing 0 or 1."""
    from application.src.create_service import Customer, predict_logic

    fake_runner = MagicMock()
    fake_runner.run.return_value = [1]  # mimics XGBClassifier.predict output

    c = Customer()
    result = predict_logic(c, fake_runner)

    # The model is called exactly once with a (1, 31) array
    assert fake_runner.run.called
    (called_arg,), _ = fake_runner.run.call_args
    assert called_arg.shape == (1, 31)

    # Return value is array-like with values in {0, 1}
    assert int(result[0]) in {0, 1}
