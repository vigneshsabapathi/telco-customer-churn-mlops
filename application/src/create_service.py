"""BentoML service for churn predictions.

Layout:
- `Customer` Pydantic model — raw input shape (19 fields).
- `transform_data` / `transform_data_with_columns` — apply the same patsy
  formula used in training. Uses a `dummy_df` of reference rows covering
  every categorical level so single-row inputs produce the same column
  set as training.
- `predict_logic(customer, model_runner)` — pure prediction function;
  testable without spinning up BentoML.
- BentoML Service + Runner wired at the BOTTOM of this module so tests
  can import the helpers without triggering a model load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hydra import compose, initialize
from patsy import dmatrix
from pydantic import BaseModel

from training.src.process import rename_patsy_columns

# ---------- Hydra config (read once at import) -------------------------------

with initialize(version_base=None, config_path="../../config"):
    _cfg = compose(config_name="main")
    FEATURES = list(_cfg.process.features)
    MODEL_NAME = _cfg.model.name


# ---------- Pydantic input ---------------------------------------------------


class Customer(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "Yes"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 840.0


# ---------- dummy_df: lock the categorical level set --------------------------

# 4 reference rows that together cover every categorical level seen in training.
# Without this, patsy emits different columns for single-row inputs (missing
# levels → missing one-hot columns), breaking column-parity with X_train.csv.
_DUMMY_ROWS = pd.DataFrame(
    {
        "gender": ["Female", "Male", "Female", "Male"],
        "SeniorCitizen": ["0", "1", "0", "1"],
        "Partner": ["Yes", "No", "Yes", "No"],
        "Dependents": ["No", "Yes", "No", "Yes"],
        "tenure": [1, 2, 3, 4],
        "PhoneService": ["Yes", "No", "Yes", "Yes"],
        "MultipleLines": ["No", "Yes", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
        "OnlineSecurity": ["No", "Yes", "No internet service", "Yes"],
        "OnlineBackup": ["No", "Yes", "No internet service", "Yes"],
        "DeviceProtection": ["No", "Yes", "No internet service", "Yes"],
        "TechSupport": ["No", "Yes", "No internet service", "Yes"],
        "StreamingTV": ["No", "Yes", "No internet service", "Yes"],
        "StreamingMovies": ["No", "Yes", "No internet service", "Yes"],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
        ],
        "PaperlessBilling": ["No", "Yes", "No", "Yes"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        "MonthlyCharges": [0.0, 0.0, 0.0, 0.0],
        "TotalCharges": [0.0, 0.0, 0.0, 0.0],
    }
)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Match training-time clean(): SeniorCitizen → string."""
    df = df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    return df


def transform_data_with_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return the patsy-encoded + renamed DataFrame (keeps column names visible).

    Splits out from `transform_data` so tests can compare column names against
    `data/processed/X_train.csv` byte-for-byte.
    """
    real = _prepare(df)
    combined = pd.concat([real, _DUMMY_ROWS], ignore_index=True)
    formula = " + ".join(FEATURES) + " - 1"
    X = dmatrix(formula, combined, return_type="dataframe")
    X = rename_patsy_columns(X)
    # Keep only the real input rows (the first len(real) rows).
    return X.iloc[: len(real)].reset_index(drop=True)


def transform_data(df: pd.DataFrame) -> np.ndarray:
    """Production path — returns a numpy array shaped (n, 31) ready for predict."""
    return transform_data_with_columns(df).values


def predict_logic(customer: "Customer", model_runner) -> np.ndarray:
    """Pure prediction: testable without BentoML. `model_runner` is any object
    exposing `.run(X) -> array_like`."""
    # Pydantic v2 — .model_dump() (v1's .dict() emits PydanticDeprecatedSince20).
    df = pd.DataFrame([customer.model_dump()])
    X = transform_data(df)
    return np.asarray(model_runner.run(X))


# ---------- BentoML wiring (bottom — keeps the rest importable) --------------

try:
    import bentoml
    from bentoml.io import JSON, NumpyNdarray

    # BentoML 1.2 API: get(tag).to_runner() (load_runner was removed).
    # The signatures={"predict": ...} from save_model exposes runner.predict.run.
    _runner = bentoml.picklable_model.get(f"{MODEL_NAME}:latest").to_runner()
    service = bentoml.Service("churn_service", runners=[_runner])

    class _RunnerAdapter:
        """Adapter so predict_logic's `model_runner.run(X)` works with the
        BentoML runner's `runner.predict.run(X)` shape."""

        def __init__(self, r):
            self._r = r

        def run(self, X):
            return self._r.predict.run(X)

    @service.api(input=JSON(pydantic_model=Customer), output=NumpyNdarray())
    def predict(customer: Customer) -> np.ndarray:
        return predict_logic(customer, _RunnerAdapter(_runner))

except Exception as _e:  # pragma: no cover  # noqa: BLE001
    # Helpers (Customer, transform_data, predict_logic) stay usable even if
    # BentoML can't initialize a Service (e.g. no model in store during tests).
    # We still log the failure to stderr so production misconfigurations
    # (wrong tag, corrupted runner, etc.) leave a breadcrumb instead of
    # silently producing a module with no `service` attribute.
    import sys

    print(
        f"create_service: BentoML service init skipped ({type(_e).__name__}: {_e})",
        file=sys.stderr,
    )
