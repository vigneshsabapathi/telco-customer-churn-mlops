"""HTTP integration tests against a running BentoML service.

These tests POST to /predict on a live server. They auto-skip if no server
is reachable, so they're safe to include in the default pytest suite.

To run against the local dev server:
    bentoml serve application.src.create_service:service --port 3000
    pytest application/tests/test_predict_endpoint.py -v

To run against a Dockerized container (Phase 7):
    docker run -p 3000:3000 churn_service:latest
    pytest application/tests/test_predict_endpoint.py -v

To target a different host:
    CHURN_API_URL=http://my-host:3000/predict pytest ...
"""

from __future__ import annotations

import os

import pytest
import requests

API_BASE = os.environ.get("CHURN_API_BASE", "http://localhost:3000")
PREDICT_URL = os.environ.get("CHURN_API_URL", f"{API_BASE}/predict")
HEALTH_URL = f"{API_BASE}/healthz"

# Two test profiles deliberately chosen to span opposite ends of the
# churn-risk spectrum based on Telco domain knowledge:
HIGH_RISK_CHURNING = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 85.0,
}

LOW_RISK_RETAINED = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 30.0,
    "TotalCharges": 1800.0,
}


@pytest.fixture(scope="module", autouse=True)
def _require_running_service():
    """Skip the whole module if no server is reachable at HEALTH_URL."""
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        if r.status_code != 200:
            pytest.skip(f"{HEALTH_URL} returned {r.status_code}; skipping")
    except requests.RequestException as exc:
        pytest.skip(f"No BentoML server reachable at {HEALTH_URL}: {exc}")


def _predict(payload: dict) -> int:
    r = requests.post(PREDICT_URL, json=payload, timeout=10)
    r.raise_for_status()
    body = r.json()
    assert (
        isinstance(body, list) and len(body) == 1
    ), f"unexpected body: {body!r}"
    return int(body[0])


def test_high_risk_customer_predicted_churn():
    """Month-to-month + tenure=1 + no add-ons + electronic check → churn likely."""
    assert (
        _predict(HIGH_RISK_CHURNING) == 1
    ), "High-risk profile should be classified as churn (1)"


def test_low_risk_customer_predicted_retained():
    """Two-year contract + tenure=60 + all add-ons + bank transfer → retained."""
    assert (
        _predict(LOW_RISK_RETAINED) == 0
    ), "Low-risk profile should be classified as retained (0)"


def test_endpoint_rejects_malformed_payload():
    """Missing required fields → 400 (Pydantic validation)."""
    bad = {"gender": "Female", "tenure": "not an int"}
    r = requests.post(PREDICT_URL, json=bad, timeout=10)
    assert r.status_code in (
        400,
        422,
    ), f"expected 400/422 for malformed payload; got {r.status_code}: {r.text[:200]}"
