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

# Why the directional assertions (== 1 / == 0) are intentional

The high-risk and low-risk profiles are designed to be **unambiguous** —
Month-to-month + tenure=1 + no add-ons + Electronic check is a textbook
churn-likely customer; Two-year + tenure=60 + all add-ons + Bank transfer
is a textbook retained customer. Any properly-trained classifier with
ROC-AUC ≥ 0.78 (the gate enforced by test_evaluate_model.test_metrics_meet_baselines)
should reliably distinguish them. If a retrain flips either assertion,
that's a real signal worth catching — either the model degraded or the
training data drifted in a way that inverted the model's interpretation
of these features. The smoke test serves as a directional contract, not
a flaky-and-brittle assertion.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import pytest
import requests

API_BASE = os.environ.get("CHURN_API_BASE", "http://localhost:3000").rstrip(
    "/"
)
PREDICT_URL = os.environ.get("CHURN_API_URL", f"{API_BASE}/predict")
# Derive HEALTH_URL from PREDICT_URL's host:port so that setting only
# CHURN_API_URL (e.g. pointing at a remote container) doesn't leave HEALTH_URL
# pinned to localhost:3000 — which would cause the autouse fixture to
# spuriously skip the suite even when the remote server is reachable.
_predict_parts = urlparse(PREDICT_URL)
HEALTH_URL = f"{_predict_parts.scheme}://{_predict_parts.netloc}/healthz"

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
