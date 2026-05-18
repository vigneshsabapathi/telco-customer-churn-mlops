"""Streamlit form posting JSON to the running BentoML service.

Usage:
    bentoml serve application/src/create_service.py:service --reload   # in one terminal
    streamlit run application/src/create_app.py                        # in another

Set the API_URL env var if the service runs anywhere other than localhost:3000.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("CHURN_API_URL", "http://localhost:3000/predict")

CONTRACT_OPTIONS = ("Month-to-month", "One year", "Two year")
INTERNET_OPTIONS = ("DSL", "Fiber optic", "No")
INTERNET_DEPENDENT_OPTIONS = ("No", "Yes", "No internet service")
PHONE_DEPENDENT_OPTIONS = ("No", "Yes", "No phone service")
PAYMENT_OPTIONS = (
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
)

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊")
st.title("Telco Customer Churn Predictor")
st.caption(f"POSTing to **{API_URL}**")

with st.form("customer_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("gender", ("Female", "Male"))
        SeniorCitizen = st.selectbox("SeniorCitizen", (0, 1))
        Partner = st.selectbox("Partner", ("Yes", "No"))
        Dependents = st.selectbox("Dependents", ("Yes", "No"))
        tenure = st.number_input("tenure (months)", min_value=0, value=12)
        PhoneService = st.selectbox("PhoneService", ("Yes", "No"))
        MultipleLines = st.selectbox("MultipleLines", PHONE_DEPENDENT_OPTIONS)
        InternetService = st.selectbox("InternetService", INTERNET_OPTIONS)
        OnlineSecurity = st.selectbox(
            "OnlineSecurity", INTERNET_DEPENDENT_OPTIONS
        )
        OnlineBackup = st.selectbox("OnlineBackup", INTERNET_DEPENDENT_OPTIONS)

    with col2:
        DeviceProtection = st.selectbox(
            "DeviceProtection", INTERNET_DEPENDENT_OPTIONS
        )
        TechSupport = st.selectbox("TechSupport", INTERNET_DEPENDENT_OPTIONS)
        StreamingTV = st.selectbox("StreamingTV", INTERNET_DEPENDENT_OPTIONS)
        StreamingMovies = st.selectbox(
            "StreamingMovies", INTERNET_DEPENDENT_OPTIONS
        )
        Contract = st.selectbox("Contract", CONTRACT_OPTIONS)
        PaperlessBilling = st.selectbox("PaperlessBilling", ("Yes", "No"))
        PaymentMethod = st.selectbox("PaymentMethod", PAYMENT_OPTIONS)
        MonthlyCharges = st.number_input(
            "MonthlyCharges", min_value=0.0, value=70.0
        )
        TotalCharges = st.number_input(
            "TotalCharges", min_value=0.0, value=840.0
        )

    submitted = st.form_submit_button("Predict churn")

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Service call failed: {exc}")
    else:
        prediction = response.json()
        churn_class = int(prediction[0])
        if churn_class == 1:
            st.error("Prediction: **churn likely** (class 1)")
        else:
            st.success("Prediction: **retained** (class 0)")
        st.json(payload)
