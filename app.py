import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing files
model = joblib.load("LogisticRegression.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Title
st.title("Customer Churn Prediction")
st.markdown("Provide the following details to predict the risk of customer churn.")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])
PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 1000.0, 50.0)
TotalCharges = st.number_input("Total Charges ($)", 0.0, 100000.0, 500.0)
tenure = st.number_input("Tenure (Months)", 0, 72, 12)

# Prediction
if st.button("Predict"):

    # Raw input (KEEP CATEGORICAL VALUES - IMPORTANT)
    raw_input = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
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
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "tenure": tenure
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # One-hot encoding (must match training)
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct order
    input_df = input_df[columns]

    # Scale numerical columns
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Output
    if prediction == 1:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")

    