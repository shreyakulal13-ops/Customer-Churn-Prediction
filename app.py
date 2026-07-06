import streamlit as st
import pandas as pd
import joblib


model = joblib.load("LogisticRegression.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Customer Churn Prediction")
st.markdown("provide the following details to predict the risk of customer churn")

gender=st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen=st.selectbox("Senior Citizen", ["No", "Yes"])
Partner=st.selectbox("Partner", ["No", "Yes"])
Dependents=st.selectbox("Dependents", ["No", "Yes"])
PhoneService=st.selectbox("Phone Service", ["No", "Yes"])
MultipleLines=st.selectbox("Multiple Lines", ["No", "Yes"])
InternetService=st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity=st.selectbox("Online Security", ["No", "Yes"])
OnlineBackup=st.selectbox("Online Backup", ["No", "Yes"])
DeviceProtection=st.selectbox("Device Protection", ["No", "Yes"])
TechSupport=st.selectbox("Tech Support", ["No", "Yes"])
StreamingTV=st.selectbox("Streaming TV", ["No", "Yes"])
StreamingMovies=st.selectbox("Streaming Movies", ["No", "Yes"])
Contract=st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling=st.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod=st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges=st.number_input("Monthly Charges ($)", 0.0, 100.0, 50.0)
TotalCharges=st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
tenure = st.number_input("Tenure (Months)", 0, 72, 12)

if st.button("Predict"):
    raw_input={
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'tenure': tenure
    }	
    
  # Convert dictionary to DataFrame
input_df = pd.DataFrame([raw_input])

    # One-hot encode
input_df = pd.get_dummies(input_df)

# Add any missing columns
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Keep columns in the same order as training
input_df = input_df[columns]

# Scale numerical columns
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Predict
prediction = model.predict(input_df)[0]
            
if prediction == 1:
        st.error("⚠️ The customer is likely to churn.")
else:
    st.success("✅ The customer is likely to stay.")