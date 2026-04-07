import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('churn_model_pipeline.pkl')

st.title("📊 Customer Churn Prediction App")

st.write("Fill in customer details:")

# Example inputs (adjust based on your dataset)
tenure = st.slider("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Create input dataframe
input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract],
    'InternetService': [internet_service]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer will stay")
