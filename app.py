import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import get_trained_model

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction App")
st.markdown("Predict if a customer is likely to churn based on their details.")

# Load model
@st.cache_resource
def load_model():
    return get_trained_model("telco.csv")

model, columns = load_model()

# Load dataset for dropdown options
df = pd.read_csv("telco.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.drop('customerID', axis=1)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Identify numeric and categorical
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'Churn']

categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

# Layout: 2 columns
col1, col2 = st.columns(2)
input_data = {}

# Numeric inputs
with col1:
    for col in numeric_cols:
        if col == "tenure":
            input_data[col] = st.slider("Tenure (months)", 0, 72)
        else:
            input_data[col] = st.number_input(col, value=0.0, step=1.0)

# Categorical dropdowns
with col2:
    for col in categorical_cols:
        if col in df.columns:
            options = df[col].dropna().unique().tolist()
            input_data[col] = st.selectbox(col, options)

# Convert to dataframe
input_df = pd.DataFrame([input_data])

st.markdown("---")

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    stay_prob = 1 - prob

    # Display metrics and chart side by side
    col3, col4 = st.columns([1, 1])
    
    with col3:
        if prediction == 1:
            st.metric("Prediction", "Likely to Churn ⚠️", delta=f"{prob:.0%} probability")
            st.error("Customer is likely to churn. Consider taking action!")
        else:
            st.metric("Prediction", "Will Stay ✅", delta=f"{stay_prob:.0%} probability")
            st.success("Customer is likely to stay. Keep up the engagement!")

    with col4:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.barh(['Stay', 'Churn'], [stay_prob, prob], color=['green', 'red'])
        ax.set_xlim(0,1)
        ax.set_xlabel("Probability")
        ax.set_title("Churn vs Stay Probability")
        st.pyplot(fig)

    # Input summary
    st.subheader("Customer Input Summary")
    st.table(input_df.T.rename(columns={0: "Value"}))
