import streamlit as st
import pandas as pd
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

# Define the categorical columns explicitly as per your list
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

    col3, col4 = st.columns(2)
    if prediction == 1:
        col3.metric("Prediction", "Likely to Churn ⚠️", delta=f"{prob:.0%} probability")
        st.error("Customer is likely to churn. Consider taking action!")
    else:
        col3.metric("Prediction", "Will Stay ✅", delta=f"{prob:.0%} probability")
        st.success("Customer is likely to stay. Keep up the engagement!")

    # Visual probability bar
    st.progress(int(prob * 100))

    # Input summary
    st.subheader("Customer Input Summary")
    st.table(input_df.T.rename(columns={0: "Value"}))
