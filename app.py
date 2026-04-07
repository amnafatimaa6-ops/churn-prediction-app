import streamlit as st
import pandas as pd
from model import get_trained_model

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction App")
st.markdown("Predict if a customer is likely to churn based on their details.")

# Load & cache model
@st.cache_resource
def load_model():
    return get_trained_model("telco.csv")

model, columns = load_model()

st.header("Enter Customer Details")

# Separate numeric and categorical columns dynamically
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = [col for col in columns if col not in numeric_cols]

# Split UI into two columns
col1, col2 = st.columns(2)
input_data = {}

with col1:
    for col in numeric_cols:
        if col == "tenure":
            input_data[col] = st.slider("Tenure (months)", 0, 72)
        else:
            input_data[col] = st.number_input(col, value=0.0, step=1.0)

with col2:
    for col in categorical_cols:
        # Example: gender dropdown
        if col.lower() == "gender":
            input_data[col] = st.selectbox("Gender", ["Male", "Female"])
        else:
            # Default: text input turned into selectbox if possible
            input_data[col] = st.text_input(col, value="")

input_df = pd.DataFrame([input_data])

st.markdown("---")

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Show metrics visually
    col3, col4 = st.columns(2)
    if prediction == 1:
        col3.metric("Prediction", "Likely to Churn ⚠️", delta=f"{prob:.0%} probability")
        st.error("Customer is likely to churn. Consider taking action!")
    else:
        col3.metric("Prediction", "Will Stay ✅", delta=f"{prob:.0%} probability")
        st.success("Customer is likely to stay. Keep up the engagement!")

    # Optional: probability bar
    st.progress(int(prob*100))
