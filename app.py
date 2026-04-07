# app.py
import streamlit as st
import pandas as pd
from model import get_trained_model

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")

# Load & cache model
@st.cache_resource
def load_model():
    return get_trained_model("telco.csv")

model, columns = load_model()

st.write("Enter customer details:")

# Dynamic input fields
input_data = {}
for col in columns:
    if col == "tenure":
        input_data[col] = st.slider(col, 0, 72)
    elif col in ["MonthlyCharges", "TotalCharges"]:
        input_data[col] = st.number_input(col, value=0.0)
    else:
        # Show dropdown for categorical features (improve UX)
        input_data[col] = st.text_input(col, value="")

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer likely to churn ({prob:.2f})")
    else:
        st.success(f"✅ Customer will stay ({prob:.2f})")
