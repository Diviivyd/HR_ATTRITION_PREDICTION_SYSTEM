import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("attrition_model.pkl")

st.set_page_config(page_title="HR Attrition Dashboard", layout="centered")

# ===== HEADER =====
st.markdown("<h1 style='text-align: center;'>🧠 HR Attrition Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Predict employee attrition risk in real time</p>", unsafe_allow_html=True)

st.divider()

# ===== INPUT SECTION =====
st.subheader("📥 Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 60, 30)
    income = st.number_input("Monthly Income", 1000, 200000, 20000)
    years = st.number_input("Years at Company", 0, 40, 3)

with col2:
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    overtime = 1 if overtime == "Yes" else 0

    satisfaction = st.slider("Job Satisfaction (1 = Low, 4 = High)", 1, 4, 2)

st.divider()

# ===== PREDICTION =====
features = np.array([[age, income, years, overtime, satisfaction]])

if st.button("🚀 Predict Attrition Risk"):

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    # ===== RESULT CARD =====
    st.subheader("📊 Prediction Result")

    # Risk level logic
    if probability < 0.3:
        risk = "🟢 Low Risk (Safe)"
        color = "green"
    elif probability < 0.6:
        risk = "🟡 Medium Risk (Watch)"
        color = "orange"
    else:
        risk = "🔴 High Risk (Leave likely)"
        color = "red"

    st.markdown(f"### {risk}")

    st.markdown("### 📊 Attrition Probability")

    st.progress(float(probability))

    st.write(f"Risk Score: **{probability:.2f}**")