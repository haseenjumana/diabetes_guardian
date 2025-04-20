import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="🩺 HealthGuard UAE - Diabetes AI", layout="centered")

st.title("🌟 HealthGuard UAE - Diabetes AI Agent")
st.markdown("Helping you stay healthy and informed 🇦🇪")

# Load the model using an absolute path with error handling
model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Error: The model file 'diabetes_model.pkl' was not found.")
    st.stop()  # Stop the app if the model file is not found
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()  # Stop the app if any other error occurs

st.header("🔍 Predict Your Diabetes Risk")
glucose = st.slider("Glucose Level", 0, 200, 100)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
age = st.slider("Age", 10, 100, 30)
insulin = st.slider("Insulin", 0, 846, 80)
bp = st.slider("Blood Pressure", 40, 130, 80)
skin = st.slider("Skin Thickness", 0, 100, 20)
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

if st.button("🧠 Predict"):
    user_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(user_data)
    result = "🟢 No Diabetes Risk" if prediction[0] == 0 else "🔴 Risk of Diabetes"
    st.success(f"🧾 Result: **{result}**")

    if prediction[0] == 1:
        st.warning("💡 Suggestion: Maintain a healthy lifestyle. Reduce sugar, walk daily, and stay hydrated.")
        st.markdown("### 🏥 Nearby Diabetes Clinics in UAE")
        st.info("- Dubai Diabetes Center\n- Cleveland Clinic Abu Dhabi\n- Mediclinic City Hospital\n- NMC Specialty Hospital")
