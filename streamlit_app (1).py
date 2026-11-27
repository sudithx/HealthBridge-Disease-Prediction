
import streamlit as st
import requests

st.title("HealthBridge Disease Prediction")
st.write("Enter patient details to get a prediction.")

age = st.number_input("Age", 1, 100, 25)
gender = st.selectbox("Gender", ["Female", "Male"])

st.subheader("Symptoms")
fever = st.checkbox("Fever")
cough = st.checkbox("Cough")
fatigue = st.checkbox("Fatigue")
breathlessness = st.checkbox("Breathlessness")
chest_pain = st.checkbox("Chest Pain")
nausea = st.checkbox("Nausea")
headache = st.checkbox("Headache")

st.subheader("Health measurements")
bmi = st.number_input("BMI", 10.0, 60.0, 22.5)
bp_sys = st.number_input("Systolic BP", 70, 250, 120)
bp_dia = st.number_input("Diastolic BP", 40, 150, 80)
spo2 = st.number_input("SpO2", 50, 100, 98)
glucose = st.number_input("Glucose (mg/dL)", 40, 500, 95)
hba1c = st.number_input("HbA1c", 3.0, 20.0, 5.2)

if st.button("Predict Disease"):
    patient = {
        "Age": age,
        "Fever": int(fever),
        "Cough": int(cough),
        "Fatigue": int(fatigue),
        "Breathlessness": int(breathlessness),
        "Chest_Pain": int(chest_pain),
        "Nausea": int(nausea),
        "Headache": int(headache),
        "BMI": bmi,
        "BP_Systolic": bp_sys,
        "BP_Diastolic": bp_dia,
        "SpO2": spo2,
        "Glucose_mg_dL": glucose,
        "HbA1c": hba1c,
        "Gender_M": int(gender == "Male")
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=patient
        )
        response.raise_for_status()
        result = response.json()

        st.success(f"Predicted disease: {result['predicted_disease']}")
        st.write(f"Model confidence: {result['confidence']}%")

        if "shap_explanation" in result:
            st.subheader("Why this prediction?")
            for item in result["shap_explanation"]:
                st.write(f"{item['feature']}: {item['impact']}")

    except requests.exceptions.RequestException:
        st.error("Could not connect to the prediction API. Start the FastAPI server first.")

st.caption("Educational project only. Not a medical diagnosis.")
