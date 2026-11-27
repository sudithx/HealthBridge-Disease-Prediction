from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import numpy as np

model = joblib.load("models/healthbridge_best_model.pkl")
explainer = shap.TreeExplainer(model)

app = FastAPI(title="HealthBridge Disease Prediction API")

class PatientData(BaseModel):
    Age: float
    Fever: int
    Cough: int
    Fatigue: int
    Breathlessness: int
    Chest_Pain: int
    Nausea: int
    Headache: int
    BMI: float
    BP_Systolic: float
    BP_Diastolic: float
    SpO2: float
    Glucose_mg_dL: float
    HbA1c: float
    Gender_M: int

@app.get("/health")
def health():
    return {"status": "HealthBridge API is running"}

@app.post("/predict")
def predict(data: PatientData):
    input_df = pd.DataFrame([data.model_dump()])
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    shap_values = explainer.shap_values(input_df)
    class_index = list(model.classes_).index(prediction)

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[class_index])[0]
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[0, :, class_index]
        else:
            values = values[0]

    importance = sorted(
        zip(input_df.columns, values),
        key=lambda x: abs(float(x[1])),
        reverse=True
    )[:5]

    return {
        "predicted_disease": str(prediction),
        "confidence": round(float(max(probabilities)) * 100, 2),
        "shap_explanation": [
            {"feature": name, "impact": round(float(value), 4)}
            for name, value in importance
        ]
    }
