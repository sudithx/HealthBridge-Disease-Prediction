
from fastapi.testclient import TestClient
import app

client = TestClient(app.app)

sample = {
    "Age": 25,
    "Fever": 1,
    "Cough": 1,
    "Fatigue": 1,
    "Breathlessness": 0,
    "Chest_Pain": 0,
    "Nausea": 0,
    "Headache": 1,
    "BMI": 22.5,
    "BP_Systolic": 120,
    "BP_Diastolic": 80,
    "SpO2": 98,
    "Glucose_mg_dL": 95,
    "HbA1c": 5.2,
    "Gender_M": 1
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_prediction():
    response = client.post("/predict", json=sample)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_disease" in data
    assert "confidence" in data
    assert "shap_explanation" in data
    assert len(data["shap_explanation"]) == 5
