# HealthBridge – Checkpoint 3: Core Model and Experimentation

## 1. Objective
The objective was to train and compare multiple Random Forest model configurations and identify the best-performing configuration using evaluation metrics.

## 2. Dataset
- Dataset: HealthBridge healthcare symptoms dataset
- Total records: 800
- Input features: 15
- Diagnosis categories: 8
- Training records: 640
- Testing records: 160
- Train-test split: 80:20

## 3. Model and Experiment Setup
Algorithm: Random Forest Classifier

Two hyperparameter configurations were tested:

| Parameter | Configuration 1 | Configuration 2 |
|---|---:|---:|
| Number of trees | 100 | 200 |
| Maximum depth | 10 | 20 |
| Random state | 42 | 42 |

Both configurations used the same training and testing split.

## 4. Experiment Results

| Metric | Configuration 1 | Configuration 2 |
|---|---:|---:|
| Accuracy | 81.25% | 81.88% |
| Precision | 0.8180 | 0.8212 |
| Recall | 0.8125 | 0.8187 |
| F1-score | 0.8111 | 0.8160 |

## 5. Best Configuration
Configuration 2 was selected as the best-performing configuration.

- Algorithm: Random Forest Classifier
- Number of trees: 200
- Maximum depth: 20
- Accuracy: 81.88%
- Precision: 0.8212
- Recall: 0.8187
- F1-score: 0.8160

The model achieved slightly better results than Configuration 1 on the same test dataset.

## 6. Experiment Tracking
MLflow was used to track both model experiments.

The following information was logged:
- Hyperparameters
- Accuracy
- Precision
- Recall
- F1-score

The experiment evidence was exported as a CSV file.

## 7. Best Model
The selected model was saved as:

`healthbridge_best_model.pkl`

## 8. Conclusion and Future Work
Two Random Forest configurations were trained and evaluated. Configuration 2 achieved the highest test accuracy of 81.88%.

Future work includes expanding the dataset to at least 25 disease categories, further model validation, SHAP explainability, and integration with the FastAPI backend and Streamlit interface.

**Disclaimer:** This is an educational project and the model is not clinically validated for medical diagnosis.
