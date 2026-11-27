# HealthBridge – Checkpoint 2 Model Evaluation Report

## 1. Dataset Information
- Total records: 800
- Input features: 15
- Target variable: Diagnosis
- Number of diagnosis categories: 8
- Records per category: 100

## 2. Data Preparation
- Separated input features and target variable.
- Excluded Patient_ID from model training.
- Converted Gender into numerical format using one-hot encoding.
- Split the dataset into 80% training and 20% testing data.
- Training records: 640
- Testing records: 160

## 3. Baseline Model
- Algorithm: Random Forest Classifier
- Number of estimators: 100
- Random state: 42

## 4. Model Performance
- Accuracy: 81.25%
- Precision (macro average): 0.82
- Recall (macro average): 0.81
- F1-score (macro average): 0.81

## 5. Evaluation
The baseline model achieved 81.25% accuracy on the test dataset. A confusion matrix was generated to visualize correct and incorrect predictions across the eight diagnosis categories.

## 6. Project Status
The initial baseline model has been trained, evaluated, and saved as `healthbridge_model.pkl`. Further improvements and expansion to 25 or more disease categories are still required.

**Note:** This is an educational project using a project dataset. It is not a clinically validated diagnostic tool.
