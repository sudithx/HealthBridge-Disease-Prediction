
import pandas as pd
from pathlib import Path

# Dataset location
data_path = Path("data/aiml_healthcare_symptoms.csv")

# Load dataset
df = pd.read_csv(data_path)

# Display dataset information
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Display missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Save cleaned dataset
Path("processed_data").mkdir(exist_ok=True)
df.to_csv("processed_data/cleaned_data.csv", index=False)

print("\nData preprocessing completed!")
print("Cleaned dataset shape:", df.shape)
