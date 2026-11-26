
import pandas as pd
from pathlib import Path

# Dataset path
data_path = Path("data/aiml_healthcare_symptoms.csv")

# Check whether dataset exists
if not data_path.exists():
    print("Dataset not found. Please upload the CSV into the data folder.")
else:
    # Load dataset
    df = pd.read_csv(data_path)

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    # Display column information
    print("\nDataset columns:")
    print(df.columns.tolist())

    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Create output folder
    Path("processed_data").mkdir(exist_ok=True)

    # Save cleaned dataset
    df.to_csv("processed_data/cleaned_data.csv", index=False)

    print("\nData preprocessing completed!")
    print("Cleaned dataset shape:", df.shape)
