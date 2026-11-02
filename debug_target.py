import pandas as pd

# Load the cleaned dataset
df = pd.read_csv('data/cleaned_disease_symptom.csv')
target_col = 'diseases' # Updated to match your raw data

# Check total unique classes
unique_classes = df[target_col].nunique()
print(f"Unique diseases: {unique_classes}")

# View the class distribution
print(df[target_col].value_counts().head(10))