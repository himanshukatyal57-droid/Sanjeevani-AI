import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------------------------
# Update filename if your downloaded Kaggle CSV has a different name
CSV_FILENAME = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
DATA_PATH = os.path.join("data", CSV_FILENAME)

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Could not find dataset at: {DATA_PATH}")
    exit(1)

print(f"[*] Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# -------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------------
print("\n" + "="*50)
print(" 📊 EXPLORATORY DATA ANALYSIS")
print("="*50)

print(f"Total Records (Rows): {df.shape[0]:,}")
print(f"Total Features (Columns): {df.shape[1]}")

# Check missing values
missing_count = df.isnull().sum().sum()
print(f"Total Missing Values (NaNs): {missing_count}")

# Check duplicate rows
duplicate_count = df.duplicated().sum()
print(f"Duplicate Rows Found: {duplicate_count:,}")

# -------------------------------------------------------------------
# 3. COLUMN HEADER SANITIZATION (CRITICAL FOR NLP MATCHING)
# -------------------------------------------------------------------
print("\n[*] Sanitizing column headers...")

# Clean column headers: remove whitespace, convert to lowercase, replace spaces/hyphens with underscores
raw_columns = list(df.columns)
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(' ', '_')
              .str.replace('-', '_')
              .str.replace('/', '_')
)

# Detect Target Column (commonly 'disease', 'prognosis', or the last column)
possible_targets = ['diseases', 'disease', 'prognosis', 'target', 'condition']
target_col = None

for col in df.columns:
    if col in possible_targets:
        target_col = col
        break

if not target_col:
    # Default to the last column if no standard name matched
    target_col = df.columns[-1]

print(f"[✓] Target Column Identified: '{target_col}'")

# -------------------------------------------------------------------
# 4. VERIFY FEATURE DATA TYPES & BINARY MATRIX INTEGRITY
# -------------------------------------------------------------------
feature_cols = [c for c in df.columns if c != target_col]

# Verify all features are strictly binary (0 or 1)
non_binary_features = []
for col in feature_cols:
    unique_vals = set(df[col].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        non_binary_features.append((col, unique_vals))

if non_binary_features:
    print(f"[WARNING] {len(non_binary_features)} features contain non-binary values!")
    print("Sample non-binary feature:", non_binary_features[0])
else:
    print("[✓] All symptom features confirmed strictly binary (0 or 1).")

# Class Distribution Check
unique_diseases = df[target_col].nunique()
print(f"[✓] Unique Medical Conditions / Diseases: {unique_diseases}")

# -------------------------------------------------------------------
# 5. SAVE CLEANED DATASET
# -------------------------------------------------------------------
cleaned_csv_path = os.path.join("data", "cleaned_disease_symptom.csv")
df.to_csv(cleaned_csv_path, index=False)
print(f"[✓] Cleaned dataset saved to '{cleaned_csv_path}'.")
print("\n[READY] Days 1–2 milestone complete. Ready for Model Training (Day 3)!")