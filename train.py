import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Ensure output directory exists
os.makedirs("models", exist_ok=True)

# 2. Load dataset from local 'data/' folder
print("Loading dataset from local data/ folder...")
csv_path = os.path.join("data", "disease_symptom.csv")

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found!")
    print("Please place your downloaded Kaggle CSV file inside the data/ folder.")
    exit(1)

df = pd.read_csv(csv_path)
print(f"Dataset Loaded Successfully! Shape: {df.shape}")

# 3. Separate features (X) and target (y)
target_col = "Disease"  # Update column name if different in your dataset (e.g., 'prognosis')
X = df.drop(columns=[target_col])
y = df[target_col]

# 4. Save feature schema (symptom names) to JSON for NLP matching
symptom_schema = list(X.columns)
with open("symptom_schema.json", "w") as f:
    json.dump(symptom_schema, f, indent=2)

print(f"Saved {len(symptom_schema)} symptoms to symptom_schema.json")

# 5. Split dataset into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Decision Tree Classifier (Unit 5 Syllabus: Entropy criterion)
print("Training Decision Tree Classifier...")
clf = DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=42)
clf.fit(X_train, y_train)

# 7. Evaluate Model Performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n--- Model Evaluation ---")
print(f"Test Accuracy: {accuracy:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save the trained model to disk (.pkl file)
model_path = os.path.join("models", "decision_tree_model.pkl")
joblib.dump(clf, model_path)
print(f"\nModel successfully saved to {model_path}")
