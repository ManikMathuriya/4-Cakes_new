import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# 1. DIABETES MODEL TRAINING
# ==============================
print("Training Diabetes Model...")

diabetes_path = os.path.join(DATA_DIR, "diabetes.csv")
df_diabetes = pd.read_csv(diabetes_path)

X = df_diabetes.drop("Outcome", axis=1)
y = df_diabetes["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestClassifier
diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
diabetes_model.fit(X_train, y_train)

y_pred = diabetes_model.predict(X_test)
diabetes_accuracy = accuracy_score(y_test, y_pred)

joblib.dump(diabetes_model, os.path.join(MODEL_DIR, "diabetes_model.joblib"))

print(f"✅ Diabetes Model Accuracy: {diabetes_accuracy * 100:.2f}%")
print("✅ Diabetes model saved to /models/diabetes_model.joblib")

# ==============================
# 2. STUDENT STRESS MODEL TRAINING
# ==============================
print("\nTraining Student Stress Model...")

stress_path = os.path.join(DATA_DIR, "StressLevelDataset.csv")

if os.path.exists(stress_path):
    df_stress = pd.read_csv(stress_path)

    # Pick 4 best columns for stress prediction
    FEATURE_COLUMNS = [
        "anxiety_level",
        "depression",
        "sleep_quality",
        "peer_pressure"
    ]

    # Check if all columns exist
    missing_cols = [col for col in FEATURE_COLUMNS + ["stress_level"] if col not in df_stress.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing Columns in CSV: {missing_cols}")

    Xs = df_stress[FEATURE_COLUMNS]
    ys = df_stress["stress_level"]

else:
    print("⚠ StressLevelDataset.csv NOT found")
    print("⚠ Generating synthetic dataset...")

    np.random.seed(42)

    anxiety_level = np.random.randint(1, 21, 150)
    depression = np.random.randint(1, 21, 150)
    sleep_quality = np.random.randint(1, 6, 150)
    peer_pressure = np.random.randint(1, 6, 150)

    stress_level = [
        1 if a > 12 or d > 14 or sq < 3 or pp > 4 else 0
        for a, d, sq, pp in zip(anxiety_level, depression, sleep_quality, peer_pressure)
    ]

    df_stress = pd.DataFrame({
        "anxiety_level": anxiety_level,
        "depression": depression,
        "sleep_quality": sleep_quality,
        "peer_pressure": peer_pressure,
        "stress_level": stress_level
    })

    Xs = df_stress[["anxiety_level", "depression", "sleep_quality", "peer_pressure"]]
    ys = df_stress["stress_level"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    Xs, ys, test_size=0.2, random_state=42
)

stress_model = LogisticRegression(max_iter=200)
stress_model.fit(X_train_s, y_train_s)

y_pred_s = stress_model.predict(X_test_s)
stress_accuracy = accuracy_score(y_test_s, y_pred_s)

joblib.dump(stress_model, os.path.join(MODEL_DIR, "stress_model.joblib"))

print(f"✅ Stress Model Accuracy: {stress_accuracy * 100:.2f}%")
print("✅ Stress model saved to /models/stress_model.joblib")

print("\n🎉 ALL MODELS TRAINED & SAVED SUCCESSFULLY!")
