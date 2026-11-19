"""
PreTex - fault prediction model

Trains an XGBoost classifier that predicts whether a machine is faulty from four
sensor readings (temperature, pressure, vibration, humidity) and its equipment type.
The classes are imbalanced (few faulty machines), so the training set is balanced
with SMOTE, and a machine is flagged as faulty only when the predicted probability
is above 0.7, to limit false alarms.

Usage:
    python train_fault_classifier.py [path/to/equipment_anomaly_data.csv]
"""
import sys
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/equipment_anomaly_data.csv"
MODEL_PATH = "../models/machine_failure_predictor.pkl"
THRESHOLD = 0.7  # probability above which a machine is flagged as faulty

# ----- Load and prepare the data -----
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["location"])                  # not a physical signal
df = pd.get_dummies(df, columns=["equipment"])      # Compressor / Pump / Turbine

X = df.drop(columns=["faulty"])
y = df["faulty"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- Balance the training set (test set stays untouched) -----
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)

# ----- Train -----
model = XGBClassifier(
    eval_metric="logloss",
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_res, y_res)

# ----- Evaluate -----
proba = model.predict_proba(X_test_scaled)[:, 1]
preds = (proba > THRESHOLD).astype(int)
print("Confusion matrix:")
print(confusion_matrix(y_test, preds))
print("\nClassification report:")
print(classification_report(y_test, preds))

# ----- Save model + scaler together -----
joblib.dump({"model": model, "scaler": scaler, "columns": list(X.columns),
             "threshold": THRESHOLD}, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
