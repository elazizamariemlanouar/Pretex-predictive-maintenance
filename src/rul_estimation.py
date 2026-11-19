"""
PreTex - Remaining Useful Life (RUL) estimation (experimental)

The dataset has no real failure history, so RUL cannot be learned from data.
This script builds an ESTIMATED RUL with a rule-based degradation formula
(a base lifetime per equipment type minus penalties for high temperature,
vibration, pressure and humidity), then trains an XGBoost regressor on it,
alongside a fault classifier.

Because the regressor learns a formula we wrote ourselves, its error only shows
that the formula can be reproduced, not that real RUL can be predicted. With
real run-to-failure data (e.g. NASA C-MAPSS), the same pipeline could be
trained on true RUL values.

Usage:
    python rul_estimation.py [path/to/equipment_anomaly_data.csv]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/equipment_anomaly_data.csv"

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["location"])

# ----- Step 1: rule-based RUL estimate (hours) -----
BASE_RUL = {"Pump": 1000, "Compressor": 800, "Turbine": 1200}

def estimate_rul(row):
    penalty = (
        (row["temperature"] - 60) * 2
        + row["vibration"] * 50
        + (row["pressure"] - 30) * 1.5
        + (row["humidity"] - 50) * 0.5
    )
    return max(BASE_RUL.get(row["equipment"], 1000) - penalty, 0)

df["RUL"] = df.apply(estimate_rul, axis=1)

# ----- Step 2: preprocess -----
df = pd.get_dummies(df, columns=["equipment"])
X = df.drop(columns=["RUL", "faulty"])
y_rul, y_faulty = df["RUL"], df["faulty"]

X_train, X_test, y_rul_train, y_rul_test, y_faulty_train, y_faulty_test = train_test_split(
    X, y_rul, y_faulty, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- Step 3: RUL regressor + fault classifier -----
rul_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
rul_model.fit(X_train_scaled, y_rul_train)

fault_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
fault_model.fit(X_train_scaled, y_faulty_train)

# ----- Step 4: evaluate -----
rul_pred = rul_model.predict(X_test_scaled)
print(f"MAE  (estimated RUL): {mean_absolute_error(y_rul_test, rul_pred):.2f} hours")
print(f"RMSE (estimated RUL): {np.sqrt(mean_squared_error(y_rul_test, rul_pred)):.2f} hours")

fault_pred = fault_model.predict(X_test_scaled)
print("\nFault classification report:")
print(classification_report(y_faulty_test, fault_pred))
print("Confusion matrix:")
print(confusion_matrix(y_faulty_test, fault_pred))
