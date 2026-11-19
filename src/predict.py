"""
PreTex - predict the state of one machine from its sensor readings.

Usage:
    python predict.py --temperature 50 --pressure 34 --vibration 1.5 --humidity 20 --equipment Compressor
"""
import argparse
import joblib
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, required=True)
parser.add_argument("--pressure", type=float, required=True)
parser.add_argument("--vibration", type=float, required=True)
parser.add_argument("--humidity", type=float, required=True)
parser.add_argument("--equipment", choices=["Compressor", "Pump", "Turbine"], required=True)
parser.add_argument("--model", default="../models/machine_failure_predictor.pkl")
args = parser.parse_args()

bundle = joblib.load(args.model)
model, scaler = bundle["model"], bundle["scaler"]
threshold = bundle.get("threshold", 0.7)

sample = pd.DataFrame([{
    "temperature": args.temperature,
    "pressure": args.pressure,
    "vibration": args.vibration,
    "humidity": args.humidity,
    "equipment_Compressor": int(args.equipment == "Compressor"),
    "equipment_Pump": int(args.equipment == "Pump"),
    "equipment_Turbine": int(args.equipment == "Turbine"),
}])

proba = model.predict_proba(scaler.transform(sample))[:, 1][0]
if proba > threshold:
    print(f"FAULTY  - probability of fault: {proba:.2%}  -> schedule maintenance")
else:
    print(f"HEALTHY - probability of fault: {proba:.2%}")
