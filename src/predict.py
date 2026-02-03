from __future__ import annotations
import joblib
import pandas as pd
import numpy as np
from src.config import Paths

def predict_proba_from_row(row) :

    paths = Paths()
    artifact = joblib.load(paths.model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]

    df = pd.DataFrame([row])

    df["hour"] = (df["Time"] // 3600) % 24
    df["log_amount"] = np.log1p(df["Amount"])

    df = df.reindex(columns=feature_names, fill_value=0.0)

    to_scale = [c for c in ["Amount", "log_amount"] if c in df.columns]
    if to_scale and scaler is not None:
        df[to_scale] = scaler.transform(df[to_scale])

    proba = float(model.predict_proba(df)[:, 1][0])
    return proba

if __name__ == "__main__":
    example = {"Time": 100000, "Amount": 50.0}
    for i in range(1, 29):
        example[f"V{i}"] = 0.0

    p = predict_proba_from_row(example)
    print("Fraud risk (proba):", p)
