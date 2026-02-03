from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import Paths, TrainConfig
from src.data_utils import load_data, add_features, split_xy, train_test_split_stratified, scale_amount_features

def main():
    paths = Paths()
    cfg = TrainConfig()
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    (paths.root / "models").mkdir(parents=True, exist_ok=True)

    df = load_data(paths.data_csv)
    df = add_features(df)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, cfg.test_size, cfg.random_state)
    X_train, X_test, scaler = scale_amount_features(X_train, X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=None
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": X_train.columns.tolist(),
        "threshold": cfg.threshold,
    }
    joblib.dump(artifact, paths.model_path)

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": cfg.threshold,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "fraud_rate": y.mean(),
    }
    with open(paths.outputs_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Modèle entraîné et sauvegardé :", paths.model_path)
    print("ROC-AUC:", roc_auc, "| PR-AUC:", pr_auc)
    print("Metrics:", paths.outputs_dir / "metrics.json")

if __name__ == "__main__":
    main()
