from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve,
    classification_report
)
import joblib

from src.config import Paths, TrainConfig
from src.data_utils import load_data, add_features, split_xy, train_test_split_stratified, scale_amount_features

def plot_and_save_roc(y_true, y_proba, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_and_save_pr(y_true, y_proba, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    
#Pour savoir quelles variables ont le plus d’influence sur la décision du modèle
def plot_and_save_feature_importance(feature_names, coefs, out_path, top_k=15):
    importances = np.abs(coefs)
    idx = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    plt.figure()
    plt.barh(range(len(vals))[::-1], vals)
    plt.yticks(range(len(vals))[::-1], names)
    plt.xlabel("|coefficient|")
    plt.title(f"Top {top_k} Feature Importance (LogReg)")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    paths = Paths()
    cfg = TrainConfig()
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(paths.model_path)
    model = artifact["model"]
    threshold = artifact.get("threshold", cfg.threshold)

    df = load_data(str(paths.data_csv))
    df = add_features(df)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, cfg.test_size, cfg.random_state)
    X_train, X_test, _ = scale_amount_features(X_train, X_test)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure()
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix (threshold={threshold})")
    plt.savefig(paths.outputs_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    plot_and_save_roc(y_test, proba, paths.outputs_dir / "roc_curve.png")
    plot_and_save_pr(y_test, proba, paths.outputs_dir / "pr_curve.png")

    feature_names = X_train.columns.tolist()
    plot_and_save_feature_importance(
        feature_names,
        model.coef_.ravel(),
        paths.outputs_dir / "feature_importance.png",
        top_k=15
    )

    report = classification_report(y_test, y_pred, output_dict=True)
    with open(paths.outputs_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Évaluation terminée. Fichiers générés dans outputs/")
    print("confusion_matrix.png, roc_curve.png, pr_curve.png, feature_importance.png")
    print("classification_report.json")

if __name__ == "__main__":
    main()
