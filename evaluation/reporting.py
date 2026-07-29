"""Evaluation metrics, model diagnostics, and graphics."""
from pathlib import Path
import json
import matplotlib
# The pipeline is intended for batch/research-server execution as well as IDEs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def evaluate_model(model, X_test, y_test) -> tuple[dict, np.ndarray, np.ndarray]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {"accuracy": accuracy_score(y_test, predictions), "precision": precision_score(y_test, predictions, zero_division=0), "recall": recall_score(y_test, predictions, zero_division=0), "f1": f1_score(y_test, predictions, zero_division=0), "roc_auc": roc_auc_score(y_test, probabilities) if len(np.unique(y_test)) > 1 else None}
    return metrics, predictions, probabilities


def save_metrics(metrics: dict, path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def generate_eda_plots(data: pd.DataFrame, close_column: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = data.select_dtypes(include="number")
    plt.figure(figsize=(14, 10)); sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, linewidths=.1); plt.tight_layout(); plt.savefig(output_dir / "correlation_heatmap.png", dpi=180); plt.close()
    columns = [close_column, "daily_return", "rolling_volatility_20", "rsi_14"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for axis, col in zip(axes.ravel(), columns): sns.histplot(data[col], kde=True, ax=axis); axis.set_title(f"Distribution: {col}")
    fig.tight_layout(); fig.savefig(output_dir / "distributions.png", dpi=180); plt.close(fig)
    plt.figure(figsize=(14, 5)); plt.plot(data["Date"], data[close_column], label=close_column); plt.title("Primary Market Closing Price"); plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / "time_series.png", dpi=180); plt.close()


def generate_evaluation_plots(model, y_test, probabilities, feature_names: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted = (probabilities >= .5).astype(int)
    plt.figure(figsize=(6, 5)); sns.heatmap(confusion_matrix(y_test, predicted), annot=True, fmt="d", cmap="Blues", xticklabels=["Stable", "High risk"], yticklabels=["Stable", "High risk"]); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(output_dir / "confusion_matrix.png", dpi=180); plt.close()
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        plt.figure(figsize=(6, 5)); plt.plot(fpr, tpr, label="Random Forest"); plt.plot([0,1], [0,1], "--", color="grey"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / "roc_curve.png", dpi=180); plt.close()
    importance = pd.Series(model.feature_importances_, index=feature_names).nlargest(25).sort_values()
    plt.figure(figsize=(10, 8)); importance.plot.barh(); plt.title("Top 25 Sliding-Window Feature Importances"); plt.tight_layout(); plt.savefig(output_dir / "feature_importance.png", dpi=180); plt.close()
