"""Expanding-window out-of-sample evaluation for the main binary risk task."""
from __future__ import annotations

from importlib.util import find_spec
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.baseline_models import train_models
from src.calibration import TemperatureScaler, calibration_metrics


def walk_forward_indices(row_count, folds, purge_gap):
    if folds < 2:
        raise ValueError("Walk-forward evaluation requires at least two folds.")
    if row_count <= folds * (purge_gap + 1):
        raise ValueError("Too few rows for the configured walk-forward folds and purge gap.")
    splitter = TimeSeriesSplit(n_splits=folds, gap=purge_gap)
    return list(splitter.split(np.arange(row_count)))


def walk_forward_evaluate(data, features, config, purge_gap, folds):
    model_names = ["Random Forest"]
    if config["models"].get("enable_xgboost", False) and find_spec("xgboost"):
        model_names.append("XGBoost")
    if config["models"].get("enable_vqc", False):
        model_names.append("VQC (Statevector Simulator)")
    fold_predictions = []
    fold_metrics = []
    for fold_number, (train_indices, test_indices) in enumerate(
        walk_forward_indices(len(data), folds, purge_gap), start=1
    ):
        calibration_size = max(purge_gap + 2, int(len(train_indices) * 0.15))
        calibration_start = len(train_indices) - calibration_size
        fit_end = calibration_start - purge_gap
        if fit_end <= 0:
            raise ValueError(f"Walk-forward fold {fold_number} has no training rows after purging.")

        model_train = data.iloc[train_indices[:fit_end]].reset_index(drop=True)
        calibration = data.iloc[train_indices[calibration_start:]].reset_index(drop=True)
        test = data.iloc[test_indices].reset_index(drop=True)
        models, scaler, x_test, y_test = train_models(
            model_train,
            test,
            features,
            config,
            model_names=model_names,
        )
        x_calibration = scaler.transform(calibration[features])
        calibrators = {}
        for name, model in models.items():
            calibrators[name] = TemperatureScaler().fit(
                model.predict_proba(x_calibration),
                calibration["target"].to_numpy(),
                model.classes_,
            )

        prediction = pd.DataFrame({
            "Date": test["Date"].to_numpy(),
            "fold": fold_number,
            "actual": y_test,
        })
        for name, model in models.items():
            probabilities = calibrators[name].predict_proba(
                model.predict_proba(x_test)
            )
            positive = probabilities[:, 1]
            prediction[_probability_column(name)] = positive
            predicted = (positive >= 0.5).astype(int)
            calibration_summary = calibration_metrics(
                probabilities, y_test, classes=model.classes_
            )
            fold_metrics.append({
                "fold": fold_number,
                "model": name,
                "feature_set": "Financial + TDA",
                "train_start": str(model_train["Date"].min().date()),
                "train_end": str(model_train["Date"].max().date()),
                "calibration_start": str(calibration["Date"].min().date()),
                "calibration_end": str(calibration["Date"].max().date()),
                "test_start": str(test["Date"].min().date()),
                "test_end": str(test["Date"].max().date()),
                "calibration_method": calibrators[name].status_,
                "accuracy_at_0_5": float(accuracy_score(y_test, predicted)),
                "f1_at_0_5": float(f1_score(y_test, predicted, zero_division=0)),
                "pr_auc": float(average_precision_score(y_test, positive))
                if np.any(y_test == 1) else np.nan,
                "roc_auc": float(roc_auc_score(y_test, positive))
                if np.unique(y_test).size > 1 else np.nan,
                "brier_score": float(brier_score_loss(y_test, positive)),
                "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
                "expected_calibration_error": calibration_summary[
                    "expected_calibration_error"
                ],
            })
        fold_predictions.append(prediction)

    return pd.concat(fold_predictions, ignore_index=True), pd.DataFrame(fold_metrics)


def _probability_column(model_name):
    if model_name == "Random Forest":
        return "crash_probability"
    if model_name == "VQC (Statevector Simulator)":
        return "vqc_probability"
    return "xgboost_probability"
