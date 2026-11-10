"""One-session downside risk targets and a calibrated five-class model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from src.calibration import TemperatureScaler, calibration_metrics

RISK_LABELS = ("Stable", "Low", "Moderate", "High", "Critical")
RISK_CUTS = (-0.005, -0.01, -0.015, -0.025)


def next_session_targets(market_features):
    """Label next-session low versus today's close using fixed downside bands."""
    required = {"Date", "Low", "Close"}
    missing = required.difference(market_features.columns)
    if missing:
        raise ValueError(f"Market features are missing target columns: {sorted(missing)}")
    if market_features["Date"].duplicated().any():
        raise ValueError("Market feature dates must be unique before labeling next sessions.")
    if not market_features["Date"].is_monotonic_increasing:
        raise ValueError("Market feature dates must be sorted before labeling next sessions.")
    next_low = market_features["Low"].shift(-1)
    drawdown = next_low / market_features["Close"] - 1.0
    valid = (
        drawdown.notna()
        & np.isfinite(drawdown)
        & np.isfinite(market_features["Close"])
        & market_features["Close"].gt(0)
    )
    targets = np.select(
        [
            drawdown <= RISK_CUTS[3],
            drawdown <= RISK_CUTS[2],
            drawdown <= RISK_CUTS[1],
            drawdown <= RISK_CUTS[0],
        ],
        [4, 3, 2, 1],
        default=0,
    )
    return pd.DataFrame({
        "Date": market_features.loc[valid, "Date"].to_numpy(),
        "next_session_max_drawdown": drawdown.loc[valid].to_numpy(),
        "next_session_risk_target": targets[valid.to_numpy()].astype(int),
    })


def fit_next_session_model(train, calibration, test, features, random_seed, rf_config):
    targets = train["next_session_risk_target"].to_numpy(dtype=int)
    if np.unique(targets).size < 2:
        raise ValueError("Next-session training partition must contain multiple risk classes.")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features])
    model = RandomForestClassifier(
        **rf_config,
        class_weight="balanced_subsample",
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(x_train, targets)
    x_calibration = scaler.transform(calibration[features])
    calibrator = TemperatureScaler().fit(
        model.predict_proba(x_calibration),
        calibration["next_session_risk_target"].to_numpy(dtype=int),
        model.classes_,
    )
    x_test = scaler.transform(test[features])
    raw_probabilities = model.predict_proba(x_test)
    calibrated = calibrator.predict_proba(raw_probabilities)
    return model, scaler, calibrator, calibrated


def next_session_metrics(targets, probabilities, classes):
    targets = np.asarray(targets, dtype=int)
    classes = np.asarray(classes, dtype=int)
    predicted_indices = np.argmax(probabilities, axis=1)
    predicted = classes[predicted_indices]
    target_indices = np.asarray([
        np.flatnonzero(classes == target)[0] for target in targets
    ])
    return {
        "accuracy": float(accuracy_score(targets, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
        "log_loss": float(log_loss(
            target_indices,
            probabilities,
            labels=np.arange(len(classes)),
        )),
        **calibration_metrics(probabilities, targets, classes),
    }
