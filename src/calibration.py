"""Out-of-sample probability calibration and calibration metrics."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss


class TemperatureScaler:
    """Fit one temperature on a chronological calibration partition."""

    def __init__(self):
        self.temperature_ = 1.0
        self.status_ = "not fitted"

    def fit(self, probabilities, targets, classes):
        probabilities = np.asarray(probabilities, dtype=float)
        targets = np.asarray(targets)
        self.classes_ = np.asarray(classes)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.classes_):
            raise ValueError("Calibration probabilities do not match model classes.")
        if len(targets) != len(probabilities) or not np.isfinite(probabilities).all():
            raise ValueError("Calibration rows must have finite probabilities and matching labels.")
        class_indices = {label: index for index, label in enumerate(self.classes_)}
        if any(label not in class_indices for label in np.unique(targets)):
            raise ValueError("Calibration labels contain a class absent from the model.")
        if np.unique(targets).size < 2:
            self.status_ = "not calibrated: calibration partition has one class"
            return self

        logits = np.log(np.clip(probabilities, 1e-8, 1.0))
        target_indices = np.asarray([class_indices[label] for label in targets])

        def objective(log_temperature):
            scaled = logits / np.exp(log_temperature)
            scaled -= scaled.max(axis=1, keepdims=True)
            exp_scaled = np.exp(scaled)
            calibrated = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
            return log_loss(
                target_indices,
                calibrated,
                labels=np.arange(len(self.classes_)),
            )

        result = minimize_scalar(
            objective,
            bounds=(np.log(0.05), np.log(20.0)),
            method="bounded",
            options={"xatol": 1e-4},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Probability temperature fitting failed: {result.message}")
        self.temperature_ = float(np.exp(result.x))
        self.status_ = "temperature-scaled"
        return self

    def predict_proba(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.classes_):
            raise ValueError("Prediction probabilities do not match calibration classes.")
        logits = np.log(np.clip(probabilities, 1e-8, 1.0)) / self.temperature_
        logits -= logits.max(axis=1, keepdims=True)
        calibrated = np.exp(logits)
        return calibrated / calibrated.sum(axis=1, keepdims=True)


def expected_calibration_error(probabilities, targets, classes=None, bins=10):
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets)
    classes = np.arange(probabilities.shape[1]) if classes is None else np.asarray(classes)
    confidence = probabilities.max(axis=1)
    predicted = np.argmax(probabilities, axis=1)
    target_indices = np.asarray([
        np.flatnonzero(classes == target)[0] for target in targets
    ])
    accuracy = predicted == target_indices
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        selected = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if selected.any():
            error += selected.mean() * abs(
                accuracy[selected].mean() - confidence[selected].mean()
            )
    return float(error)


def calibration_metrics(probabilities, targets, classes=None):
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets)
    classes = np.arange(probabilities.shape[1]) if classes is None else np.asarray(classes)
    class_indices = {label: index for index, label in enumerate(classes)}
    target_indices = np.asarray([class_indices[label] for label in targets])
    one_hot = np.eye(len(classes))[target_indices]
    return {
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "log_loss": float(log_loss(
            target_indices,
            np.clip(probabilities, 1e-8, 1.0),
            labels=np.arange(len(classes)),
        )),
        "expected_calibration_error": expected_calibration_error(
            probabilities, targets, classes=classes
        ),
    }


def chronological_calibration_split(validation, purge_gap):
    """Split validation in time, purging between calibration and threshold tuning."""
    boundary = len(validation) // 2
    calibration = validation.iloc[:boundary].reset_index(drop=True)
    threshold_tuning = validation.iloc[boundary + purge_gap:].reset_index(drop=True)
    if calibration.empty or threshold_tuning.empty:
        raise ValueError(
            "Validation data is too small for separate calibration and threshold partitions."
        )
    return calibration, threshold_tuning


def fit_model_calibrators(models, scaler, calibration, features):
    x_calibration = scaler.transform(calibration[features])
    targets = calibration["target"].to_numpy()
    calibrators = {}
    for name, model in models.items():
        calibrators[name] = TemperatureScaler().fit(
            model.predict_proba(x_calibration),
            targets,
            model.classes_,
        )
    return calibrators
