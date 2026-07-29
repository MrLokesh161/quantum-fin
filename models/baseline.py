"""Leakage-safe Random Forest baseline training."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def fit_baseline(X_windows: np.ndarray, y: np.ndarray, test_size: float, scaler_name: str, random_state: int):
    """Chronologically split, scale using train data only, then tune a RF."""
    X = X_windows.reshape(len(X_windows), -1)
    split = int(len(X) * (1 - test_size))
    if split < 30 or len(X) - split < 10:
        raise ValueError("Insufficient windows for a chronological train/test split.")
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    scaler = StandardScaler() if scaler_name == "standard" else MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    cv_splits = min(5, max(2, len(X_train) // 100))
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [None, 8], "min_samples_leaf": [1, 3], "max_features": ["sqrt"]},
        scoring="f1", cv=TimeSeriesSplit(n_splits=cv_splits), n_jobs=-1, verbose=1,
    )
    grid.fit(X_train_scaled, y_train)
    return grid, scaler, X_train_scaled, X_test_scaled, y_train, y_test, split
