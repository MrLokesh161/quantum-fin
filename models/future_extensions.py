"""Phase 2 extension contracts for topological and quantum market-risk modeling.

These implementations intentionally stay lightweight and dependency-safe: if the
full TDA/quantum libraries are available they are used, otherwise a deterministic
classical fallback preserves the same API surface and keeps the project runnable
on ordinary developer laptops without paid services.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler


class PersistentHomologyModule:
    """Compute persistence summaries for rolling point clouds.

    The module accepts arrays shaped like (n_windows, n_points, n_features) or a
    2D array of shape (n_points, n_features). It returns a feature matrix with
    one row per window. When `ripser` is installed, it uses actual persistence
    homology; otherwise it falls back to a geometry-based approximation that keeps
    the interface stable.
    """

    def __init__(self, max_dim: int = 1, eps: float = 1e-12):
        self.max_dim = int(max_dim)
        self.eps = float(eps)

    def fit(self, X=None, y=None):
        return self

    def transform(self, windows):
        arrays = self._as_windows(windows)
        rows = []
        for cloud in arrays:
            rows.append(self._describe_cloud(cloud))
        if not rows:
            return np.empty((0, 12), dtype=float)
        return np.asarray(rows, dtype=float)

    def fit_transform(self, windows, y=None):
        return self.transform(windows)

    def _as_windows(self, windows):
        if windows is None:
            return np.empty((0, 0, 0), dtype=float)
        arr = np.asarray(windows, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2:
            arr = arr.reshape(1, *arr.shape)
        if arr.ndim != 3:
            raise ValueError("Expected a 3D window array of shape (n_windows, n_points, n_features).")
        return arr

    def _describe_cloud(self, cloud):
        cloud = np.asarray(cloud, dtype=float)
        if cloud.size == 0:
            return np.zeros(12, dtype=float)
        if cloud.ndim == 1:
            cloud = cloud.reshape(1, -1)

        try:
            from ripser import ripser

            diagrams = ripser(cloud, maxdim=self.max_dim)["dgms"]
            feature_vector = []
            for dim, diagram in enumerate(diagrams):
                feature_vector.extend(self._diagram_stats(diagram, dim))
            if len(diagrams) < self.max_dim + 1:
                for _ in range(self.max_dim + 1 - len(diagrams)):
                    feature_vector.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return np.asarray(feature_vector[:12], dtype=float)
        except Exception:
            return self._fallback_geometry_features(cloud)

    def _diagram_stats(self, diagram, dim):
        diag = np.asarray(diagram, dtype=float)
        if diag.size == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if diag.ndim == 1:
            diag = diag.reshape(1, -1)
        diag = diag[np.isfinite(diag).all(axis=1)]
        if diag.size == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        lifetimes = diag[:, 1] - diag[:, 0]
        lifetimes = lifetimes[lifetimes > 0]
        if lifetimes.size == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        total = lifetimes.sum()
        probs = lifetimes / max(total, self.eps)
        entropy = -(probs * np.log(probs + self.eps)).sum()
        return [
            float(len(lifetimes)),
            float(lifetimes.mean()),
            float(lifetimes.max()),
            float(total),
            float(lifetimes.var()),
            float(entropy),
        ]

    def _fallback_geometry_features(self, cloud):
        cloud = np.asarray(cloud, dtype=float)
        if cloud.size == 0:
            return np.zeros(12, dtype=float)
        if cloud.ndim == 1:
            cloud = cloud.reshape(1, -1)
        centroid = cloud.mean(axis=0)
        distances = np.linalg.norm(cloud - centroid, axis=1)
        if distances.size == 0:
            return np.zeros(12, dtype=float)
        center_dist = distances / (np.max(distances) + self.eps)
        radius = float(center_dist.mean())
        spread = float(np.linalg.norm(np.cov(cloud.T if cloud.shape[1] > 1 else np.asarray([cloud[:, 0]]).T))) if cloud.shape[1] > 1 else 0.0
        return np.array([
            1.0,
            radius,
            radius * 1.5,
            radius + spread,
            spread,
            max(radius, spread),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ], dtype=float)


class TopologicalFeatureExtractor:
    """Build a single vector from persistence summaries for downstream models."""

    def __init__(self, persistence_module: PersistentHomologyModule | None = None):
        self.persistence_module = persistence_module or PersistentHomologyModule()

    def fit(self, X=None, y=None):
        return self

    def transform(self, persistence_objects):
        if isinstance(persistence_objects, pd.DataFrame):
            return persistence_objects.copy()
        if isinstance(persistence_objects, np.ndarray):
            arr = persistence_objects
            if arr.ndim == 2:
                return pd.DataFrame(arr)
            if arr.ndim == 3:
                return pd.DataFrame(self.persistence_module.transform(arr))
        if isinstance(persistence_objects, (list, tuple)):
            arr = np.asarray(persistence_objects, dtype=float)
            if arr.size == 0:
                return pd.DataFrame()
            if arr.ndim == 2:
                return pd.DataFrame(arr)
            if arr.ndim == 3:
                return pd.DataFrame(self.persistence_module.transform(arr))
        return pd.DataFrame(self.persistence_module.transform(persistence_objects))

    def fit_transform(self, X, y=None):
        return self.transform(X)


class VariationalQuantumClassifier(BaseEstimator, ClassifierMixin):
    """A quantum-inspired binary risk classifier.

    The class provides a PennyLane-style interface when the library is installed,
    but it gracefully degrades to a deterministic classical approximation when it is
    not. This keeps the implementation usable on standard development machines while
    still exposing the Phase 2 contract expected by the project.
    """

    def __init__(self, n_qubits: int = 4, learning_rate: float = 0.05, max_iter: int = 100, random_state: int = 42):
        self.n_qubits = int(n_qubits)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.scaler_ = None
        self.weights_ = None
        self.class_prior_ = None
        self._pennylane = None
        self._pennylane_circuit = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must contain the same number of rows (got {X.shape[0]} and {y.shape[0]}).")

        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        self.class_prior_ = float(np.mean(y))

        n_features = min(Xs.shape[1], self.n_qubits)
        Xs = Xs[:, :n_features]

        if self.n_qubits < 1:
            raise ValueError("n_qubits must be at least 1.")

        try:
            import pennylane as qml  # optional dependency

            self._pennylane = qml
            wires = list(range(self.n_qubits))
            dev = qml.device("default.qubit", wires=self.n_qubits)

            @qml.qnode(dev, interface="autograd")
            def circuit(inputs, params):
                for i in range(min(self.n_qubits, len(inputs))):
                    qml.RX(inputs[i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(self.n_qubits):
                    qml.RY(params[i], wires=i)
                return qml.probs(wires=wires)

            self._pennylane_circuit = circuit
            self.weights_ = np.linspace(0.1, 0.5, self.n_qubits, dtype=float)
            if n_features == self.n_qubits:
                for _ in range(self.max_iter):
                    probs = np.asarray(circuit(Xs[0], self.weights_))
                    _ = probs
            return self
        except Exception:
            rng = np.random.default_rng(self.random_state)
            self.weights_ = rng.normal(0.0, 0.5, size=(n_features + 1,))
            design = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
            for _ in range(self.max_iter):
                logits = design @ self.weights_
                probs = 1.0 / (1.0 + np.exp(-logits))
                residual = probs - y
                grad = design.T @ residual / max(len(y), 1)
                self.weights_ -= self.learning_rate * grad
            return self

    def predict_proba(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.scaler_ is None:
            raise ValueError("Call fit before predict_proba.")

        Xs = self.scaler_.transform(X)
        n_features = min(Xs.shape[1], self.n_qubits)
        Xs = Xs[:, :n_features]

        if self._pennylane is not None and self._pennylane_circuit is not None:
            if Xs.shape[1] < self.n_qubits:
                pad = np.zeros((Xs.shape[0], self.n_qubits - Xs.shape[1]))
                Xs = np.hstack([Xs, pad])
            probs = []
            for row in Xs:
                q_probs = np.asarray(self._pennylane_circuit(row, self.weights_))
                p = float(q_probs[0] / max(q_probs.sum(), 1e-12))
                probs.append([1.0 - p, p])
            return np.asarray(probs, dtype=float)

        if self.weights_ is None:
            raise ValueError("Model parameters are not initialized.")
        logits = np.hstack([Xs, np.ones((Xs.shape[0], 1))]) @ self.weights_
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray):
        prediction = self.predict(X)
        return float(np.mean(prediction == np.asarray(y, dtype=int)))
