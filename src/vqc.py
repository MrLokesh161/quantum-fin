"""Small statevector variational quantum classifier for binary research tasks."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class VQCClassifier:
    """Binary VQC with angle encoding, a trainable RY/RZ ansatz, and SPSA."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        iterations: int = 30,
        batch_size: int = 64,
        max_samples: int = 512,
        learning_rate: float = 0.15,
        perturbation: float = 0.1,
        random_state: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.iterations = iterations
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("VQC input must be a finite two-dimensional array.")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("VQC training data must contain both binary classes.")
        if self.n_qubits < 2 or self.n_layers < 1:
            raise ValueError("VQC requires at least two qubits and one circuit layer.")
        if self.iterations < 1 or self.batch_size < 2 or self.max_samples < 2:
            raise ValueError("VQC iterations and sample/batch sizes must be positive.")

        self.classes_ = np.asarray([0, 1])
        self.rng_ = np.random.default_rng(self.random_state)
        self.n_qubits_ = min(self.n_qubits, X.shape[1], len(X))
        if self.n_qubits_ < 2:
            raise ValueError("VQC needs at least two input features and training rows.")
        self.pca_ = PCA(n_components=self.n_qubits_, random_state=self.random_state)
        encoded = self.pca_.fit_transform(X)
        component_scale = np.sqrt(self.pca_.explained_variance_)
        component_scale = np.where(component_scale > 1e-8, component_scale, 1.0)
        self.component_scale_ = component_scale
        encoded = self._encode_features(encoded)
        encoded, y = self._balanced_training_subset(encoded, y)

        self.parameters_ = self.rng_.normal(
            0.0, 0.15, size=(self.n_layers, self.n_qubits_, 2)
        )
        class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
        batch_per_class = max(1, self.batch_size // 2)
        for iteration in range(self.iterations):
            batch_indices = np.concatenate([
                self.rng_.choice(indices, batch_per_class, replace=len(indices) < batch_per_class)
                for indices in class_indices
            ])
            self.rng_.shuffle(batch_indices)
            batch_X = encoded[batch_indices]
            batch_y = y[batch_indices]

            direction = self.rng_.choice(
                np.asarray([-1.0, 1.0]), size=self.parameters_.shape
            )
            plus_loss = self._loss(
                self._probability(batch_X, self.parameters_ + self.perturbation * direction),
                batch_y,
            )
            minus_loss = self._loss(
                self._probability(batch_X, self.parameters_ - self.perturbation * direction),
                batch_y,
            )
            gradient = (
                (plus_loss - minus_loss)
                / (2.0 * self.perturbation)
                * direction
            )
            step_size = self.learning_rate / ((iteration + 1) ** 0.101)
            self.parameters_ -= step_size * gradient

        self.training_loss_ = self._loss(
            self._probability(encoded, self.parameters_), y
        )
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        if not hasattr(self, "parameters_"):
            raise ValueError("VQC must be fitted before requesting predictions.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"VQC expected {self.n_features_in_} input features."
            )
        if not np.isfinite(X).all():
            raise ValueError("VQC prediction input must contain only finite values.")
        encoded = self._encode_features(self.pca_.transform(X))
        positive = self._probability(encoded, self.parameters_)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _encode_features(self, X):
        return np.pi * np.tanh(X / self.component_scale_)

    def _balanced_training_subset(self, X, y):
        if len(y) <= self.max_samples:
            return X, y
        class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
        quota = self.max_samples // 2
        chosen = np.concatenate([
            self.rng_.choice(indices, quota, replace=len(indices) < quota)
            for indices in class_indices
        ])
        self.rng_.shuffle(chosen)
        return X[chosen], y[chosen]

    @staticmethod
    def _loss(probabilities, y):
        probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        positive_weight = len(y) / (2.0 * np.count_nonzero(y == 1))
        negative_weight = len(y) / (2.0 * np.count_nonzero(y == 0))
        weights = np.where(y == 1, positive_weight, negative_weight)
        losses = -(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
        return float(np.average(losses, weights=weights))

    def _probability(self, encoded, parameters):
        n_states = 1 << self.n_qubits_
        state = np.zeros((len(encoded), n_states), dtype=np.complex128)
        state[:, 0] = 1.0
        for qubit in range(self.n_qubits_):
            self._apply_ry(state, qubit, encoded[:, qubit])
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits_):
                self._apply_ry(state, qubit, parameters[layer, qubit, 0])
                self._apply_rz(state, qubit, parameters[layer, qubit, 1])
            for qubit in range(self.n_qubits_ - 1):
                self._apply_cnot(state, qubit, qubit + 1)

        probabilities = np.abs(state) ** 2
        measured_bit = 1 << 0
        bit_zero = np.arange(n_states)[(np.arange(n_states) & measured_bit) == 0]
        bit_one = bit_zero | measured_bit
        expectation = probabilities[:, bit_zero].sum(axis=1) - probabilities[:, bit_one].sum(axis=1)
        return np.clip((1.0 - expectation) / 2.0, 0.0, 1.0)

    @staticmethod
    def _apply_ry(state, qubit, angle):
        mask = 1 << qubit
        zero = np.arange(state.shape[1])[(np.arange(state.shape[1]) & mask) == 0]
        one = zero | mask
        cosine = np.cos(np.asarray(angle) / 2.0)
        sine = np.sin(np.asarray(angle) / 2.0)
        if cosine.ndim == 0:
            cosine = np.full(state.shape[0], cosine)
            sine = np.full(state.shape[0], sine)
        old_zero = state[:, zero].copy()
        old_one = state[:, one].copy()
        state[:, zero] = cosine[:, None] * old_zero - sine[:, None] * old_one
        state[:, one] = sine[:, None] * old_zero + cosine[:, None] * old_one

    @staticmethod
    def _apply_rz(state, qubit, angle):
        mask = 1 << qubit
        indices = np.arange(state.shape[1])
        zero = indices[(indices & mask) == 0]
        one = zero | mask
        state[:, zero] *= np.exp(-0.5j * angle)
        state[:, one] *= np.exp(0.5j * angle)

    @staticmethod
    def _apply_cnot(state, control, target):
        indices = np.arange(state.shape[1])
        control_mask = 1 << control
        target_mask = 1 << target
        permutation = np.where(
            indices & control_mask,
            indices ^ target_mask,
            indices,
        )
        state[:] = state[:, permutation]
