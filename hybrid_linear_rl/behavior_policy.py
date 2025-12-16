"""
Behavior policy estimation using logistic regression.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class BehaviorPolicy:
    """State-dependent behavior policy."""

    def __init__(self, n_actions: int, softening: float, random_seed: int):
        self.n_actions = n_actions
        self.softening = softening
        self.model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=random_seed,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, states: np.ndarray, actions: np.ndarray):
        logger.info("Fitting behavior policy (logistic regression)...")
        states_scaled = self.scaler.fit_transform(states)
        self.model.fit(states_scaled, actions)
        self.classes_ = self.model.classes_
        self.fitted = True
        logger.info(f"  Behavior policy fitted with {len(self.classes_)} action classes")

    def predict_probs(self, states: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("BehaviorPolicy not fitted")

        states_scaled = self.scaler.transform(states)
        probs_raw = self.model.predict_proba(states_scaled)

        n_samples = len(states)
        probs = np.full((n_samples, self.n_actions), self.softening / self.n_actions)

        for i, cls in enumerate(self.classes_):
            probs[:, int(cls)] = probs_raw[:, i]

        uniform = 1.0 / self.n_actions
        probs = (1 - self.softening) * probs + self.softening * uniform
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def get_action_probs(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        probs = self.predict_probs(states)
        return probs[np.arange(len(actions)), actions.astype(int)]
