"""Behavior policy estimation."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class BehaviorPolicy:
    """Logistic regression behavior policy."""

    def __init__(self, n_actions: int, softening: float = 0.01, random_seed: int = 42):
        self.n_actions = n_actions
        self.softening = softening
        self.random_seed = random_seed

        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=random_seed,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, states: np.ndarray, actions: np.ndarray):
        """Fit behavior policy."""
        logger.info("Fitting behavior policy (logistic regression)...")
        states_scaled = self.scaler.fit_transform(states)
        self.model.fit(states_scaled, actions)
        self.classes_ = self.model.classes_
        self.fitted = True
        logger.info(f"  Behavior policy fitted with {len(self.classes_)} action classes")

    def get_action_probs(self, states: np.ndarray, actions: np.ndarray = None) -> np.ndarray:
        """Get action probabilities."""
        if not self.fitted:
            raise RuntimeError("Behavior policy not fitted")

        states_scaled = self.scaler.transform(states)
        probs_raw = self.model.predict_proba(states_scaled)

        n_samples = len(states)
        probs = np.full((n_samples, self.n_actions), self.softening / self.n_actions)

        for i, cls in enumerate(self.classes_):
            probs[:, int(cls)] = probs_raw[:, i]

        uniform = 1.0 / self.n_actions
        probs = (1 - self.softening) * probs + self.softening * uniform
        probs = probs / probs.sum(axis=1, keepdims=True)

        if actions is not None:
            return probs[np.arange(len(actions)), actions.astype(int)]

        return probs
