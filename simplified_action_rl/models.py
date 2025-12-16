"""
Neural network models for Q-learning.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """Q-function neural network."""

    def __init__(self, n_states: int, n_actions: int, hidden_dims: list, dropout: float):
        super().__init__()

        layers = []
        prev_dim = n_states

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_actions))
        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)

    def get_q_values(self, states: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        q_all = self.forward(states)
        if actions is None:
            return q_all
        return q_all.gather(1, actions.unsqueeze(1)).squeeze(1)


class BehaviorPolicy:
    """State-dependent behavior policy using logistic regression."""

    def __init__(self, n_actions: int, softening: float, random_seed: int):
        self.n_actions = n_actions
        self.softening = softening
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
