"""GBM agent with state noise for diversity."""

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import List
import logging

logger = logging.getLogger(__name__)


class GBMQAgent:
    """GBM Q-Learning agent with state noise."""

    def __init__(self, n_states: int, n_actions: int, config):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config
        self.models = [None] * n_actions
        self.scaler = StandardScaler()
        self.scaler_fitted = False

    def fit_scaler(self, states: np.ndarray):
        """Fit state scaler."""
        self.scaler.fit(states)
        self.scaler_fitted = True

    def scale_states(self, states: np.ndarray) -> np.ndarray:
        """Scale states."""
        if not self.scaler_fitted:
            raise RuntimeError("Scaler not fitted")
        return self.scaler.transform(states)

    def train(self, train_data: dict, val_data: dict):
        """Train GBM models."""
        train_states = self.scale_states(train_data['states'])
        train_actions = train_data['actions']
        train_rewards = train_data['rewards']
        train_next_states = self.scale_states(train_data['next_states'])
        train_dones = train_data['dones']

        val_states = self.scale_states(val_data['states'])
        val_actions = val_data['actions']
        val_rewards = val_data['rewards']
        val_next_states = self.scale_states(val_data['next_states'])
        val_dones = val_data['dones']

        q_targets = train_rewards.copy()

        for iteration in range(self.config.gbm.n_iterations):
            for action in range(self.n_actions):
                mask = train_actions == action
                if mask.sum() < 100:
                    continue

                X_train = train_states[mask]
                y_train = q_targets[mask]

                model = xgb.XGBRegressor(
                    n_estimators=self.config.gbm.n_estimators,
                    max_depth=self.config.gbm.max_depth,
                    learning_rate=self.config.gbm.learning_rate,
                    subsample=self.config.gbm.subsample,
                    colsample_bytree=self.config.gbm.colsample_bytree,
                    min_child_weight=self.config.gbm.min_child_weight,
                    random_state=self.config.random_seed + action,
                    n_jobs=-1,
                    verbosity=0
                )

                model.fit(X_train, y_train, verbose=False)
                self.models[action] = model

            if iteration < self.config.gbm.n_iterations - 1:
                next_q_values = self._predict_all_actions(train_next_states)
                next_q_max = next_q_values.max(axis=1)
                q_targets = train_rewards + self.config.gbm.gamma * next_q_max * (~train_dones)

    def _predict_all_actions(self, states: np.ndarray) -> np.ndarray:
        """Predict Q-values for all actions."""
        q_values = np.zeros((len(states), self.n_actions))
        for action in range(self.n_actions):
            if self.models[action] is not None:
                q_values[:, action] = self.models[action].predict(states)
        return q_values

    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        states_scaled = self.scale_states(states)
        return self._predict_all_actions(states_scaled)

    def select_actions(self, states: np.ndarray, use_noise: bool = False) -> np.ndarray:
        """Select actions with optional state noise."""
        if use_noise and self.config.gbm.use_state_noise:
            noise = np.random.randn(*states.shape) * self.config.gbm.state_noise_std
            states_noisy = states + noise
            q_values = self.get_q_values(states_noisy)
        else:
            q_values = self.get_q_values(states)

        return np.argmax(q_values, axis=1)


class EnsembleGBMAgent:
    """Ensemble of GBM agents."""

    def __init__(self, n_states: int, n_actions: int, config):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config
        self.agents: List[GBMQAgent] = []

        for i in range(config.gbm.ensemble_size):
            agent = GBMQAgent(n_states, n_actions, config)
            self.agents.append(agent)

    def fit_scalers(self, states: np.ndarray):
        """Fit scalers for all agents."""
        for agent in self.agents:
            agent.fit_scaler(states)

    def train(self, train_data: dict, val_data: dict):
        """Train all agents."""
        logger.info("=" * 60)
        logger.info("TRAINING GBM ENSEMBLE")
        logger.info("=" * 60)
        logger.info(f"Ensemble size: {self.config.gbm.ensemble_size}")
        logger.info(f"Q-iterations: {self.config.gbm.n_iterations}")

        for i, agent in enumerate(self.agents):
            logger.info(f"\nTraining member {i+1}/{self.config.gbm.ensemble_size}...")

            if i > 0:
                n_samples = len(train_data['states'])
                indices = np.random.choice(n_samples, n_samples, replace=True)
                bootstrap_data = {
                    'states': train_data['states'][indices],
                    'actions': train_data['actions'][indices],
                    'rewards': train_data['rewards'][indices],
                    'next_states': train_data['next_states'][indices],
                    'dones': train_data['dones'][indices]
                }
                agent.train(bootstrap_data, val_data)
            else:
                agent.train(train_data, val_data)

        logger.info("\nEnsemble training complete")

    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        """Get ensemble-averaged Q-values."""
        q_values_list = [agent.get_q_values(states) for agent in self.agents]
        return np.mean(q_values_list, axis=0)

    def select_actions(self, states: np.ndarray, use_noise: bool = False) -> np.ndarray:
        """Select actions using ensemble with optional state noise."""
        if use_noise and self.config.gbm.use_state_noise:
            noise = np.random.randn(*states.shape) * self.config.gbm.state_noise_std
            states_noisy = states + noise
            q_values = self.get_q_values(states_noisy)
        else:
            q_values = self.get_q_values(states)

        return np.argmax(q_values, axis=1)
