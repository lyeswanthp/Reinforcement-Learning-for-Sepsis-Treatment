"""GBM Q-Learning agent with entropy regularization."""

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import List
import logging

logger = logging.getLogger(__name__)


class GBMQAgent:
    """Gradient Boosting Q-Learning agent with ensemble and entropy."""

    def __init__(self, n_states: int, n_actions: int, config):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config
        self.models = [None] * n_actions
        self.scaler = StandardScaler()
        self.scaler_fitted = False

        logger.info(f"GBM agent initialized:")
        logger.info(f"  State features: {n_states}")
        logger.info(f"  Actions: {n_actions}")
        logger.info(f"  Ensemble size: {config.gbm.ensemble_size}")
        logger.info(f"  N estimators: {config.gbm.n_estimators}")

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
        logger.info("=" * 60)
        logger.info("TRAINING GBM Q-LEARNING")
        logger.info("=" * 60)
        logger.info(f"Q-learning iterations: {self.config.gbm.n_iterations}")

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
            logger.info(f"\nIteration {iteration + 1}/{self.config.gbm.n_iterations}")

            for action in range(self.n_actions):
                mask = train_actions == action
                if mask.sum() < 100:
                    logger.warning(f"  Action {action}: skipped (only {mask.sum()} samples)")
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

                train_rmse = np.sqrt(np.mean((self.get_q_values(train_states)[
                    np.arange(len(train_actions)), train_actions
                ] - q_targets) ** 2))

                val_q_values = self.get_q_values(val_states)
                val_rmse = np.sqrt(np.mean((val_q_values[
                    np.arange(len(val_actions)), val_actions
                ] - val_rewards) ** 2))

                logger.info(f"  Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")

        logger.info("Training complete")

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

    def select_actions(self, states: np.ndarray, temperature: float = 0.0) -> np.ndarray:
        """Select actions with optional temperature sampling."""
        q_values = self.get_q_values(states)

        if temperature > 0:
            q_values = q_values - q_values.max(axis=1, keepdims=True)
            exp_q = np.exp(q_values / temperature)
            probs = exp_q / exp_q.sum(axis=1, keepdims=True)
            actions = np.array([np.random.choice(self.n_actions, p=p) for p in probs])
        else:
            actions = np.argmax(q_values, axis=1)

        return actions


class EnsembleGBMAgent:
    """Ensemble of GBM agents for robust predictions."""

    def __init__(self, n_states: int, n_actions: int, config):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config
        self.agents: List[GBMQAgent] = []

        logger.info("=" * 60)
        logger.info("CREATING GBM ENSEMBLE")
        logger.info("=" * 60)

        for i in range(config.gbm.ensemble_size):
            agent = GBMQAgent(n_states, n_actions, config)
            self.agents.append(agent)

    def fit_scalers(self, states: np.ndarray):
        """Fit scalers for all agents."""
        for agent in self.agents:
            agent.fit_scaler(states)

    def train(self, train_data: dict, val_data: dict):
        """Train all agents in ensemble."""
        for i, agent in enumerate(self.agents):
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING ENSEMBLE MEMBER {i+1}/{self.config.gbm.ensemble_size}")
            logger.info(f"{'='*60}")

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

    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        """Get ensemble-averaged Q-values."""
        q_values_list = [agent.get_q_values(states) for agent in self.agents]
        return np.mean(q_values_list, axis=0)

    def select_actions(self, states: np.ndarray, temperature: float = 0.0) -> np.ndarray:
        """Select actions using ensemble predictions."""
        q_values = self.get_q_values(states)

        if temperature > 0:
            q_values = q_values - q_values.max(axis=1, keepdims=True)
            exp_q = np.exp(q_values / temperature)
            probs = exp_q / exp_q.sum(axis=1, keepdims=True)
            actions = np.array([np.random.choice(self.n_actions, p=p) for p in probs])
        else:
            actions = np.argmax(q_values, axis=1)

        return actions
