"""
Gradient Boosting Q-learning using XGBoost.
"""
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from config import Config
import logging

logger = logging.getLogger(__name__)


class GBMQAgent:
    """Gradient Boosting Machine for Q-function approximation."""

    def __init__(self, n_states: int, n_actions: int, config: Config):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config

        self.models = {}
        for action in range(n_actions):
            self.models[action] = None

        self.scaler = StandardScaler()
        self.scaler_fitted = False

        logger.info(f"GBM Q-agent initialized:")
        logger.info(f"  State features: {n_states}")
        logger.info(f"  Actions: {n_actions}")
        logger.info(f"  N estimators: {config.gbm.n_estimators}")
        logger.info(f"  Max depth: {config.gbm.max_depth}")

    def fit_scaler(self, states: np.ndarray):
        self.scaler.fit(states)
        self.scaler_fitted = True

    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        if not self.scaler_fitted:
            raise RuntimeError("Scaler not fitted")

        states_scaled = self.scaler.transform(states)
        n_samples = len(states)
        q_values = np.zeros((n_samples, self.n_actions))

        for action in range(self.n_actions):
            if self.models[action] is not None:
                q_values[:, action] = self.models[action].predict(states_scaled)

        return q_values

    def select_actions(self, states: np.ndarray) -> np.ndarray:
        q_values = self.get_q_values(states)
        return np.argmax(q_values, axis=1)

    def train(self, train_data: dict, val_data: dict) -> dict:
        logger.info("="*60)
        logger.info("TRAINING GRADIENT BOOSTING Q-LEARNING")
        logger.info("="*60)

        train_states_scaled = self.scaler.transform(train_data['states'])
        val_states_scaled = self.scaler.transform(val_data['states'])

        next_q_values_train = np.zeros((len(train_data['states']), self.n_actions))
        next_q_values_val = np.zeros((len(val_data['states']), self.n_actions))

        history = {'train_loss': [], 'val_loss': [], 'n_unique_actions': []}

        for iteration in range(3):
            logger.info(f"\nIteration {iteration + 1}/3:")

            for action in range(self.n_actions):
                action_mask = train_data['actions'] == action

                if np.sum(action_mask) < 10:
                    logger.warning(f"  Action {action}: insufficient samples ({np.sum(action_mask)})")
                    continue

                X_train = train_states_scaled[action_mask]
                next_states_train = train_data['next_states'][action_mask]
                rewards_train = train_data['rewards'][action_mask]
                dones_train = train_data['dones'][action_mask]

                next_states_train_scaled = self.scaler.transform(next_states_train)
                next_q_max = np.max(next_q_values_train[action_mask], axis=1)
                y_train = rewards_train + self.config.linear.gamma * next_q_max * (1 - dones_train.astype(float))

                model = xgb.XGBRegressor(
                    n_estimators=self.config.gbm.n_estimators,
                    max_depth=self.config.gbm.max_depth,
                    learning_rate=self.config.gbm.learning_rate,
                    subsample=self.config.gbm.subsample,
                    random_state=self.config.gbm.random_state,
                    n_jobs=-1,
                    verbosity=0
                )

                model.fit(X_train, y_train)
                self.models[action] = model

                next_q_values_train[:, action] = model.predict(self.scaler.transform(train_data['next_states']))
                next_q_values_val[:, action] = model.predict(self.scaler.transform(val_data['next_states']))

                logger.info(f"  Action {action}: {np.sum(action_mask):,} samples trained")

            train_loss = self._compute_loss(
                train_data['states'], train_data['actions'],
                train_data['rewards'], train_data['next_states'],
                train_data['dones'], next_q_values_train
            )
            val_loss = self._compute_loss(
                val_data['states'], val_data['actions'],
                val_data['rewards'], val_data['next_states'],
                val_data['dones'], next_q_values_val
            )

            val_actions = self.select_actions(val_data['states'][:10000])
            n_unique = len(np.unique(val_actions))

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['n_unique_actions'].append(n_unique)

            logger.info(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Actions: {n_unique}")

        logger.info("Training complete")
        return history

    def _compute_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        next_q_values: np.ndarray
    ) -> float:
        q_values = self.get_q_values(states)
        current_q = q_values[np.arange(len(actions)), actions.astype(int)]

        next_q_max = np.max(next_q_values, axis=1)
        targets = rewards + self.config.linear.gamma * next_q_max * (1 - dones.astype(float))

        return np.mean((targets - current_q) ** 2)
