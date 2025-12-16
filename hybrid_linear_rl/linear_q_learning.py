"""
Linear Q-learning with L2 regularization (Ridge regression).
Implementation follows the proposal's linear function approximation.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import Config
import logging

logger = logging.getLogger(__name__)


class LinearQAgent:
    """Linear Q-function with state-action features and L2 regularization."""

    def __init__(self, n_states: int, n_actions: int, config: Config):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config

        n_state_features = n_states
        n_action_features = n_actions
        n_interaction_features = n_states * n_actions

        self.n_features = n_state_features + n_action_features + n_interaction_features

        self.weights = np.zeros(self.n_features)
        self.scaler = StandardScaler()
        self.scaler_fitted = False

        logger.info(f"Linear Q-agent initialized:")
        logger.info(f"  State features: {n_state_features}")
        logger.info(f"  Action features: {n_action_features}")
        logger.info(f"  Interaction features: {n_interaction_features}")
        logger.info(f"  Total features: {self.n_features}")

    def fit_scaler(self, states: np.ndarray):
        self.scaler.fit(states)
        self.scaler_fitted = True

    def _create_features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if not self.scaler_fitted:
            raise RuntimeError("Scaler not fitted")

        states_scaled = self.scaler.transform(states)
        n_samples = len(states)

        action_onehot = np.zeros((n_samples, self.n_actions))
        action_onehot[np.arange(n_samples), actions.astype(int)] = 1

        interactions = []
        for i in range(n_samples):
            interaction = np.outer(states_scaled[i], action_onehot[i]).flatten()
            interactions.append(interaction)
        interactions = np.array(interactions)

        features = np.concatenate([
            states_scaled,
            action_onehot,
            interactions
        ], axis=1)

        return features

    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        n_samples = len(states)
        q_values = np.zeros((n_samples, self.n_actions))

        for action in range(self.n_actions):
            actions = np.full(n_samples, action)
            features = self._create_features(states, actions)
            q_values[:, action] = features @ self.weights

        return q_values

    def select_actions(self, states: np.ndarray) -> np.ndarray:
        q_values = self.get_q_values(states)
        return np.argmax(q_values, axis=1)

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> float:
        features = self._create_features(states, actions)
        current_q = features @ self.weights

        next_q_values = self.get_q_values(next_states)
        next_q_max = np.max(next_q_values, axis=1)
        targets = rewards + self.config.linear.gamma * next_q_max * (1 - dones.astype(float))

        td_error = targets - current_q

        gradient = features.T @ td_error / len(states)
        l2_penalty = self.config.linear.l2_lambda * self.weights

        self.weights += self.config.linear.learning_rate * (gradient - l2_penalty)

        loss = np.mean(td_error ** 2)
        return loss

    def train(
        self,
        train_data: dict,
        val_data: dict
    ) -> dict:
        logger.info("="*60)
        logger.info("TRAINING LINEAR Q-LEARNING")
        logger.info("="*60)
        logger.info(f"Learning rate: {self.config.linear.learning_rate}")
        logger.info(f"L2 lambda: {self.config.linear.l2_lambda}")
        logger.info(f"Gamma: {self.config.linear.gamma}")

        history = {
            'train_loss': [],
            'val_loss': [],
            'n_unique_actions': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for epoch in range(self.config.linear.n_epochs):
            indices = np.random.permutation(len(train_data['states']))

            epoch_losses = []
            for start_idx in range(0, len(indices), self.config.linear.batch_size):
                batch_idx = indices[start_idx:start_idx + self.config.linear.batch_size]

                loss = self.update(
                    train_data['states'][batch_idx],
                    train_data['actions'][batch_idx],
                    train_data['rewards'][batch_idx],
                    train_data['next_states'][batch_idx],
                    train_data['dones'][batch_idx]
                )
                epoch_losses.append(loss)

            train_loss = np.mean(epoch_losses)
            history['train_loss'].append(train_loss)

            val_loss, n_unique = self._validate(val_data)
            history['val_loss'].append(val_loss)
            history['n_unique_actions'].append(n_unique)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}: "
                    f"Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, "
                    f"Actions={n_unique}"
                )

            if val_loss < best_val_loss - self.config.linear.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = self.weights.copy()
            else:
                patience_counter += 1

            if patience_counter >= self.config.linear.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if best_weights is not None:
            self.weights = best_weights

        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        return history

    def _validate(self, val_data: dict) -> tuple:
        features = self._create_features(val_data['states'], val_data['actions'])
        current_q = features @ self.weights

        next_q_values = self.get_q_values(val_data['next_states'])
        next_q_max = np.max(next_q_values, axis=1)
        targets = val_data['rewards'] + self.config.linear.gamma * next_q_max * (1 - val_data['dones'].astype(float))

        val_loss = np.mean((targets - current_q) ** 2)

        val_actions = self.select_actions(val_data['states'][:10000])
        n_unique = len(np.unique(val_actions))

        return val_loss, n_unique
