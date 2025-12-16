"""
Linear Q-Learning with SGD

Implements approximate Q-learning using linear function approximation.
Optimized using stochastic gradient descent (SGD) with L2 regularization.

Based on:
- Komorowski et al. (2018) "The AI Clinician"
- Sutton & Barto (2018) "Reinforcement Learning: An Introduction"

Author: AI Clinician Project
Date: 2024-11-16
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LinearQLearning:
    """
    Linear Approximate Q-Learning with SGD.

    Q(s, a) = w^T * φ(s, a)

    where:
    - w: weight vector
    - φ(s, a): feature vector (state features + action one-hot + interactions)
    """

    def __init__(
        self,
        n_state_features: int,
        n_actions: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        l2_lambda: float = 0.0001,
        random_seed: int = 42
    ):
        """
        Initialize LinearQLearning.

        Args:
            n_state_features: Number of state features
            n_actions: Number of discrete actions
            learning_rate: Learning rate (α) for SGD
            gamma: Discount factor (γ)
            l2_lambda: L2 regularization coefficient (λ)
            random_seed: Random seed for reproducibility
        """
        self.n_state_features = n_state_features
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.l2_lambda = l2_lambda
        self.random_seed = random_seed

        np.random.seed(random_seed)

        # Feature engineering: state + action one-hot + interactions
        # Total features = n_state + n_actions + (n_state * n_actions)
        self.n_action_features = n_actions
        self.n_interaction_features = n_state_features * n_actions
        self.n_total_features = n_state_features + n_action_features + self.n_interaction_features

        # Initialize weights (small random values)
        self.weights = np.random.randn(self.n_total_features) * 0.01

        # Training history
        self.training_history = []

        logger.info(f"LinearQLearning initialized")
        logger.info(f"  State features: {n_state_features}")
        logger.info(f"  Actions: {n_actions}")
        logger.info(f"  Total features (with interactions): {self.n_total_features}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Gamma: {gamma}")
        logger.info(f"  L2 lambda: {l2_lambda}")

    def compute_features(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> np.ndarray:
        """
        Compute feature vector φ(s, a) for linear Q-function.

        Features = [state_features, action_one_hot, state_x_action_interactions]

        Args:
            states: (batch_size, n_state_features) array
            actions: (batch_size,) array of action indices

        Returns:
            (batch_size, n_total_features) feature array
        """
        batch_size = states.shape[0]

        # Ensure states is 2D
        if states.ndim == 1:
            states = states.reshape(1, -1)

        # Ensure actions is 1D
        if actions.ndim == 0:
            actions = np.array([actions])

        # 1. State features (as is)
        state_features = states  # (batch_size, n_state_features)

        # 2. Action one-hot encoding
        action_one_hot = np.zeros((batch_size, self.n_actions))
        action_one_hot[np.arange(batch_size), actions.astype(int)] = 1

        # 3. State-action interactions (outer product)
        # For each sample: state × action_one_hot
        interactions = []
        for i in range(batch_size):
            interaction = np.outer(states[i], action_one_hot[i]).flatten()
            interactions.append(interaction)
        interactions = np.array(interactions)  # (batch_size, n_state * n_actions)

        # Concatenate all features
        features = np.concatenate([state_features, action_one_hot, interactions], axis=1)

        return features

    def predict_q_values(
        self,
        states: np.ndarray,
        actions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict Q-values for given states and actions.

        If actions is None, returns Q-values for all actions.

        Args:
            states: (batch_size, n_state_features) or (n_state_features,) array
            actions: (batch_size,) array of actions, or None for all actions

        Returns:
            Q-values: (batch_size,) if actions given, (batch_size, n_actions) otherwise
        """
        # Ensure states is 2D
        if states.ndim == 1:
            states = states.reshape(1, -1)
            single_state = True
        else:
            single_state = False

        batch_size = states.shape[0]

        if actions is None:
            # Compute Q-values for all actions
            q_values = np.zeros((batch_size, self.n_actions))

            for action in range(self.n_actions):
                actions_batch = np.full(batch_size, action)
                features = self.compute_features(states, actions_batch)
                q_values[:, action] = features @ self.weights

            if single_state:
                return q_values.flatten()
            return q_values
        else:
            # Compute Q-values for specific actions
            features = self.compute_features(states, actions)
            q_values = features @ self.weights

            if single_state:
                return q_values[0]
            return q_values

    def get_greedy_actions(self, states: np.ndarray) -> np.ndarray:
        """
        Get greedy actions (argmax Q-value) for given states.

        Args:
            states: (batch_size, n_state_features) array

        Returns:
            (batch_size,) array of greedy actions
        """
        q_values_all = self.predict_q_values(states, actions=None)  # (batch_size, n_actions)
        greedy_actions = np.argmax(q_values_all, axis=1)
        return greedy_actions

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> float:
        """
        Perform one SGD update step.

        Update rule:
        w ← w + α * [r + γ * max_a' Q(s', a') - Q(s, a)] * ∇_w Q(s, a) - α * λ * w

        Args:
            states: (batch_size, n_state_features) array
            actions: (batch_size,) array
            rewards: (batch_size,) array
            next_states: (batch_size, n_state_features) array
            dones: (batch_size,) array of booleans

        Returns:
            Mean squared Bellman error (loss)
        """
        # Current Q-values: Q(s, a)
        features_current = self.compute_features(states, actions)
        q_current = features_current @ self.weights

        # Next Q-values: max_a' Q(s', a')
        q_next_all = self.predict_q_values(next_states, actions=None)  # (batch_size, n_actions)
        q_next_max = np.max(q_next_all, axis=1)

        # Terminal states have no future value
        q_next_max = q_next_max * (1 - dones)

        # TD target: r + γ * max_a' Q(s', a')
        td_target = rewards + self.gamma * q_next_max

        # TD error: δ = target - current
        td_error = td_target - q_current

        # Gradient: ∇_w Q(s, a) = φ(s, a)
        gradient = features_current.T @ td_error / len(states)

        # L2 regularization gradient
        reg_gradient = self.l2_lambda * self.weights

        # SGD update
        self.weights += self.learning_rate * gradient - self.learning_rate * reg_gradient

        # Compute loss (mean squared Bellman error)
        loss = np.mean(td_error ** 2)

        return loss

    def train_epoch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        batch_size: int = 256
    ) -> float:
        """
        Train for one epoch (one pass through the dataset).

        Args:
            states: Training states
            actions: Training actions
            rewards: Training rewards
            next_states: Training next states
            dones: Training dones
            batch_size: Mini-batch size for SGD

        Returns:
            Mean loss over epoch
        """
        n_samples = states.shape[0]
        indices = np.random.permutation(n_samples)

        epoch_losses = []

        # Mini-batch SGD
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_loss = self.update(
                states[batch_indices],
                actions[batch_indices],
                rewards[batch_indices],
                next_states[batch_indices],
                dones[batch_indices]
            )

            epoch_losses.append(batch_loss)

        return np.mean(epoch_losses)

    def fit(
        self,
        train_states: np.ndarray,
        train_actions: np.ndarray,
        train_rewards: np.ndarray,
        train_next_states: np.ndarray,
        train_dones: np.ndarray,
        val_states: Optional[np.ndarray] = None,
        val_actions: Optional[np.ndarray] = None,
        val_rewards: Optional[np.ndarray] = None,
        val_next_states: Optional[np.ndarray] = None,
        val_dones: Optional[np.ndarray] = None,
        n_epochs: int = 100,
        batch_size: int = 256,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Fit the Q-learning model.

        Args:
            train_*: Training data
            val_*: Validation data (optional)
            n_epochs: Maximum number of epochs
            batch_size: Mini-batch size
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        logger.info(f"Training Linear Q-Learning...")
        logger.info(f"  Training samples: {train_states.shape[0]:,}")
        if val_states is not None:
            logger.info(f"  Validation samples: {val_states.shape[0]:,}")
        logger.info(f"  Epochs: {n_epochs}")
        logger.info(f"  Batch size: {batch_size}")

        best_val_loss = np.inf
        patience_counter = 0
        best_weights = self.weights.copy()

        for epoch in range(n_epochs):
            # Train one epoch
            train_loss = self.train_epoch(
                train_states, train_actions, train_rewards,
                train_next_states, train_dones, batch_size
            )

            # Validation loss
            val_loss = None
            if val_states is not None:
                val_q = self.predict_q_values(val_states, val_actions)
                val_q_next = self.predict_q_values(val_next_states, actions=None)
                val_q_next_max = np.max(val_q_next, axis=1) * (1 - val_dones)
                val_target = val_rewards + self.gamma * val_q_next_max
                val_loss = np.mean((val_target - val_q) ** 2)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = self.weights.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

            # Log progress
            if verbose and (epoch + 1) % 10 == 0:
                if val_loss is not None:
                    logger.info(f"  Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                else:
                    logger.info(f"  Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}")

            # Save history
            history_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            self.training_history.append(history_entry)

        # Restore best weights
        if val_states is not None:
            self.weights = best_weights
            logger.info(f"✓ Restored best weights (val_loss={best_val_loss:.4f})")

        logger.info(f"✓ Training complete ({epoch+1} epochs)")

        return {'history': self.training_history, 'best_val_loss': best_val_loss}

    def save(self, output_path: str):
        """Save model to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'weights': self.weights,
            'n_state_features': self.n_state_features,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'l2_lambda': self.l2_lambda,
            'training_history': self.training_history,
        }

        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"✓ Model saved to {output_path}")

    def load(self, input_path: str):
        """Load model from file."""
        with open(input_path, 'rb') as f:
            model_data = pickle.load(f)

        self.weights = model_data['weights']
        self.n_state_features = model_data['n_state_features']
        self.n_actions = model_data['n_actions']
        self.learning_rate = model_data['learning_rate']
        self.gamma = model_data['gamma']
        self.l2_lambda = model_data['l2_lambda']
        self.training_history = model_data.get('training_history', [])

        # Recompute feature dimensions
        self.n_action_features = self.n_actions
        self.n_interaction_features = self.n_state_features * self.n_actions
        self.n_total_features = self.n_state_features + self.n_action_features + self.n_interaction_features

        logger.info(f"✓ Model loaded from {input_path}")


def main():
    """Example usage of LinearQLearning."""
    # Create dummy data
    n_samples = 1000
    n_state_features = 10
    n_actions = 25

    states = np.random.randn(n_samples, n_state_features)
    actions = np.random.randint(0, n_actions, n_samples)
    rewards = np.random.randn(n_samples)
    next_states = np.random.randn(n_samples, n_state_features)
    dones = np.random.rand(n_samples) < 0.1

    # Initialize model
    model = LinearQLearning(
        n_state_features=n_state_features,
        n_actions=n_actions,
        learning_rate=0.001,
        gamma=0.99
    )

    # Train
    model.fit(
        states, actions, rewards, next_states, dones,
        n_epochs=50,
        batch_size=32,
        verbose=True
    )

    # Predict
    q_values = model.predict_q_values(states[:5])
    logger.info(f"Q-values shape: {q_values.shape}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
