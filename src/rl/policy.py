"""
Policy Classes

Implements policies for action selection:
- Greedy policy (deterministic)
- Epsilon-greedy policy (exploration)

Author: AI Clinician Project
Date: 2024-11-16
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GreedyPolicy:
    """
    Greedy policy: always select action with highest Q-value.

    π(s) = argmax_a Q(s, a)
    """

    def __init__(self, q_function):
        """
        Initialize GreedyPolicy.

        Args:
            q_function: Q-learning model with predict_q_values method
        """
        self.q_function = q_function

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action for given state.

        Args:
            state: State vector (n_features,) or (1, n_features)

        Returns:
            Action index
        """
        # Ensure state is 2D
        if state.ndim == 1:
            state = state.reshape(1, -1)

        # Get Q-values for all actions
        q_values = self.q_function.predict_q_values(state, actions=None)

        # Select action with highest Q-value
        action = np.argmax(q_values)

        return int(action)

    def select_actions_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Select actions for batch of states.

        Args:
            states: State array (batch_size, n_features)

        Returns:
            Action array (batch_size,)
        """
        q_values = self.q_function.predict_q_values(states, actions=None)
        actions = np.argmax(q_values, axis=1)
        return actions


class EpsilonGreedyPolicy:
    """
    Epsilon-greedy policy: explore with probability epsilon.

    π(s) = {
        random action with probability ε
        argmax_a Q(s, a) with probability 1-ε
    }
    """

    def __init__(
        self,
        q_function,
        epsilon: float = 0.1,
        n_actions: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Initialize EpsilonGreedyPolicy.

        Args:
            q_function: Q-learning model
            epsilon: Exploration probability (0 to 1)
            n_actions: Number of actions (required for random exploration)
            random_seed: Random seed
        """
        self.q_function = q_function
        self.epsilon = epsilon
        self.n_actions = n_actions or q_function.n_actions
        self.rng = np.random.RandomState(random_seed)

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy strategy.

        Args:
            state: State vector

        Returns:
            Action index
        """
        # Explore: random action
        if self.rng.rand() < self.epsilon:
            action = self.rng.randint(0, self.n_actions)
        # Exploit: greedy action
        else:
            if state.ndim == 1:
                state = state.reshape(1, -1)
            q_values = self.q_function.predict_q_values(state, actions=None)
            action = np.argmax(q_values)

        return int(action)

    def select_actions_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Select actions for batch of states.

        Args:
            states: State array (batch_size, n_features)

        Returns:
            Action array (batch_size,)
        """
        batch_size = states.shape[0]

        # Random exploration mask
        explore_mask = self.rng.rand(batch_size) < self.epsilon

        # Initialize with greedy actions
        q_values = self.q_function.predict_q_values(states, actions=None)
        actions = np.argmax(q_values, axis=1)

        # Replace with random actions for exploration
        n_explore = np.sum(explore_mask)
        if n_explore > 0:
            random_actions = self.rng.randint(0, self.n_actions, n_explore)
            actions[explore_mask] = random_actions

        return actions


class BehaviorPolicy:
    """
    Estimated behavior policy from observed data.

    Uses empirical action probabilities: π_b(a|s) ≈ freq(a|s)
    """

    def __init__(self, n_actions: int, softening_epsilon: float = 0.01):
        """
        Initialize BehaviorPolicy.

        Args:
            n_actions: Number of actions
            softening_epsilon: Softening parameter to avoid zero probabilities
        """
        self.n_actions = n_actions
        self.epsilon = softening_epsilon

        # Will store empirical action probabilities
        self.action_probs = None  # (n_states, n_actions) - if tractable
        self.global_action_probs = None  # (n_actions,) - global empirical distribution

    def fit(self, states: np.ndarray, actions: np.ndarray):
        """
        Fit behavior policy from observed (state, action) pairs.

        Uses global action distribution (not state-dependent) for simplicity.

        Args:
            states: Observed states
            actions: Observed actions
        """
        logger.info("Fitting behavior policy from data...")

        # Compute empirical action distribution
        action_counts = np.bincount(actions.astype(int), minlength=self.n_actions)
        action_freqs = action_counts / len(actions)

        # Apply softening: (1-ε) * freq + ε / |A|
        self.global_action_probs = (
            (1 - self.epsilon) * action_freqs +
            self.epsilon / self.n_actions
        )

        logger.info(f"  Action distribution:")
        for a in range(min(10, self.n_actions)):
            logger.info(f"    Action {a}: {self.global_action_probs[a]:.4f}")
        if self.n_actions > 10:
            logger.info(f"    ... ({self.n_actions - 10} more actions)")

        logger.info(f"✓ Behavior policy fitted")

    def get_action_prob(self, state: np.ndarray, action: int) -> float:
        """
        Get probability of action under behavior policy.

        Args:
            state: State vector (not used - global policy)
            action: Action index

        Returns:
            Probability π_b(a|s)
        """
        if self.global_action_probs is None:
            raise ValueError("Behavior policy not fitted. Call fit() first.")

        return self.global_action_probs[action]

    def get_action_probs_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Get probabilities for batch of (state, action) pairs.

        Args:
            states: State array (batch_size, n_features)
            actions: Action array (batch_size,)

        Returns:
            Probability array (batch_size,)
        """
        if self.global_action_probs is None:
            raise ValueError("Behavior policy not fitted. Call fit() first.")

        return self.global_action_probs[actions.astype(int)]


def main():
    """Example usage of policy classes."""
    # Create dummy Q-function
    class DummyQFunction:
        def __init__(self):
            self.n_actions = 25

        def predict_q_values(self, states, actions=None):
            if actions is None:
                return np.random.randn(states.shape[0], self.n_actions)
            return np.random.randn(states.shape[0])

    q_func = DummyQFunction()

    # Greedy policy
    greedy_policy = GreedyPolicy(q_func)
    state = np.random.randn(10)
    action = greedy_policy.select_action(state)
    logger.info(f"Greedy action: {action}")

    # Epsilon-greedy policy
    eps_greedy_policy = EpsilonGreedyPolicy(q_func, epsilon=0.1)
    action = eps_greedy_policy.select_action(state)
    logger.info(f"Epsilon-greedy action: {action}")

    # Behavior policy
    behavior_policy = BehaviorPolicy(n_actions=25)
    states = np.random.randn(1000, 10)
    actions = np.random.randint(0, 25, 1000)
    behavior_policy.fit(states, actions)
    prob = behavior_policy.get_action_prob(states[0], actions[0])
    logger.info(f"Behavior policy prob: {prob:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
