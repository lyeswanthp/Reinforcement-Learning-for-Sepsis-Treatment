"""
Q-Learning Training Pipeline

Trains a linear Q-function on the sepsis treatment trajectories.

Features:
- Linear function approximation: Q(s,a) = w^T * φ(s,a)
- State features + action one-hot + state-action interactions
- Fitted Q-Iteration (FQI) algorithm
- Validation-based early stopping
- WDR off-policy evaluation

Author: AI Clinician Project
Date: 2024-11-22
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import ConfigLoader


class LinearQFunction:
    """
    Linear Q-function with feature engineering.

    Q(s, a) = w^T * φ(s, a)

    Where φ(s, a) includes:
    - State features (normalized)
    - Action one-hot encoding (25 actions)
    - State × Action interactions
    """

    def __init__(
        self,
        n_state_features: int,
        n_actions: int = 25,
        alpha: float = 1.0,  # L2 regularization
        gamma: float = 0.99
    ):
        """
        Initialize linear Q-function.

        Args:
            n_state_features: Number of state features
            n_actions: Number of discrete actions
            alpha: L2 regularization strength
            gamma: Discount factor
        """
        self.n_state_features = n_state_features
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        # Feature dimension: state + action_onehot + state*action
        self.n_features = n_state_features + n_actions + n_state_features * n_actions

        # State scaler (will be fit on training data)
        self.state_scaler = StandardScaler()

        # Ridge regression model for each iteration
        self.model = Ridge(alpha=alpha, fit_intercept=True)

        # Store training history
        self.training_history = []

        logger.info(f"LinearQFunction initialized:")
        logger.info(f"  State features: {n_state_features}")
        logger.info(f"  Actions: {n_actions}")
        logger.info(f"  Total features: {self.n_features}")
        logger.info(f"  Regularization: α={alpha}")
        logger.info(f"  Discount: γ={gamma}")

    def _create_features(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Create feature vector φ(s, a).

        Args:
            states: State features (n_samples, n_state_features)
            actions: Action indices (n_samples,)
            normalize: Whether to normalize states

        Returns:
            Feature matrix (n_samples, n_features)
        """
        n_samples = len(states)

        # Normalize states
        if normalize:
            states_norm = self.state_scaler.transform(states)
        else:
            states_norm = states

        # One-hot encode actions
        action_onehot = np.zeros((n_samples, self.n_actions))
        action_onehot[np.arange(n_samples), actions.astype(int)] = 1

        # State-action interactions (state × action_onehot)
        # This creates different weights for each action
        interactions = np.zeros((n_samples, self.n_state_features * self.n_actions))
        for i in range(n_samples):
            a = int(actions[i])
            start_idx = a * self.n_state_features
            end_idx = (a + 1) * self.n_state_features
            interactions[i, start_idx:end_idx] = states_norm[i]

        # Concatenate all features
        features = np.hstack([states_norm, action_onehot, interactions])

        return features

    def fit_scaler(self, states: np.ndarray):
        """Fit the state scaler on training data."""
        self.state_scaler.fit(states)
        logger.info("State scaler fitted")

    def predict_q(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> np.ndarray:
        """
        Predict Q-values for state-action pairs.

        Args:
            states: State features (n_samples, n_state_features)
            actions: Action indices (n_samples,)

        Returns:
            Q-values (n_samples,)
        """
        features = self._create_features(states, actions)
        return self.model.predict(features)

    def predict_all_q(self, states: np.ndarray) -> np.ndarray:
        """
        Predict Q-values for all actions.

        Args:
            states: State features (n_samples, n_state_features)

        Returns:
            Q-values for all actions (n_samples, n_actions)
        """
        n_samples = len(states)
        q_values = np.zeros((n_samples, self.n_actions))

        for a in range(self.n_actions):
            actions = np.full(n_samples, a)
            q_values[:, a] = self.predict_q(states, actions)

        return q_values

    def get_greedy_actions(self, states: np.ndarray) -> np.ndarray:
        """Get greedy actions (argmax Q)."""
        q_values = self.predict_all_q(states)
        return np.argmax(q_values, axis=1)

    def get_policy_probs(
        self,
        states: np.ndarray,
        temperature: float = 0.1
    ) -> np.ndarray:
        """
        Get softmax policy probabilities.

        Args:
            states: State features
            temperature: Softmax temperature (lower = more greedy)

        Returns:
            Policy probabilities (n_samples, n_actions)
        """
        q_values = self.predict_all_q(states)

        # Softmax with temperature
        q_scaled = q_values / temperature
        q_max = np.max(q_scaled, axis=1, keepdims=True)
        exp_q = np.exp(q_scaled - q_max)
        probs = exp_q / np.sum(exp_q, axis=1, keepdims=True)

        return probs


class FittedQIteration:
    """
    Fitted Q-Iteration algorithm for batch reinforcement learning.

    Algorithm:
    1. Initialize Q arbitrarily
    2. For each iteration:
       a. Compute targets: y_i = r_i + γ * max_a' Q(s'_i, a')
       b. Fit Q to minimize MSE between Q(s_i, a_i) and y_i
    3. Repeat until convergence
    """

    def __init__(
        self,
        q_function: LinearQFunction,
        gamma: float = 0.99,
        n_iterations: int = 100,
        convergence_threshold: float = 1e-4,
        patience: int = 10
    ):
        """
        Initialize FQI.

        Args:
            q_function: Linear Q-function to train
            gamma: Discount factor
            n_iterations: Maximum iterations
            convergence_threshold: Stop if improvement < threshold
            patience: Early stopping patience
        """
        self.q_function = q_function
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        self.patience = patience

        logger.info(f"FittedQIteration initialized:")
        logger.info(f"  Max iterations: {n_iterations}")
        logger.info(f"  Convergence threshold: {convergence_threshold}")
        logger.info(f"  Patience: {patience}")

    def fit(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict:
        """
        Train Q-function using Fitted Q-Iteration.

        Args:
            train_data: Dictionary with keys:
                - 'states': (n_samples, n_features)
                - 'actions': (n_samples,)
                - 'rewards': (n_samples,)
                - 'next_states': (n_samples, n_features)
                - 'dones': (n_samples,)
            val_data: Optional validation data (same format)

        Returns:
            Training history dictionary
        """
        logger.info("Starting Fitted Q-Iteration training...")

        # Extract training data
        states = train_data['states']
        actions = train_data['actions']
        rewards = train_data['rewards']
        next_states = train_data['next_states']
        dones = train_data['dones']

        n_samples = len(states)
        logger.info(f"Training samples: {n_samples:,}")

        # Fit state scaler on training states
        self.q_function.fit_scaler(states)

        # Create features for current states
        features = self.q_function._create_features(states, actions)

        # Initialize targets with rewards (Q = R for first iteration)
        targets = rewards.copy()

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'q_mean': [],
            'q_std': [],
            'iteration': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for iteration in range(self.n_iterations):
            # Fit Q-function to current targets
            self.q_function.model.fit(features, targets)

            # Compute Q-values for current state-actions
            q_pred = self.q_function.model.predict(features)
            train_loss = np.mean((q_pred - targets) ** 2)

            # Compute new targets: y = r + γ * max_a' Q(s', a')
            next_q_all = self.q_function.predict_all_q(next_states)
            next_q_max = np.max(next_q_all, axis=1)

            # Terminal states have no future value
            next_q_max[dones.astype(bool)] = 0

            # Update targets
            new_targets = rewards + self.gamma * next_q_max

            # Compute target change (for convergence check)
            target_change = np.mean(np.abs(new_targets - targets))
            targets = new_targets

            # Validation loss
            val_loss = None
            if val_data is not None:
                val_q_pred = self.q_function.predict_q(
                    val_data['states'],
                    val_data['actions']
                )
                val_targets = val_data['rewards'] + self.gamma * np.where(
                    val_data['dones'].astype(bool),
                    0,
                    np.max(self.q_function.predict_all_q(val_data['next_states']), axis=1)
                )
                val_loss = np.mean((val_q_pred - val_targets) ** 2)

            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['q_mean'].append(np.mean(q_pred))
            history['q_std'].append(np.std(q_pred))
            history['iteration'].append(iteration)

            # Logging
            if (iteration + 1) % 10 == 0 or iteration == 0:
                log_msg = f"Iteration {iteration + 1}/{self.n_iterations}: "
                log_msg += f"Train Loss={train_loss:.4f}, "
                log_msg += f"Target Δ={target_change:.4f}, "
                log_msg += f"Q={np.mean(q_pred):.2f}±{np.std(q_pred):.2f}"
                if val_loss is not None:
                    log_msg += f", Val Loss={val_loss:.4f}"
                logger.info(log_msg)

            # Early stopping based on validation loss
            if val_loss is not None:
                if val_loss < best_val_loss - self.convergence_threshold:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model weights
                    best_weights = self.q_function.model.coef_.copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at iteration {iteration + 1}")
                        # Restore best weights
                        if best_weights is not None:
                            self.q_function.model.coef_ = best_weights
                        break

            # Check convergence
            if target_change < self.convergence_threshold:
                logger.info(f"Converged at iteration {iteration + 1}")
                break

        logger.info("Training complete!")
        logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        if val_loss is not None:
            logger.info(f"  Best val loss: {best_val_loss:.4f}")
        logger.info(f"  Final Q: {history['q_mean'][-1]:.2f} ± {history['q_std'][-1]:.2f}")

        return history


def load_trajectory_data(file_path: str, state_cols: List[str]) -> Dict[str, np.ndarray]:
    """
    Load trajectory data from CSV.

    Args:
        file_path: Path to trajectory CSV
        state_cols: List of state feature column names

    Returns:
        Dictionary with states, actions, rewards, next_states, dones
    """
    logger.info(f"Loading {file_path}...")
    df = pd.read_csv(file_path)

    # State columns (current and next)
    next_state_cols = [f'next_{col}' for col in state_cols]

    # Extract data
    states = df[state_cols].values.astype(np.float32)
    actions = df['action'].values.astype(np.int32)
    rewards = df['reward'].values.astype(np.float32)
    next_states = df[next_state_cols].values.astype(np.float32)
    dones = df['done'].values.astype(np.float32)

    # Handle NaN in next states (terminal states)
    next_states = np.nan_to_num(next_states, nan=0.0)

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'dones': dones,
        'df': df  # Keep original for analysis
    }


def compute_behavior_policy(df: pd.DataFrame, n_actions: int = 25) -> np.ndarray:
    """
    Estimate behavior policy probabilities from data.

    Uses empirical action frequencies per state cluster.
    """
    # Simple approach: global action frequencies
    action_counts = df['action'].value_counts()
    probs = np.zeros(n_actions)
    for a, count in action_counts.items():
        probs[int(a)] = count
    probs = probs / probs.sum()

    # Return probabilities for each sample
    behavior_probs = probs[df['action'].values.astype(int)]

    return behavior_probs


def evaluate_wdr(
    q_function: LinearQFunction,
    data: Dict[str, np.ndarray],
    df: pd.DataFrame,
    gamma: float = 0.99
) -> Dict:
    """
    Evaluate policy using Weighted Doubly Robust estimator.
    """
    logger.info("Computing WDR evaluation...")

    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']

    # Get Q-values and policy
    q_all = q_function.predict_all_q(states)
    target_actions = np.argmax(q_all, axis=1)

    # Target policy (greedy)
    target_probs = np.zeros_like(actions, dtype=np.float64)
    target_probs[actions == target_actions] = 1.0

    # For non-greedy actions, assign small probability
    target_probs = np.where(target_probs > 0, 0.95, 0.05 / 24)

    # Behavior policy (estimated from data)
    behavior_probs = compute_behavior_policy(df)

    # Importance sampling ratio
    rho = target_probs / (behavior_probs + 1e-10)
    rho = np.clip(rho, 0, 100)  # Clip extreme ratios

    # Compute per-trajectory WDR estimate
    wdr_values = []
    for stay_id in df['stay_id'].unique():
        mask = df['stay_id'] == stay_id
        traj_rewards = rewards[mask]
        traj_rho = rho[mask]
        traj_q = q_all[mask]
        traj_actions = actions[mask]

        # Simple WDR approximation
        n_steps = len(traj_rewards)
        cum_rho = np.cumprod(traj_rho)

        # Value estimate
        V = 0
        for t in range(n_steps - 1, -1, -1):
            r_t = traj_rewards[t]
            q_t = traj_q[t, int(traj_actions[t])]
            V_next = traj_q[t, int(traj_actions[t])] if t < n_steps - 1 else 0
            rho_t = cum_rho[t] if t > 0 else traj_rho[t]

            # WDR term
            V = rho_t * (r_t + gamma * V_next - q_t) + q_t

        wdr_values.append(V)

    wdr_estimate = np.mean(wdr_values)
    wdr_std = np.std(wdr_values)

    # 95% CI
    n_traj = len(wdr_values)
    ci_95 = 1.96 * wdr_std / np.sqrt(n_traj)

    logger.info(f"WDR estimate: {wdr_estimate:.3f} ± {ci_95:.3f}")
    logger.info(f"  Standard deviation: {wdr_std:.3f}")
    logger.info(f"  Number of trajectories: {n_traj:,}")

    return {
        'wdr_estimate': wdr_estimate,
        'wdr_std': wdr_std,
        'wdr_ci_95': ci_95,
        'n_trajectories': n_traj,
        'trajectory_values': wdr_values
    }


def main():
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("Q-LEARNING TRAINING PIPELINE")
    logger.info("=" * 80)

    # Load config
    config = ConfigLoader('configs/config.yaml').config

    # Paths
    data_dir = Path('data/processed')
    output_dir = Path('outputs/models')
    output_dir.mkdir(parents=True, exist_ok=True)

    # State feature columns (from trajectory files)
    # These are the features we use for the Q-function
    # State features that have corresponding next_* columns
    # Note: Treatment columns (input_4hourly, max_dose_vaso, etc.) are actions, not state
    state_cols = [
        # Vital signs
        'DiaBP', 'FiO2_1', 'HR', 'MeanBP', 'RR', 'SpO2', 'SysBP', 'Temp_C',
        # Lab values
        'Arterial_BE', 'Arterial_lactate', 'Arterial_pH', 'Calcium', 'Chloride',
        'Creatinine', 'Glucose', 'HCO3', 'Hb', 'INR', 'Magnesium', 'PT',
        'Platelets_count', 'Potassium', 'SGOT', 'SGPT', 'Sodium', 'Total_bili',
        'WBC_count', 'paCO2', 'paO2',
        # Demographics
        'gender', 'age', 're_admission',
        # Derived features
        'PaO2_FiO2', 'Shock_Index', 'SOFA'
    ]

    # ==========================================================================
    # STEP 1: LOAD DATA
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 1: LOADING TRAJECTORY DATA")
    logger.info("=" * 40)

    train_data = load_trajectory_data(data_dir / 'train_trajectories.csv', state_cols)
    val_data = load_trajectory_data(data_dir / 'val_trajectories.csv', state_cols)
    test_data = load_trajectory_data(data_dir / 'test_trajectories.csv', state_cols)

    logger.info(f"  Train: {len(train_data['states']):,} transitions")
    logger.info(f"  Val: {len(val_data['states']):,} transitions")
    logger.info(f"  Test: {len(test_data['states']):,} transitions")

    # ==========================================================================
    # STEP 2: INITIALIZE Q-FUNCTION
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 2: INITIALIZING Q-FUNCTION")
    logger.info("=" * 40)

    # Get hyperparameters from config
    rl_config = config.get('rl', {})
    gamma = rl_config.get('gamma', 0.99)
    alpha = rl_config.get('l2_regularization', 1.0)
    n_iterations = rl_config.get('n_iterations', 100)

    n_state_features = len(state_cols)
    n_actions = 25

    q_function = LinearQFunction(
        n_state_features=n_state_features,
        n_actions=n_actions,
        alpha=alpha,
        gamma=gamma
    )

    # ==========================================================================
    # STEP 3: TRAIN Q-FUNCTION
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 3: TRAINING Q-FUNCTION")
    logger.info("=" * 40)

    fqi = FittedQIteration(
        q_function=q_function,
        gamma=gamma,
        n_iterations=n_iterations,
        convergence_threshold=1e-4,
        patience=15
    )

    history = fqi.fit(train_data, val_data)

    # ==========================================================================
    # STEP 4: EVALUATE ON TEST SET
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 4: EVALUATING ON TEST SET")
    logger.info("=" * 40)

    # Get optimal actions on test set
    test_optimal_actions = q_function.get_greedy_actions(test_data['states'])

    # Agreement with clinician actions
    agreement = np.mean(test_optimal_actions == test_data['actions'])
    logger.info(f"Agreement with clinician actions: {agreement*100:.1f}%")

    # Q-value statistics
    test_q_all = q_function.predict_all_q(test_data['states'])
    logger.info(f"Test Q-values: {np.mean(test_q_all):.2f} ± {np.std(test_q_all):.2f}")

    # WDR evaluation
    wdr_results = evaluate_wdr(q_function, test_data, test_data['df'], gamma)

    # ==========================================================================
    # STEP 5: ANALYZE POLICY
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 5: POLICY ANALYSIS")
    logger.info("=" * 40)

    # Action distribution comparison
    train_actions = train_data['actions']
    train_optimal = q_function.get_greedy_actions(train_data['states'])

    logger.info("\nClinician vs. AI Policy Action Distribution:")
    logger.info("Action | Clinician | AI Policy")
    logger.info("-" * 40)

    for a in range(n_actions):
        clin_pct = np.mean(train_actions == a) * 100
        ai_pct = np.mean(train_optimal == a) * 100
        iv_bin = a // 5
        vaso_bin = a % 5
        logger.info(f"A{a:2d} (IV={iv_bin}, V={vaso_bin}) | {clin_pct:5.1f}% | {ai_pct:5.1f}%")

    # ==========================================================================
    # STEP 6: SAVE MODEL
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 6: SAVING MODEL")
    logger.info("=" * 40)

    # Save Q-function
    model_path = output_dir / 'q_function.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'q_function': q_function,
            'state_cols': state_cols,
            'n_actions': n_actions,
            'gamma': gamma,
            'history': history,
            'wdr_results': wdr_results
        }, f)
    logger.info(f"✓ Saved model to {model_path}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = output_dir / 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    logger.info(f"✓ Saved training history to {history_path}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nModel:")
    logger.info(f"  State features: {n_state_features}")
    logger.info(f"  Actions: {n_actions}")
    logger.info(f"  Total parameters: {q_function.n_features}")
    logger.info(f"\nPerformance:")
    logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"  Agreement with clinicians: {agreement*100:.1f}%")
    logger.info(f"  WDR estimate: {wdr_results['wdr_estimate']:.3f} ± {wdr_results['wdr_ci_95']:.3f}")
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info("=" * 80)

    return q_function, history, wdr_results


if __name__ == "__main__":
    main()
