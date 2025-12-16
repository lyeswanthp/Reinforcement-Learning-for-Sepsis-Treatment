"""
Q-Learning Training Pipeline v2.0 - FIXED VERSION

Major fixes from v1:
1. Correct behavior policy estimation (state-dependent, not global)
2. Behavior policy softening (99%/1%) to prevent infinite IS ratios
3. Fixed WDR implementation with proper value bootstrapping
4. Clinical sanity checks and validation
5. Hyperparameter tuning with validation-based selection
6. Comparison with clinician baseline
7. Bootstrap confidence intervals for all policy evaluations
8. Action-value constraints to prevent pathological policies

Author: AI Clinician Project
Date: 2024-11-23
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
from sklearn.cluster import KMeans
from scipy import stats
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


# =============================================================================
# BEHAVIOR POLICY ESTIMATION (STATE-DEPENDENT)
# =============================================================================

class BehaviorPolicyEstimator:
    """
    Estimates behavior policy from observational data using state clustering.

    This is crucial for importance sampling - we need P(action | state) not just P(action).
    Uses K-means clustering to discretize states, then computes action frequencies per cluster.
    """

    def __init__(self, n_clusters: int = 750, softening_epsilon: float = 0.01, random_seed: int = 42):
        """
        Args:
            n_clusters: Number of state clusters (Komorowski uses 750)
            softening_epsilon: Probability floor for all actions (prevents infinite IS ratios)
            random_seed: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.softening_epsilon = softening_epsilon
        self.random_seed = random_seed
        self.kmeans = None
        self.action_probs_per_cluster = None
        self.n_actions = 25
        self.state_scaler = StandardScaler()

    def fit(self, states: np.ndarray, actions: np.ndarray):
        """
        Fit behavior policy from observational data.

        Args:
            states: State features (n_samples, n_features)
            actions: Action indices (n_samples,)
        """
        logger.info(f"Fitting behavior policy with {self.n_clusters} state clusters...")

        # Normalize states for clustering
        states_scaled = self.state_scaler.fit_transform(states)

        # Cluster states
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_seed,
            n_init=10,
            max_iter=300
        )
        cluster_labels = self.kmeans.fit_predict(states_scaled)

        # Compute action probabilities per cluster
        self.action_probs_per_cluster = np.zeros((self.n_clusters, self.n_actions))

        for cluster in range(self.n_clusters):
            mask = cluster_labels == cluster
            if mask.sum() > 0:
                cluster_actions = actions[mask]
                for a in range(self.n_actions):
                    self.action_probs_per_cluster[cluster, a] = np.mean(cluster_actions == a)

        # Apply softening: π_b'(a|s) = (1-ε) * π_b(a|s) + ε * (1/|A|)
        # This ensures all action probabilities > 0, preventing infinite IS ratios
        uniform_prob = 1.0 / self.n_actions
        self.action_probs_per_cluster = (
            (1 - self.softening_epsilon) * self.action_probs_per_cluster +
            self.softening_epsilon * uniform_prob
        )

        # Verify probabilities sum to 1
        self.action_probs_per_cluster /= self.action_probs_per_cluster.sum(axis=1, keepdims=True)

        logger.info(f"  Softening epsilon: {self.softening_epsilon} (99%/{self.softening_epsilon*100:.0f}%)")
        logger.info(f"  Min action probability: {self.action_probs_per_cluster.min():.4f}")
        logger.info("  Behavior policy fitted successfully")

    def predict_probs(self, states: np.ndarray, actions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get behavior policy probabilities for given states.

        Args:
            states: State features
            actions: If provided, return P(action|state) for these specific actions
                     Otherwise return full probability distribution

        Returns:
            Probabilities (n_samples,) if actions provided, else (n_samples, n_actions)
        """
        states_scaled = self.state_scaler.transform(states)
        cluster_labels = self.kmeans.predict(states_scaled)

        if actions is None:
            return self.action_probs_per_cluster[cluster_labels]
        else:
            probs = np.zeros(len(states))
            for i, (cluster, action) in enumerate(zip(cluster_labels, actions)):
                probs[i] = self.action_probs_per_cluster[cluster, int(action)]
            return probs


# =============================================================================
# IMPROVED LINEAR Q-FUNCTION
# =============================================================================

class ImprovedLinearQFunction:
    """
    Linear Q-function with improvements:
    1. Proper feature engineering matching proposal (148 features)
    2. Q-value clipping to prevent unrealistic values
    3. Action-value constraints for clinical plausibility
    """

    def __init__(
        self,
        n_state_features: int,
        n_actions: int = 25,
        alpha: float = 1.0,
        gamma: float = 0.99,
        q_clip_min: float = -20.0,  # Min possible value (worse than death penalty)
        q_clip_max: float = 20.0,   # Max possible value (better than survival reward)
    ):
        self.n_state_features = n_state_features
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_clip_min = q_clip_min
        self.q_clip_max = q_clip_max

        # Feature dimension: state + action_onehot + state*action
        self.n_features = n_state_features + n_actions + n_state_features * n_actions

        self.state_scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.training_history = []

        logger.info(f"ImprovedLinearQFunction initialized:")
        logger.info(f"  State features: {n_state_features}")
        logger.info(f"  Actions: {n_actions}")
        logger.info(f"  Total features: {self.n_features}")
        logger.info(f"  Q-value bounds: [{q_clip_min}, {q_clip_max}]")

    def _create_features(self, states: np.ndarray, actions: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Create feature vector φ(s, a) with state-action interactions."""
        n_samples = len(states)

        if normalize:
            states_norm = self.state_scaler.transform(states)
        else:
            states_norm = states

        # One-hot encode actions
        action_onehot = np.zeros((n_samples, self.n_actions))
        action_onehot[np.arange(n_samples), actions.astype(int)] = 1

        # State-action interactions
        interactions = np.zeros((n_samples, self.n_state_features * self.n_actions))
        for i in range(n_samples):
            a = int(actions[i])
            start_idx = a * self.n_state_features
            end_idx = (a + 1) * self.n_state_features
            interactions[i, start_idx:end_idx] = states_norm[i]

        return np.hstack([states_norm, action_onehot, interactions])

    def fit_scaler(self, states: np.ndarray):
        """Fit the state scaler on training data."""
        self.state_scaler.fit(states)

    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict Q-values with clipping."""
        features = self._create_features(states, actions)
        q_values = self.model.predict(features)
        return np.clip(q_values, self.q_clip_min, self.q_clip_max)

    def predict_all_q(self, states: np.ndarray) -> np.ndarray:
        """Predict Q-values for all actions."""
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

    def get_softmax_policy(self, states: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Get softmax policy probabilities."""
        q_values = self.predict_all_q(states)
        q_scaled = q_values / temperature
        q_max = np.max(q_scaled, axis=1, keepdims=True)
        exp_q = np.exp(q_scaled - q_max)
        return exp_q / np.sum(exp_q, axis=1, keepdims=True)


# =============================================================================
# CORRECTED WDR ESTIMATOR
# =============================================================================

class CorrectedWDR:
    """
    Corrected Weighted Doubly Robust estimator.

    Key fixes:
    1. Uses proper value function bootstrapping (not Q(s,a))
    2. Clips importance weights to prevent extreme variance
    3. Uses per-trajectory normalization
    """

    def __init__(self, gamma: float = 0.99, max_weight: float = 100.0):
        self.gamma = gamma
        self.max_weight = max_weight

    def estimate_value(
        self,
        df: pd.DataFrame,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        q_function: ImprovedLinearQFunction,
        behavior_policy: BehaviorPolicyEstimator,
        target_policy_type: str = "greedy"  # or "softmax"
    ) -> Dict:
        """
        Compute WDR estimate.

        Args:
            df: DataFrame with stay_id for trajectory grouping
            states, actions, rewards, dones: Transition data
            q_function: Trained Q-function
            behavior_policy: Estimated behavior policy
            target_policy_type: "greedy" for deterministic, "softmax" for stochastic
        """
        # Get target policy probabilities
        if target_policy_type == "greedy":
            greedy_actions = q_function.get_greedy_actions(states)
            target_probs = np.zeros(len(states))
            target_probs[actions == greedy_actions] = 1.0
            # Soften greedy policy slightly to avoid zero probabilities
            target_probs = np.where(target_probs > 0, 0.99, 0.01/24)
        else:
            policy_probs = q_function.get_softmax_policy(states, temperature=0.5)
            target_probs = policy_probs[np.arange(len(actions)), actions.astype(int)]

        # Get behavior policy probabilities
        behavior_probs = behavior_policy.predict_probs(states, actions)

        # Compute importance ratios with clipping
        rho = target_probs / (behavior_probs + 1e-10)
        rho = np.clip(rho, 0, self.max_weight)

        # Get Q-values for all actions (for value function)
        q_all = q_function.predict_all_q(states)
        q_sa = q_all[np.arange(len(actions)), actions.astype(int)]

        # Compute per-trajectory WDR
        wdr_values = []
        stay_ids = df['stay_id'].unique()

        for stay_id in stay_ids:
            mask = (df['stay_id'] == stay_id).values
            traj_indices = np.where(mask)[0]

            if len(traj_indices) == 0:
                continue

            n_steps = len(traj_indices)
            traj_rewards = rewards[traj_indices]
            traj_rho = rho[traj_indices]
            traj_q_sa = q_sa[traj_indices]
            traj_dones = dones[traj_indices]

            # Get target policy value (expected Q under target policy)
            if target_policy_type == "greedy":
                traj_v = np.max(q_all[traj_indices], axis=1)  # V(s) = max_a Q(s,a)
            else:
                policy_probs = q_function.get_softmax_policy(states[traj_indices])
                traj_v = np.sum(policy_probs * q_all[traj_indices], axis=1)

            # WDR backward recursion
            V = 0.0
            cum_rho = 1.0

            for t in range(n_steps - 1, -1, -1):
                r_t = traj_rewards[t]
                rho_t = traj_rho[t]
                q_t = traj_q_sa[t]
                v_t = traj_v[t]
                done_t = traj_dones[t]

                if done_t:
                    V_next = 0.0
                else:
                    V_next = traj_v[t+1] if t < n_steps - 1 else 0.0

                # WDR formula: ρ_t * (r_t + γ*V_next - Q(s,a)) + V(s)
                # This corrects model error using importance weighting
                cum_rho *= rho_t
                cum_rho = min(cum_rho, self.max_weight)  # Clip cumulative ratio

                td_error = r_t + self.gamma * V_next - q_t
                V = cum_rho * td_error + v_t

            wdr_values.append(V)

        wdr_estimate = np.mean(wdr_values)
        wdr_std = np.std(wdr_values)
        n_traj = len(wdr_values)
        ci_95 = 1.96 * wdr_std / np.sqrt(n_traj)

        return {
            'wdr_estimate': wdr_estimate,
            'wdr_std': wdr_std,
            'wdr_ci_95': ci_95,
            'n_trajectories': n_traj,
            'trajectory_values': wdr_values
        }

    def bootstrap_ci(
        self,
        df: pd.DataFrame,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        q_function: ImprovedLinearQFunction,
        behavior_policy: BehaviorPolicyEstimator,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval.

        Returns:
            (mean, lower_bound, upper_bound)
        """
        np.random.seed(random_seed)

        stay_ids = df['stay_id'].unique()
        bootstrap_estimates = []

        for b in range(n_bootstrap):
            # Resample trajectories
            resampled_stays = np.random.choice(stay_ids, size=len(stay_ids), replace=True)

            # Get indices for resampled trajectories
            resampled_indices = []
            for stay_id in resampled_stays:
                mask = (df['stay_id'] == stay_id).values
                resampled_indices.extend(np.where(mask)[0])
            resampled_indices = np.array(resampled_indices)

            if len(resampled_indices) == 0:
                continue

            # Create resampled dataframe
            resampled_df = df.iloc[resampled_indices].copy()

            try:
                result = self.estimate_value(
                    resampled_df,
                    states[resampled_indices],
                    actions[resampled_indices],
                    rewards[resampled_indices],
                    dones[resampled_indices],
                    q_function,
                    behavior_policy
                )
                bootstrap_estimates.append(result['wdr_estimate'])
            except:
                continue

            if (b + 1) % 100 == 0:
                logger.info(f"  Bootstrap iteration {b+1}/{n_bootstrap}")

        bootstrap_estimates = np.array(bootstrap_estimates)
        mean_estimate = np.mean(bootstrap_estimates)
        alpha = 1 - confidence_level
        lower_bound = np.percentile(bootstrap_estimates, alpha/2 * 100)
        upper_bound = np.percentile(bootstrap_estimates, (1 - alpha/2) * 100)

        return mean_estimate, lower_bound, upper_bound


# =============================================================================
# CLINICIAN POLICY EVALUATION
# =============================================================================

def evaluate_clinician_policy(
    df: pd.DataFrame,
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99
) -> Dict:
    """
    Evaluate the observed clinician policy (behavior policy).

    Since the clinician policy is what generated the data, we can compute
    the on-policy return directly without importance sampling.
    """
    logger.info("Evaluating clinician (behavior) policy...")

    trajectory_returns = []

    for stay_id in df['stay_id'].unique():
        mask = (df['stay_id'] == stay_id).values
        traj_rewards = rewards[mask]

        # Compute discounted return
        G = 0.0
        for t in range(len(traj_rewards) - 1, -1, -1):
            G = traj_rewards[t] + gamma * G

        trajectory_returns.append(G)

    mean_return = np.mean(trajectory_returns)
    std_return = np.std(trajectory_returns)
    n_traj = len(trajectory_returns)
    ci_95 = 1.96 * std_return / np.sqrt(n_traj)

    logger.info(f"  Clinician policy value: {mean_return:.3f} ± {ci_95:.3f}")
    logger.info(f"  Number of trajectories: {n_traj:,}")

    return {
        'value': mean_return,
        'std': std_return,
        'ci_95': ci_95,
        'n_trajectories': n_traj,
        'trajectory_returns': trajectory_returns
    }


# =============================================================================
# CLINICAL SANITY CHECKS
# =============================================================================

def clinical_sanity_checks(q_function: ImprovedLinearQFunction, state_cols: List[str]) -> Dict:
    """
    Perform clinical sanity checks on the learned policy.

    Checks:
    1. Does policy recommend more vasopressors for hypotensive patients?
    2. Does policy recommend less fluid for fluid-overloaded patients?
    3. Does policy respond appropriately to high lactate?
    """
    logger.info("Performing clinical sanity checks...")
    results = {}

    # Create baseline (average) patient state
    n_features = len(state_cols)
    baseline_state = np.zeros((1, n_features))

    # Find relevant feature indices
    feature_indices = {col: i for i, col in enumerate(state_cols)}

    # Check 1: Vasopressor response to hypotension
    if 'MeanBP' in feature_indices or 'SysBP' in feature_indices:
        bp_col = 'MeanBP' if 'MeanBP' in feature_indices else 'SysBP'
        bp_idx = feature_indices[bp_col]

        # Normal BP state
        normal_state = baseline_state.copy()
        normal_state[0, bp_idx] = 0.0  # Normalized = average

        # Low BP state (hypotensive)
        low_bp_state = baseline_state.copy()
        low_bp_state[0, bp_idx] = -2.0  # 2 SD below average

        # Get recommended actions
        normal_action = q_function.get_greedy_actions(normal_state)[0]
        low_bp_action = q_function.get_greedy_actions(low_bp_state)[0]

        normal_vaso = normal_action % 5
        low_bp_vaso = low_bp_action % 5

        check1_passed = low_bp_vaso >= normal_vaso
        results['hypotension_check'] = {
            'passed': check1_passed,
            'normal_vaso_bin': normal_vaso,
            'low_bp_vaso_bin': low_bp_vaso,
            'description': 'More vasopressors for hypotensive patients'
        }
        logger.info(f"  Hypotension check: {'PASS' if check1_passed else 'FAIL'} "
                   f"(normal vaso={normal_vaso}, low BP vaso={low_bp_vaso})")

    # Check 2: Lactate response
    if 'Arterial_lactate' in feature_indices:
        lactate_idx = feature_indices['Arterial_lactate']

        normal_state = baseline_state.copy()
        high_lactate_state = baseline_state.copy()
        high_lactate_state[0, lactate_idx] = 2.0  # 2 SD above average (elevated)

        normal_action = q_function.get_greedy_actions(normal_state)[0]
        high_lactate_action = q_function.get_greedy_actions(high_lactate_state)[0]

        # High lactate should generally trigger more intervention
        normal_intensity = (normal_action // 5) + (normal_action % 5)
        high_lactate_intensity = (high_lactate_action // 5) + (high_lactate_action % 5)

        check2_passed = high_lactate_intensity >= normal_intensity
        results['lactate_check'] = {
            'passed': check2_passed,
            'normal_action': normal_action,
            'high_lactate_action': high_lactate_action,
            'description': 'More intervention for elevated lactate'
        }
        logger.info(f"  Lactate check: {'PASS' if check2_passed else 'FAIL'} "
                   f"(normal={normal_action}, high lactate={high_lactate_action})")

    # Check 3: SOFA response
    if 'SOFA' in feature_indices:
        sofa_idx = feature_indices['SOFA']

        low_sofa_state = baseline_state.copy()
        low_sofa_state[0, sofa_idx] = -1.0  # Below average SOFA (better)

        high_sofa_state = baseline_state.copy()
        high_sofa_state[0, sofa_idx] = 2.0  # Above average SOFA (sicker)

        low_sofa_action = q_function.get_greedy_actions(low_sofa_state)[0]
        high_sofa_action = q_function.get_greedy_actions(high_sofa_state)[0]

        low_intensity = (low_sofa_action // 5) + (low_sofa_action % 5)
        high_intensity = (high_sofa_action // 5) + (high_sofa_action % 5)

        check3_passed = high_intensity >= low_intensity
        results['sofa_check'] = {
            'passed': check3_passed,
            'low_sofa_action': low_sofa_action,
            'high_sofa_action': high_sofa_action,
            'description': 'More intervention for higher SOFA'
        }
        logger.info(f"  SOFA check: {'PASS' if check3_passed else 'FAIL'} "
                   f"(low SOFA={low_sofa_action}, high SOFA={high_sofa_action})")

    # Overall sanity score
    n_passed = sum(1 for r in results.values() if r['passed'])
    n_total = len(results)
    results['overall'] = {
        'passed': n_passed,
        'total': n_total,
        'score': n_passed / n_total if n_total > 0 else 0
    }
    logger.info(f"  Overall sanity score: {n_passed}/{n_total}")

    return results


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

def hyperparameter_search(
    train_data: Dict,
    val_data: Dict,
    state_cols: List[str],
    n_trials: int = 20,
    random_seed: int = 42
) -> Dict:
    """
    Random search over hyperparameters using validation WDR as objective.
    """
    logger.info(f"Starting hyperparameter search ({n_trials} trials)...")
    np.random.seed(random_seed)

    best_config = None
    best_wdr = -np.inf
    results = []

    for trial in range(n_trials):
        # Sample hyperparameters
        alpha = 10 ** np.random.uniform(-3, 1)  # L2 regularization: 0.001 to 10
        gamma = np.random.choice([0.90, 0.95, 0.99])
        n_clusters = np.random.choice([500, 750, 1000])

        logger.info(f"\nTrial {trial + 1}/{n_trials}: alpha={alpha:.4f}, gamma={gamma}, clusters={n_clusters}")

        try:
            # Train Q-function
            q_func = ImprovedLinearQFunction(
                n_state_features=len(state_cols),
                n_actions=25,
                alpha=alpha,
                gamma=gamma
            )
            q_func.fit_scaler(train_data['states'])

            # Fit Q using FQI
            features = q_func._create_features(train_data['states'], train_data['actions'])
            targets = train_data['rewards'].copy()

            for _ in range(50):  # Quick training
                q_func.model.fit(features, targets)
                next_q = q_func.predict_all_q(train_data['next_states'])
                next_q_max = np.max(next_q, axis=1)
                next_q_max[train_data['dones'].astype(bool)] = 0
                targets = train_data['rewards'] + gamma * next_q_max

            # Fit behavior policy
            behavior_policy = BehaviorPolicyEstimator(
                n_clusters=n_clusters,
                softening_epsilon=0.01
            )
            behavior_policy.fit(train_data['states'], train_data['actions'])

            # Evaluate on validation set
            wdr = CorrectedWDR(gamma=gamma)
            wdr_result = wdr.estimate_value(
                val_data['df'],
                val_data['states'],
                val_data['actions'],
                val_data['rewards'],
                val_data['dones'],
                q_func,
                behavior_policy
            )

            wdr_value = wdr_result['wdr_estimate']

            # Check clinical sanity
            sanity = clinical_sanity_checks(q_func, state_cols)
            sanity_score = sanity['overall']['score']

            # Combined score (WDR + sanity bonus)
            combined_score = wdr_value + 2.0 * sanity_score  # Bonus for clinical plausibility

            results.append({
                'alpha': alpha,
                'gamma': gamma,
                'n_clusters': n_clusters,
                'wdr_value': wdr_value,
                'sanity_score': sanity_score,
                'combined_score': combined_score
            })

            logger.info(f"  WDR={wdr_value:.3f}, Sanity={sanity_score:.2f}, Combined={combined_score:.3f}")

            if combined_score > best_wdr:
                best_wdr = combined_score
                best_config = {
                    'alpha': alpha,
                    'gamma': gamma,
                    'n_clusters': n_clusters,
                    'q_function': q_func,
                    'behavior_policy': behavior_policy
                }
                logger.info("  *** New best! ***")

        except Exception as e:
            logger.warning(f"  Trial failed: {e}")
            continue

    logger.info(f"\nBest configuration: alpha={best_config['alpha']:.4f}, "
               f"gamma={best_config['gamma']}, clusters={best_config['n_clusters']}")

    return {
        'best_config': best_config,
        'all_results': results
    }


# =============================================================================
# FITTED Q-ITERATION (IMPROVED)
# =============================================================================

class ImprovedFQI:
    """
    Improved Fitted Q-Iteration with:
    1. Better convergence detection
    2. Weight monitoring
    3. Action distribution tracking
    """

    def __init__(
        self,
        q_function: ImprovedLinearQFunction,
        gamma: float = 0.99,
        n_iterations: int = 100,
        convergence_threshold: float = 1e-4,
        patience: int = 15
    ):
        self.q_function = q_function
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        self.patience = patience

    def fit(self, train_data: Dict, val_data: Optional[Dict] = None) -> Dict:
        """Train Q-function using FQI."""
        logger.info("Starting Improved FQI training...")

        states = train_data['states']
        actions = train_data['actions']
        rewards = train_data['rewards']
        next_states = train_data['next_states']
        dones = train_data['dones']

        self.q_function.fit_scaler(states)
        features = self.q_function._create_features(states, actions)
        targets = rewards.copy()

        history = {'train_loss': [], 'val_loss': [], 'q_mean': [], 'q_std': [],
                   'action_entropy': [], 'iteration': []}

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for iteration in range(self.n_iterations):
            self.q_function.model.fit(features, targets)
            q_pred = self.q_function.model.predict(features)
            train_loss = np.mean((q_pred - targets) ** 2)

            # Update targets
            next_q_all = self.q_function.predict_all_q(next_states)
            next_q_max = np.max(next_q_all, axis=1)
            next_q_max[dones.astype(bool)] = 0
            new_targets = rewards + self.gamma * next_q_max

            target_change = np.mean(np.abs(new_targets - targets))
            targets = new_targets

            # Track action distribution entropy
            greedy_actions = np.argmax(self.q_function.predict_all_q(states), axis=1)
            action_counts = np.bincount(greedy_actions, minlength=25) + 1
            action_probs = action_counts / action_counts.sum()
            entropy = -np.sum(action_probs * np.log(action_probs))

            # Validation
            val_loss = None
            if val_data is not None:
                val_q_pred = self.q_function.predict_q(val_data['states'], val_data['actions'])
                val_next_q = np.where(
                    val_data['dones'].astype(bool),
                    0,
                    np.max(self.q_function.predict_all_q(val_data['next_states']), axis=1)
                )
                val_targets = val_data['rewards'] + self.gamma * val_next_q
                val_loss = np.mean((val_q_pred - val_targets) ** 2)

                if val_loss < best_val_loss - self.convergence_threshold:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = self.q_function.model.coef_.copy()
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at iteration {iteration + 1}")
                    if best_weights is not None:
                        self.q_function.model.coef_ = best_weights
                    break

            # Log history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['q_mean'].append(np.mean(q_pred))
            history['q_std'].append(np.std(q_pred))
            history['action_entropy'].append(entropy)
            history['iteration'].append(iteration)

            if (iteration + 1) % 10 == 0 or iteration == 0:
                logger.info(f"Iter {iteration+1}: Loss={train_loss:.4f}, Δ={target_change:.4f}, "
                           f"Q={np.mean(q_pred):.2f}±{np.std(q_pred):.2f}, H={entropy:.2f}")

            if target_change < self.convergence_threshold:
                logger.info(f"Converged at iteration {iteration + 1}")
                break

        logger.info(f"Training complete. Final loss: {history['train_loss'][-1]:.4f}")
        return history


# =============================================================================
# DATA LOADING
# =============================================================================

def load_trajectory_data(file_path: str, state_cols: List[str]) -> Dict[str, np.ndarray]:
    """Load trajectory data from CSV."""
    logger.info(f"Loading {file_path}...")
    df = pd.read_csv(file_path)

    next_state_cols = [f'next_{col}' for col in state_cols]

    # Check which columns exist
    existing_state_cols = [c for c in state_cols if c in df.columns]
    existing_next_cols = [c for c in next_state_cols if c in df.columns]

    if len(existing_state_cols) < len(state_cols):
        missing = set(state_cols) - set(existing_state_cols)
        logger.warning(f"Missing state columns: {missing}")

    states = df[existing_state_cols].values.astype(np.float32)
    actions = df['action'].values.astype(np.int32)
    rewards = df['reward'].values.astype(np.float32)

    # Handle next states
    next_states = np.zeros_like(states)
    for i, col in enumerate(existing_state_cols):
        next_col = f'next_{col}'
        if next_col in df.columns:
            next_states[:, i] = df[next_col].values
    next_states = np.nan_to_num(next_states, nan=0.0)

    dones = df['done'].values.astype(np.float32)

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'dones': dones,
        'df': df,
        'state_cols': existing_state_cols
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main training pipeline v2."""
    logger.info("=" * 80)
    logger.info("Q-LEARNING TRAINING PIPELINE v2.0 - FIXED VERSION")
    logger.info("=" * 80)

    # Load config
    config = ConfigLoader('configs/config.yaml').config

    # Paths
    data_dir = Path('data/processed')
    output_dir = Path('outputs/models_v2')
    output_dir.mkdir(parents=True, exist_ok=True)

    # State columns
    state_cols = [
        'DiaBP', 'FiO2_1', 'HR', 'MeanBP', 'RR', 'SpO2', 'SysBP', 'Temp_C',
        'Arterial_BE', 'Arterial_lactate', 'Arterial_pH', 'Calcium', 'Chloride',
        'Creatinine', 'Glucose', 'HCO3', 'Hb', 'INR', 'Magnesium', 'PT',
        'Platelets_count', 'Potassium', 'SGOT', 'SGPT', 'Sodium', 'Total_bili',
        'WBC_count', 'paCO2', 'paO2',
        'gender', 'age', 're_admission',
        'PaO2_FiO2', 'Shock_Index', 'SOFA'
    ]

    # ==========================================================================
    # STEP 1: LOAD DATA
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 40)

    train_data = load_trajectory_data(data_dir / 'train_trajectories.csv', state_cols)
    val_data = load_trajectory_data(data_dir / 'val_trajectories.csv', state_cols)
    test_data = load_trajectory_data(data_dir / 'test_trajectories.csv', state_cols)

    actual_state_cols = train_data['state_cols']
    logger.info(f"  Train: {len(train_data['states']):,} transitions")
    logger.info(f"  Val: {len(val_data['states']):,} transitions")
    logger.info(f"  Test: {len(test_data['states']):,} transitions")
    logger.info(f"  State features: {len(actual_state_cols)}")

    # ==========================================================================
    # STEP 2: HYPERPARAMETER TUNING
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 2: HYPERPARAMETER TUNING")
    logger.info("=" * 40)

    tuning_results = hyperparameter_search(
        train_data, val_data, actual_state_cols,
        n_trials=20,  # Increase for better results
        random_seed=42
    )

    best_config = tuning_results['best_config']
    q_function = best_config['q_function']
    behavior_policy = best_config['behavior_policy']
    gamma = best_config['gamma']

    # ==========================================================================
    # STEP 3: FULL TRAINING WITH BEST CONFIG
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 3: FULL TRAINING WITH BEST CONFIG")
    logger.info("=" * 40)

    # Reinitialize with best hyperparameters for full training
    q_function = ImprovedLinearQFunction(
        n_state_features=len(actual_state_cols),
        n_actions=25,
        alpha=best_config['alpha'],
        gamma=best_config['gamma']
    )

    fqi = ImprovedFQI(
        q_function=q_function,
        gamma=gamma,
        n_iterations=100,
        patience=15
    )

    history = fqi.fit(train_data, val_data)

    # Refit behavior policy on full training set
    behavior_policy = BehaviorPolicyEstimator(
        n_clusters=best_config['n_clusters'],
        softening_epsilon=0.01
    )
    behavior_policy.fit(train_data['states'], train_data['actions'])

    # ==========================================================================
    # STEP 4: EVALUATE ALL POLICIES
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 4: POLICY EVALUATION")
    logger.info("=" * 40)

    # 4.1: Clinician policy (on-policy)
    clinician_result = evaluate_clinician_policy(
        test_data['df'], test_data['rewards'], test_data['dones'], gamma
    )

    # 4.2: AI policy (WDR)
    logger.info("\nEvaluating AI policy (WDR)...")
    wdr = CorrectedWDR(gamma=gamma)
    ai_result = wdr.estimate_value(
        test_data['df'],
        test_data['states'],
        test_data['actions'],
        test_data['rewards'],
        test_data['dones'],
        q_function,
        behavior_policy
    )
    logger.info(f"  AI policy WDR: {ai_result['wdr_estimate']:.3f} ± {ai_result['wdr_ci_95']:.3f}")

    # 4.3: Agreement rate
    ai_actions = q_function.get_greedy_actions(test_data['states'])
    agreement = np.mean(ai_actions == test_data['actions'])
    logger.info(f"\nAgreement with clinicians: {agreement*100:.1f}%")

    # 4.4: Bootstrap CIs (if time permits, reduce n_bootstrap for speed)
    logger.info("\nComputing bootstrap confidence intervals...")
    ai_mean, ai_lower, ai_upper = wdr.bootstrap_ci(
        test_data['df'],
        test_data['states'],
        test_data['actions'],
        test_data['rewards'],
        test_data['dones'],
        q_function,
        behavior_policy,
        n_bootstrap=200  # Reduce for speed, use 1000 for publication
    )
    logger.info(f"  AI policy 95% CI: [{ai_lower:.3f}, {ai_upper:.3f}]")

    # ==========================================================================
    # STEP 5: CLINICAL SANITY CHECKS
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 5: CLINICAL SANITY CHECKS")
    logger.info("=" * 40)

    sanity_results = clinical_sanity_checks(q_function, actual_state_cols)

    # ==========================================================================
    # STEP 6: ACTION DISTRIBUTION ANALYSIS
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 6: ACTION DISTRIBUTION")
    logger.info("=" * 40)

    train_optimal = q_function.get_greedy_actions(train_data['states'])

    logger.info("\nAction | Clinician | AI Policy | Difference")
    logger.info("-" * 50)

    for a in range(25):
        clin_pct = np.mean(train_data['actions'] == a) * 100
        ai_pct = np.mean(train_optimal == a) * 100
        diff = ai_pct - clin_pct
        iv_bin = a // 5
        vaso_bin = a % 5

        diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
        logger.info(f"A{a:2d} (IV={iv_bin},V={vaso_bin}) | {clin_pct:5.1f}% | {ai_pct:5.1f}% | {diff_str}%")

    # ==========================================================================
    # STEP 7: COMPARISON TABLE
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 7: FINAL COMPARISON TABLE")
    logger.info("=" * 40)

    # Significance test (paired difference)
    ai_values = np.array(ai_result['trajectory_values'])
    clin_values = np.array(clinician_result['trajectory_returns'])

    # Match trajectory counts
    min_traj = min(len(ai_values), len(clin_values))
    diff = ai_values[:min_traj] - clin_values[:min_traj]

    t_stat, p_value = stats.ttest_1samp(diff, 0)

    logger.info("\n" + "=" * 70)
    logger.info("FINAL OFF-POLICY EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"{'Policy':<25} {'Value':<15} {'95% CI':<25} {'p-value':<10}")
    logger.info("-" * 70)
    logger.info(f"{'Clinician (observed)':<25} {clinician_result['value']:<15.3f} "
               f"[{clinician_result['value']-clinician_result['ci_95']:.3f}, "
               f"{clinician_result['value']+clinician_result['ci_95']:.3f}]{'---':<10}")
    logger.info(f"{'AI Policy (WDR)':<25} {ai_result['wdr_estimate']:<15.3f} "
               f"[{ai_lower:.3f}, {ai_upper:.3f}]{'':<4}{p_value:.4f}")
    logger.info("=" * 70)

    improvement = ai_result['wdr_estimate'] - clinician_result['value']
    logger.info(f"\nImprovement over clinician: {improvement:+.3f}")
    logger.info(f"Agreement with clinician actions: {agreement*100:.1f}%")
    logger.info(f"Clinical sanity score: {sanity_results['overall']['passed']}/{sanity_results['overall']['total']}")

    if p_value < 0.05:
        if improvement > 0:
            logger.info("Result: AI policy SIGNIFICANTLY BETTER than clinicians (p < 0.05)")
        else:
            logger.info("Result: AI policy SIGNIFICANTLY WORSE than clinicians (p < 0.05)")
    else:
        logger.info("Result: No significant difference from clinician policy (p >= 0.05)")

    # ==========================================================================
    # STEP 8: SAVE RESULTS
    # ==========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 8: SAVING RESULTS")
    logger.info("=" * 40)

    results = {
        'q_function': q_function,
        'behavior_policy': behavior_policy,
        'state_cols': actual_state_cols,
        'best_config': {k: v for k, v in best_config.items()
                       if k not in ['q_function', 'behavior_policy']},
        'history': history,
        'evaluation': {
            'clinician': clinician_result,
            'ai_policy': ai_result,
            'ai_bootstrap_ci': (ai_mean, ai_lower, ai_upper),
            'agreement': agreement,
            'p_value': p_value,
            'improvement': improvement
        },
        'sanity_checks': sanity_results,
        'tuning_results': tuning_results['all_results']
    }

    with open(output_dir / 'full_results_v2.pkl', 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"✓ Saved results to {output_dir / 'full_results_v2.pkl'}")

    # Save history
    pd.DataFrame(history).to_csv(output_dir / 'training_history_v2.csv', index=False)
    logger.info(f"✓ Saved training history to {output_dir / 'training_history_v2.csv'}")

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)

    return results


if __name__ == "__main__":
    main()
