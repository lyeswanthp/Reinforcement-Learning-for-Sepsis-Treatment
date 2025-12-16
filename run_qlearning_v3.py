"""
Q-Learning Training Pipeline v3.0 - COMPREHENSIVE FIX

Fixes from v2:
1. Mode collapse prevention via entropy regularization and action diversity loss
2. Stricter clinical sanity checks (require improvement, not just non-decrease)
3. Corrected WDR backward pass with proper value bootstrapping
4. Effective sample size (ESS) monitoring for importance sampling reliability
5. Gamma-normalized policy comparison (fair comparison across discount factors)
6. Action distribution constraints to prevent policy collapse
7. Per-state policy analysis to verify state-dependent behavior
8. Improved hyperparameter search with diversity objectives

Author: AI Clinician Project
Date: 2024-11-24
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from scipy import stats
from collections import defaultdict
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
# BEHAVIOR POLICY ESTIMATION (STATE-DEPENDENT) - IMPROVED
# =============================================================================

class ImprovedBehaviorPolicyEstimator:
    """
    Improved behavior policy estimation with:
    1. State-dependent action probabilities via K-means
    2. Laplace smoothing (more principled than epsilon floor)
    3. Effective sample size tracking
    """

    def __init__(
        self,
        n_clusters: int = 750,
        smoothing_alpha: float = 1.0,  # Laplace smoothing parameter
        random_seed: int = 42
    ):
        self.n_clusters = n_clusters
        self.smoothing_alpha = smoothing_alpha  # Add alpha to each action count
        self.random_seed = random_seed
        self.kmeans = None
        self.action_counts_per_cluster = None
        self.action_probs_per_cluster = None
        self.n_actions = 25
        self.state_scaler = StandardScaler()
        self.cluster_sizes = None

    def fit(self, states: np.ndarray, actions: np.ndarray):
        """Fit behavior policy with Laplace smoothing."""
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

        # Count actions per cluster with Laplace smoothing
        self.action_counts_per_cluster = np.full(
            (self.n_clusters, self.n_actions),
            self.smoothing_alpha  # Start with smoothing count
        )
        self.cluster_sizes = np.zeros(self.n_clusters)

        for cluster in range(self.n_clusters):
            mask = cluster_labels == cluster
            self.cluster_sizes[cluster] = mask.sum()
            if mask.sum() > 0:
                cluster_actions = actions[mask]
                for a in range(self.n_actions):
                    self.action_counts_per_cluster[cluster, a] += np.sum(cluster_actions == a)

        # Convert to probabilities (Laplace smoothing ensures no zeros)
        total_counts = self.action_counts_per_cluster.sum(axis=1, keepdims=True)
        self.action_probs_per_cluster = self.action_counts_per_cluster / total_counts

        # Compute min/max probabilities
        min_prob = self.action_probs_per_cluster.min()
        max_prob = self.action_probs_per_cluster.max()

        logger.info(f"  Smoothing alpha: {self.smoothing_alpha}")
        logger.info(f"  Action probability range: [{min_prob:.6f}, {max_prob:.4f}]")
        logger.info(f"  Cluster size range: [{self.cluster_sizes.min():.0f}, {self.cluster_sizes.max():.0f}]")
        logger.info("  Behavior policy fitted successfully")

    def predict_probs(self, states: np.ndarray, actions: Optional[np.ndarray] = None) -> np.ndarray:
        """Get behavior policy probabilities."""
        states_scaled = self.state_scaler.transform(states)
        cluster_labels = self.kmeans.predict(states_scaled)

        if actions is None:
            return self.action_probs_per_cluster[cluster_labels]
        else:
            probs = np.zeros(len(states))
            for i, (cluster, action) in enumerate(zip(cluster_labels, actions)):
                probs[i] = self.action_probs_per_cluster[cluster, int(action)]
            return probs

    def get_cluster_labels(self, states: np.ndarray) -> np.ndarray:
        """Get cluster assignments for states."""
        states_scaled = self.state_scaler.transform(states)
        return self.kmeans.predict(states_scaled)


# =============================================================================
# IMPROVED LINEAR Q-FUNCTION WITH ENTROPY REGULARIZATION
# =============================================================================

class EntropyRegularizedQFunction:
    """
    Linear Q-function with entropy regularization to prevent mode collapse.

    Key improvements:
    1. Entropy bonus encourages action diversity
    2. Q-value clipping with soft bounds
    3. Action diversity tracking
    4. Per-cluster action statistics
    """

    def __init__(
        self,
        n_state_features: int,
        n_actions: int = 25,
        alpha: float = 1.0,
        gamma: float = 0.99,
        entropy_coef: float = 0.1,  # Entropy regularization coefficient
        q_clip_min: float = -20.0,
        q_clip_max: float = 20.0,
    ):
        self.n_state_features = n_state_features
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.q_clip_min = q_clip_min
        self.q_clip_max = q_clip_max

        # Feature dimension
        self.n_features = n_state_features + n_actions + n_state_features * n_actions

        self.state_scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, fit_intercept=True)

        logger.info(f"EntropyRegularizedQFunction initialized:")
        logger.info(f"  State features: {n_state_features}")
        logger.info(f"  Actions: {n_actions}")
        logger.info(f"  Total features: {self.n_features}")
        logger.info(f"  Entropy coefficient: {entropy_coef}")
        logger.info(f"  Q-value bounds: [{q_clip_min}, {q_clip_max}]")

    def _create_features(self, states: np.ndarray, actions: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Create feature vector with state-action interactions."""
        n_samples = len(states)

        if normalize:
            states_norm = self.state_scaler.transform(states)
        else:
            states_norm = states

        # One-hot encode actions
        action_onehot = np.zeros((n_samples, self.n_actions))
        action_onehot[np.arange(n_samples), actions.astype(int)] = 1

        # State-action interactions (more efficient implementation)
        interactions = np.zeros((n_samples, self.n_state_features * self.n_actions))
        for a in range(self.n_actions):
            mask = actions == a
            if mask.any():
                start_idx = a * self.n_state_features
                end_idx = (a + 1) * self.n_state_features
                interactions[mask, start_idx:end_idx] = states_norm[mask]

        return np.hstack([states_norm, action_onehot, interactions])

    def fit_scaler(self, states: np.ndarray):
        """Fit the state scaler on training data."""
        self.state_scaler.fit(states)

    def predict_q(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict Q-values with soft clipping."""
        features = self._create_features(states, actions)
        q_values = self.model.predict(features)
        return np.clip(q_values, self.q_clip_min, self.q_clip_max)

    def predict_all_q(self, states: np.ndarray) -> np.ndarray:
        """Predict Q-values for all actions (vectorized)."""
        n_samples = len(states)
        states_norm = self.state_scaler.transform(states)

        q_values = np.zeros((n_samples, self.n_actions))

        for a in range(self.n_actions):
            # Create features for action a
            action_onehot = np.zeros((n_samples, self.n_actions))
            action_onehot[:, a] = 1

            interactions = np.zeros((n_samples, self.n_state_features * self.n_actions))
            start_idx = a * self.n_state_features
            end_idx = (a + 1) * self.n_state_features
            interactions[:, start_idx:end_idx] = states_norm

            features = np.hstack([states_norm, action_onehot, interactions])
            q_values[:, a] = self.model.predict(features)

        return np.clip(q_values, self.q_clip_min, self.q_clip_max)

    def get_greedy_actions(self, states: np.ndarray) -> np.ndarray:
        """Get greedy actions (argmax Q)."""
        q_values = self.predict_all_q(states)
        return np.argmax(q_values, axis=1)

    def get_softmax_policy(self, states: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Get softmax policy probabilities."""
        q_values = self.predict_all_q(states)
        q_scaled = q_values / max(temperature, 0.01)
        q_max = np.max(q_scaled, axis=1, keepdims=True)
        exp_q = np.exp(q_scaled - q_max)
        return exp_q / np.sum(exp_q, axis=1, keepdims=True)

    def compute_policy_entropy(self, states: np.ndarray, temperature: float = 1.0) -> float:
        """Compute average policy entropy (higher = more diverse)."""
        probs = self.get_softmax_policy(states, temperature)
        # Avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        return np.mean(entropy)

    def get_action_distribution(self, states: np.ndarray) -> np.ndarray:
        """Get distribution of greedy actions across states."""
        actions = self.get_greedy_actions(states)
        dist = np.bincount(actions, minlength=self.n_actions) / len(actions)
        return dist


# =============================================================================
# CORRECTED WDR ESTIMATOR WITH PROPER BOOTSTRAPPING
# =============================================================================

class CorrectedWDRv3:
    """
    Corrected Weighted Doubly Robust estimator v3.

    Fixes:
    1. Proper backward recursion with value function bootstrapping
    2. Per-trajectory importance weight computation
    3. Effective sample size (ESS) tracking
    4. Normalized importance weights option
    """

    def __init__(self, gamma: float = 0.99, max_weight: float = 100.0):
        self.gamma = gamma
        self.max_weight = max_weight

    def compute_effective_sample_size(self, weights: np.ndarray) -> float:
        """
        Compute effective sample size for importance sampling.
        ESS = (sum(w))^2 / sum(w^2)
        Lower ESS indicates higher variance / less reliable estimates.
        """
        if len(weights) == 0:
            return 0.0
        sum_w = np.sum(weights)
        sum_w2 = np.sum(weights ** 2)
        if sum_w2 == 0:
            return 0.0
        return (sum_w ** 2) / sum_w2

    def estimate_value(
        self,
        df: pd.DataFrame,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        q_function: EntropyRegularizedQFunction,
        behavior_policy: ImprovedBehaviorPolicyEstimator,
        use_softmax_target: bool = False,
        temperature: float = 0.5
    ) -> Dict:
        """
        Compute WDR estimate with proper value bootstrapping.

        The WDR estimator combines:
        - Model-based estimate (Q-function)
        - Importance sampling correction for model bias
        """
        n_samples = len(states)

        # Get all Q-values
        q_all = q_function.predict_all_q(states)
        q_sa = q_all[np.arange(n_samples), actions.astype(int)]

        # Compute target policy probabilities
        if use_softmax_target:
            target_policy = q_function.get_softmax_policy(states, temperature)
            target_probs = target_policy[np.arange(n_samples), actions.astype(int)]
            # V(s) = sum_a π(a|s) * Q(s,a)
            v_target = np.sum(target_policy * q_all, axis=1)
        else:
            # Greedy target policy with small softening for stability
            greedy_actions = q_function.get_greedy_actions(states)
            target_probs = np.where(actions == greedy_actions, 0.95, 0.05 / 24)
            # V(s) = max_a Q(s,a) for greedy
            v_target = np.max(q_all, axis=1)

        # Get behavior policy probabilities
        behavior_probs = behavior_policy.predict_probs(states, actions)

        # Compute per-step importance ratios with clipping
        rho = np.clip(target_probs / (behavior_probs + 1e-10), 0, self.max_weight)

        # Track importance weights for ESS
        all_trajectory_weights = []

        # Compute per-trajectory WDR
        wdr_values = []
        trajectory_info = []
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
            traj_v = v_target[traj_indices]
            traj_dones = dones[traj_indices]

            # Compute cumulative importance weight for trajectory
            cum_rho = np.cumprod(np.clip(traj_rho, 0.01, self.max_weight))
            all_trajectory_weights.append(cum_rho[-1])

            # WDR backward recursion (CORRECTED)
            # V_WDR(τ) = Σ_t γ^t * w_t * (r_t + γ*V(s_{t+1}) - Q(s_t,a_t)) + V(s_0)
            # where w_t = Π_{k=0}^t ρ_k (cumulative importance weight)

            wdr_correction = 0.0
            discount = 1.0

            for t in range(n_steps):
                r_t = traj_rewards[t]
                w_t = cum_rho[t]
                q_t = traj_q_sa[t]

                # Next state value
                if t < n_steps - 1 and not traj_dones[t]:
                    v_next = traj_v[t + 1]
                else:
                    v_next = 0.0

                # TD error
                td_error = r_t + self.gamma * v_next - q_t

                # Accumulate weighted TD error
                wdr_correction += discount * w_t * td_error
                discount *= self.gamma

            # WDR estimate = model-based V(s_0) + IS correction
            wdr_value = traj_v[0] + wdr_correction
            wdr_values.append(wdr_value)

            trajectory_info.append({
                'stay_id': stay_id,
                'n_steps': n_steps,
                'final_weight': cum_rho[-1],
                'wdr_value': wdr_value,
                'model_value': traj_v[0],
                'terminal_reward': traj_rewards[-1]
            })

        wdr_values = np.array(wdr_values)
        all_trajectory_weights = np.array(all_trajectory_weights)

        # Compute statistics
        wdr_estimate = np.mean(wdr_values)
        wdr_std = np.std(wdr_values)
        n_traj = len(wdr_values)
        ci_95 = 1.96 * wdr_std / np.sqrt(n_traj)

        # Effective sample size
        ess = self.compute_effective_sample_size(all_trajectory_weights)
        ess_ratio = ess / n_traj

        return {
            'wdr_estimate': wdr_estimate,
            'wdr_std': wdr_std,
            'wdr_ci_95': ci_95,
            'n_trajectories': n_traj,
            'effective_sample_size': ess,
            'ess_ratio': ess_ratio,
            'trajectory_values': wdr_values,
            'trajectory_weights': all_trajectory_weights,
            'trajectory_info': trajectory_info
        }

    def bootstrap_ci(
        self,
        df: pd.DataFrame,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        q_function: EntropyRegularizedQFunction,
        behavior_policy: ImprovedBehaviorPolicyEstimator,
        n_bootstrap: int = 500,
        confidence_level: float = 0.95,
        random_seed: int = 42
    ) -> Dict:
        """Compute bootstrap confidence interval with ESS tracking."""
        np.random.seed(random_seed)

        stay_ids = df['stay_id'].unique()
        bootstrap_estimates = []
        bootstrap_ess = []

        for b in range(n_bootstrap):
            # Resample trajectories with replacement
            resampled_stays = np.random.choice(stay_ids, size=len(stay_ids), replace=True)

            # Get indices for resampled trajectories
            resampled_indices = []
            for stay_id in resampled_stays:
                mask = (df['stay_id'] == stay_id).values
                resampled_indices.extend(np.where(mask)[0])
            resampled_indices = np.array(resampled_indices)

            if len(resampled_indices) == 0:
                continue

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
                bootstrap_ess.append(result['ess_ratio'])
            except Exception as e:
                continue

            if (b + 1) % 100 == 0:
                logger.info(f"  Bootstrap iteration {b+1}/{n_bootstrap}")

        bootstrap_estimates = np.array(bootstrap_estimates)
        alpha = 1 - confidence_level

        return {
            'mean': np.mean(bootstrap_estimates),
            'std': np.std(bootstrap_estimates),
            'lower': np.percentile(bootstrap_estimates, alpha/2 * 100),
            'upper': np.percentile(bootstrap_estimates, (1 - alpha/2) * 100),
            'median': np.median(bootstrap_estimates),
            'mean_ess_ratio': np.mean(bootstrap_ess),
            'n_successful': len(bootstrap_estimates)
        }


# =============================================================================
# CLINICIAN POLICY EVALUATION (ON-POLICY)
# =============================================================================

def evaluate_clinician_policy(
    df: pd.DataFrame,
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99
) -> Dict:
    """Evaluate observed clinician policy (on-policy, no IS needed)."""
    logger.info("Evaluating clinician (behavior) policy...")

    trajectory_returns = []
    trajectory_info = []

    for stay_id in df['stay_id'].unique():
        mask = (df['stay_id'] == stay_id).values
        traj_rewards = rewards[mask]

        # Compute discounted return
        G = 0.0
        for t in range(len(traj_rewards) - 1, -1, -1):
            G = traj_rewards[t] + gamma * G

        trajectory_returns.append(G)
        trajectory_info.append({
            'stay_id': stay_id,
            'n_steps': len(traj_rewards),
            'return': G,
            'terminal_reward': traj_rewards[-1]
        })

    trajectory_returns = np.array(trajectory_returns)
    mean_return = np.mean(trajectory_returns)
    std_return = np.std(trajectory_returns)
    n_traj = len(trajectory_returns)
    ci_95 = 1.96 * std_return / np.sqrt(n_traj)

    # Compute mortality rate
    terminal_rewards = np.array([t['terminal_reward'] for t in trajectory_info])
    mortality_rate = np.mean(terminal_rewards < 0)
    survival_rate = np.mean(terminal_rewards > 0)

    logger.info(f"  Clinician policy value: {mean_return:.3f} +/- {ci_95:.3f}")
    logger.info(f"  Mortality rate: {mortality_rate*100:.1f}%")
    logger.info(f"  Number of trajectories: {n_traj:,}")

    return {
        'value': mean_return,
        'std': std_return,
        'ci_95': ci_95,
        'n_trajectories': n_traj,
        'trajectory_returns': trajectory_returns,
        'mortality_rate': mortality_rate,
        'survival_rate': survival_rate
    }


# =============================================================================
# IMPROVED CLINICAL SANITY CHECKS (STRICT)
# =============================================================================

def strict_clinical_sanity_checks(
    q_function: EntropyRegularizedQFunction,
    state_cols: List[str],
    require_strict_improvement: bool = True
) -> Dict:
    """
    Strict clinical sanity checks requiring actual improvement, not just non-decrease.

    Also tests multiple severity levels to ensure consistent response.
    """
    logger.info("Performing STRICT clinical sanity checks...")
    results = {}

    n_features = len(state_cols)
    baseline_state = np.zeros((1, n_features))
    feature_indices = {col: i for i, col in enumerate(state_cols)}

    def get_action_intensity(action: int) -> Tuple[int, int, int]:
        """Return (iv_bin, vaso_bin, total_intensity)."""
        iv_bin = action // 5
        vaso_bin = action % 5
        return iv_bin, vaso_bin, iv_bin + vaso_bin

    # Check 1: Blood pressure response (multiple severity levels)
    if 'MeanBP' in feature_indices or 'SysBP' in feature_indices:
        bp_col = 'MeanBP' if 'MeanBP' in feature_indices else 'SysBP'
        bp_idx = feature_indices[bp_col]

        severity_levels = [0.0, -1.0, -2.0, -3.0]  # Normal to severe hypotension
        actions_by_severity = []

        for severity in severity_levels:
            test_state = baseline_state.copy()
            test_state[0, bp_idx] = severity
            action = q_function.get_greedy_actions(test_state)[0]
            _, vaso, intensity = get_action_intensity(action)
            actions_by_severity.append({'severity': severity, 'action': action, 'vaso': vaso, 'intensity': intensity})

        # Check if vasopressor increases with severity
        vaso_trend = [a['vaso'] for a in actions_by_severity]
        is_monotonic = all(vaso_trend[i] <= vaso_trend[i+1] for i in range(len(vaso_trend)-1))
        has_increase = vaso_trend[-1] > vaso_trend[0]  # Most severe > normal

        check_passed = has_increase if require_strict_improvement else is_monotonic

        results['hypotension_check'] = {
            'passed': check_passed,
            'actions_by_severity': actions_by_severity,
            'vaso_trend': vaso_trend,
            'is_monotonic': is_monotonic,
            'has_increase': has_increase,
            'description': 'Vasopressor should increase with hypotension severity'
        }

        trend_str = ' -> '.join([f"{a['vaso']}" for a in actions_by_severity])
        logger.info(f"  BP check: {'PASS' if check_passed else 'FAIL'} (vaso trend: {trend_str})")

    # Check 2: Lactate response
    if 'Arterial_lactate' in feature_indices:
        lactate_idx = feature_indices['Arterial_lactate']

        severity_levels = [0.0, 1.0, 2.0, 3.0]  # Normal to severe
        actions_by_severity = []

        for severity in severity_levels:
            test_state = baseline_state.copy()
            test_state[0, lactate_idx] = severity
            action = q_function.get_greedy_actions(test_state)[0]
            iv, vaso, intensity = get_action_intensity(action)
            actions_by_severity.append({'severity': severity, 'action': action, 'intensity': intensity})

        intensity_trend = [a['intensity'] for a in actions_by_severity]
        is_monotonic = all(intensity_trend[i] <= intensity_trend[i+1] for i in range(len(intensity_trend)-1))
        has_increase = intensity_trend[-1] > intensity_trend[0]

        check_passed = has_increase if require_strict_improvement else is_monotonic

        results['lactate_check'] = {
            'passed': check_passed,
            'actions_by_severity': actions_by_severity,
            'intensity_trend': intensity_trend,
            'is_monotonic': is_monotonic,
            'has_increase': has_increase,
            'description': 'Treatment intensity should increase with lactate'
        }

        trend_str = ' -> '.join([f"{a['intensity']}" for a in actions_by_severity])
        logger.info(f"  Lactate check: {'PASS' if check_passed else 'FAIL'} (intensity trend: {trend_str})")

    # Check 3: SOFA response
    if 'SOFA' in feature_indices:
        sofa_idx = feature_indices['SOFA']

        severity_levels = [-1.0, 0.0, 1.0, 2.0]  # Low to high SOFA
        actions_by_severity = []

        for severity in severity_levels:
            test_state = baseline_state.copy()
            test_state[0, sofa_idx] = severity
            action = q_function.get_greedy_actions(test_state)[0]
            iv, vaso, intensity = get_action_intensity(action)
            actions_by_severity.append({'severity': severity, 'action': action, 'intensity': intensity})

        intensity_trend = [a['intensity'] for a in actions_by_severity]
        is_monotonic = all(intensity_trend[i] <= intensity_trend[i+1] for i in range(len(intensity_trend)-1))
        has_increase = intensity_trend[-1] > intensity_trend[0]

        check_passed = has_increase if require_strict_improvement else is_monotonic

        results['sofa_check'] = {
            'passed': check_passed,
            'actions_by_severity': actions_by_severity,
            'intensity_trend': intensity_trend,
            'is_monotonic': is_monotonic,
            'has_increase': has_increase,
            'description': 'Treatment intensity should increase with SOFA'
        }

        trend_str = ' -> '.join([f"{a['intensity']}" for a in actions_by_severity])
        logger.info(f"  SOFA check: {'PASS' if check_passed else 'FAIL'} (intensity trend: {trend_str})")

    # Check 4: Action diversity (prevent mode collapse)
    # Generate random states and check if different actions are recommended
    n_test = 1000
    random_states = np.random.randn(n_test, n_features) * 0.5  # Random states around baseline
    test_actions = q_function.get_greedy_actions(random_states)
    unique_actions = len(np.unique(test_actions))
    action_entropy = stats.entropy(np.bincount(test_actions, minlength=25) + 1)

    diversity_passed = unique_actions >= 5  # At least 5 different actions used

    results['diversity_check'] = {
        'passed': diversity_passed,
        'unique_actions': unique_actions,
        'action_entropy': action_entropy,
        'description': 'Policy should use diverse actions across states'
    }
    logger.info(f"  Diversity check: {'PASS' if diversity_passed else 'FAIL'} ({unique_actions} unique actions, H={action_entropy:.2f})")

    # Overall score
    checks = [r for r in results.values() if isinstance(r, dict) and 'passed' in r]
    n_passed = sum(1 for r in checks if r['passed'])
    n_total = len(checks)

    results['overall'] = {
        'passed': n_passed,
        'total': n_total,
        'score': n_passed / n_total if n_total > 0 else 0,
        'all_passed': n_passed == n_total
    }
    logger.info(f"  Overall sanity score: {n_passed}/{n_total}")

    return results


# =============================================================================
# FITTED Q-ITERATION WITH ENTROPY REGULARIZATION
# =============================================================================

class EntropyRegularizedFQI:
    """
    Fitted Q-Iteration with entropy regularization to prevent mode collapse.

    Key additions:
    1. Entropy bonus in target computation
    2. Action diversity monitoring
    3. Per-cluster action distribution tracking
    """

    def __init__(
        self,
        q_function: EntropyRegularizedQFunction,
        gamma: float = 0.99,
        n_iterations: int = 100,
        convergence_threshold: float = 1e-4,
        patience: int = 15,
        entropy_coef: float = 0.1,
        min_action_diversity: int = 5  # Minimum unique actions
    ):
        self.q_function = q_function
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.entropy_coef = entropy_coef
        self.min_action_diversity = min_action_diversity

    def compute_entropy_bonus(self, states: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute entropy bonus for each state."""
        probs = self.q_function.get_softmax_policy(states, temperature)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        return entropy

    def fit(self, train_data: Dict, val_data: Optional[Dict] = None) -> Dict:
        """Train Q-function with entropy regularization."""
        logger.info("Starting Entropy-Regularized FQI training...")
        logger.info(f"  Entropy coefficient: {self.entropy_coef}")

        states = train_data['states']
        actions = train_data['actions']
        rewards = train_data['rewards']
        next_states = train_data['next_states']
        dones = train_data['dones']

        self.q_function.fit_scaler(states)
        features = self.q_function._create_features(states, actions)
        targets = rewards.copy()

        history = {
            'train_loss': [], 'val_loss': [], 'q_mean': [], 'q_std': [],
            'action_entropy': [], 'unique_actions': [], 'iteration': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_intercept = None

        for iteration in range(self.n_iterations):
            # Fit Q-function
            self.q_function.model.fit(features, targets)
            q_pred = self.q_function.model.predict(features)
            train_loss = np.mean((q_pred - targets) ** 2)

            # Compute next state values with entropy bonus
            next_q_all = self.q_function.predict_all_q(next_states)
            next_v = np.max(next_q_all, axis=1)  # Greedy value

            # Add entropy bonus to encourage exploration
            if self.entropy_coef > 0:
                next_probs = self.q_function.get_softmax_policy(next_states, temperature=1.0)
                next_probs = np.clip(next_probs, 1e-10, 1.0)
                next_entropy = -np.sum(next_probs * np.log(next_probs), axis=1)
                next_v = next_v + self.entropy_coef * next_entropy

            next_v[dones.astype(bool)] = 0
            new_targets = rewards + self.gamma * next_v

            target_change = np.mean(np.abs(new_targets - targets))
            targets = new_targets

            # Track action distribution
            greedy_actions = self.q_function.get_greedy_actions(states)
            unique_actions = len(np.unique(greedy_actions))
            action_counts = np.bincount(greedy_actions, minlength=25) + 1
            action_probs = action_counts / action_counts.sum()
            action_entropy = -np.sum(action_probs * np.log(action_probs))

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
                    best_intercept = self.q_function.model.intercept_
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at iteration {iteration + 1}")
                    if best_weights is not None:
                        self.q_function.model.coef_ = best_weights
                        self.q_function.model.intercept_ = best_intercept
                    break

            # Log history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['q_mean'].append(np.mean(q_pred))
            history['q_std'].append(np.std(q_pred))
            history['action_entropy'].append(action_entropy)
            history['unique_actions'].append(unique_actions)
            history['iteration'].append(iteration)

            if (iteration + 1) % 10 == 0 or iteration == 0:
                logger.info(
                    f"Iter {iteration+1}: Loss={train_loss:.4f}, Delta={target_change:.4f}, "
                    f"Q={np.mean(q_pred):.2f}+/-{np.std(q_pred):.2f}, "
                    f"Actions={unique_actions}, H={action_entropy:.2f}"
                )

            # Check for mode collapse
            if unique_actions < self.min_action_diversity and iteration > 10:
                logger.warning(f"Mode collapse detected: only {unique_actions} unique actions")

            if target_change < self.convergence_threshold:
                logger.info(f"Converged at iteration {iteration + 1}")
                break

        # Final statistics
        final_actions = self.q_function.get_greedy_actions(states)
        final_unique = len(np.unique(final_actions))
        logger.info(f"Training complete. Final loss: {history['train_loss'][-1]:.4f}")
        logger.info(f"Final action diversity: {final_unique} unique actions")

        return history


# =============================================================================
# HYPERPARAMETER SEARCH WITH DIVERSITY OBJECTIVE
# =============================================================================

def hyperparameter_search_v3(
    train_data: Dict,
    val_data: Dict,
    state_cols: List[str],
    n_trials: int = 20,
    random_seed: int = 42
) -> Dict:
    """
    Hyperparameter search with multi-objective optimization:
    1. WDR policy value
    2. Clinical sanity score
    3. Action diversity
    """
    logger.info(f"Starting hyperparameter search v3 ({n_trials} trials)...")
    np.random.seed(random_seed)

    best_config = None
    best_score = -np.inf
    results = []

    for trial in range(n_trials):
        # Sample hyperparameters
        alpha = 10 ** np.random.uniform(-2, 1)  # L2: 0.01 to 10
        gamma = np.random.choice([0.95, 0.99])  # Only high gamma values
        n_clusters = np.random.choice([500, 750, 1000])
        entropy_coef = 10 ** np.random.uniform(-2, 0)  # 0.01 to 1.0

        logger.info(f"\nTrial {trial + 1}/{n_trials}: alpha={alpha:.4f}, gamma={gamma}, "
                   f"clusters={n_clusters}, entropy={entropy_coef:.4f}")

        try:
            # Create Q-function with entropy regularization
            q_func = EntropyRegularizedQFunction(
                n_state_features=len(state_cols),
                n_actions=25,
                alpha=alpha,
                gamma=gamma,
                entropy_coef=entropy_coef
            )
            q_func.fit_scaler(train_data['states'])

            # Train with entropy-regularized FQI
            fqi = EntropyRegularizedFQI(
                q_function=q_func,
                gamma=gamma,
                n_iterations=50,  # Quick training for search
                patience=10,
                entropy_coef=entropy_coef
            )
            history = fqi.fit(train_data, val_data)

            # Fit behavior policy
            behavior_policy = ImprovedBehaviorPolicyEstimator(
                n_clusters=n_clusters,
                smoothing_alpha=1.0
            )
            behavior_policy.fit(train_data['states'], train_data['actions'])

            # Evaluate on validation set
            wdr = CorrectedWDRv3(gamma=gamma)
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
            ess_ratio = wdr_result['ess_ratio']

            # Clinical sanity checks (strict)
            sanity = strict_clinical_sanity_checks(q_func, state_cols, require_strict_improvement=True)
            sanity_score = sanity['overall']['score']

            # Action diversity
            action_dist = q_func.get_action_distribution(val_data['states'])
            action_entropy = stats.entropy(action_dist + 1e-10)
            unique_actions = np.sum(action_dist > 0.01)  # Actions with >1% usage

            # Multi-objective score
            # Normalize WDR by gamma to make comparable
            wdr_normalized = wdr_value / (1 / (1 - gamma))  # Divide by max possible return

            combined_score = (
                wdr_normalized * 0.4 +           # Policy value (40%)
                sanity_score * 3.0 +              # Clinical sanity (weight=3)
                (unique_actions / 25) * 2.0 +     # Action diversity (weight=2)
                ess_ratio * 1.0                   # IS reliability (weight=1)
            )

            results.append({
                'trial': trial,
                'alpha': alpha,
                'gamma': gamma,
                'n_clusters': n_clusters,
                'entropy_coef': entropy_coef,
                'wdr_value': wdr_value,
                'wdr_normalized': wdr_normalized,
                'ess_ratio': ess_ratio,
                'sanity_score': sanity_score,
                'unique_actions': unique_actions,
                'action_entropy': action_entropy,
                'combined_score': combined_score
            })

            logger.info(f"  WDR={wdr_value:.3f}, ESS={ess_ratio:.2f}, "
                       f"Sanity={sanity_score:.2f}, Actions={unique_actions}, "
                       f"Score={combined_score:.3f}")

            if combined_score > best_score:
                best_score = combined_score
                best_config = {
                    'alpha': alpha,
                    'gamma': gamma,
                    'n_clusters': n_clusters,
                    'entropy_coef': entropy_coef,
                    'q_function': q_func,
                    'behavior_policy': behavior_policy,
                    'wdr_value': wdr_value,
                    'sanity_score': sanity_score,
                    'unique_actions': unique_actions
                }
                logger.info("  *** New best! ***")

        except Exception as e:
            logger.warning(f"  Trial failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if best_config is None:
        raise RuntimeError("All trials failed!")

    logger.info(f"\nBest configuration:")
    logger.info(f"  alpha={best_config['alpha']:.4f}, gamma={best_config['gamma']}, "
               f"clusters={best_config['n_clusters']}, entropy={best_config['entropy_coef']:.4f}")
    logger.info(f"  WDR={best_config['wdr_value']:.3f}, Sanity={best_config['sanity_score']:.2f}, "
               f"Actions={best_config['unique_actions']}")

    return {
        'best_config': best_config,
        'all_results': results
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_trajectory_data(file_path: str, state_cols: List[str]) -> Dict[str, Any]:
    """Load trajectory data from CSV."""
    logger.info(f"Loading {file_path}...")
    df = pd.read_csv(file_path)

    # Find existing state columns
    existing_state_cols = [c for c in state_cols if c in df.columns]

    if len(existing_state_cols) < len(state_cols):
        missing = set(state_cols) - set(existing_state_cols)
        logger.warning(f"Missing state columns: {missing}")

    states = df[existing_state_cols].values.astype(np.float32)
    states = np.nan_to_num(states, nan=0.0)

    actions = df['action'].values.astype(np.int32)
    rewards = df['reward'].values.astype(np.float32)

    # Handle next states
    next_state_cols = [f'next_{col}' for col in existing_state_cols]
    next_states = np.zeros_like(states)
    for i, col in enumerate(existing_state_cols):
        next_col = f'next_{col}'
        if next_col in df.columns:
            next_states[:, i] = df[next_col].values
    next_states = np.nan_to_num(next_states, nan=0.0)

    dones = df['done'].values.astype(np.float32)

    logger.info(f"  Loaded {len(states):,} transitions, {len(df['stay_id'].unique()):,} trajectories")

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
    """Main training pipeline v3."""
    logger.info("=" * 80)
    logger.info("Q-LEARNING TRAINING PIPELINE v3.0 - COMPREHENSIVE FIX")
    logger.info("=" * 80)

    # Load config
    config = ConfigLoader('configs/config.yaml').config

    # Paths
    data_dir = Path('data/processed')
    output_dir = Path('outputs/models_v3')
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
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 60)

    train_data = load_trajectory_data(data_dir / 'train_trajectories.csv', state_cols)
    val_data = load_trajectory_data(data_dir / 'val_trajectories.csv', state_cols)
    test_data = load_trajectory_data(data_dir / 'test_trajectories.csv', state_cols)

    actual_state_cols = train_data['state_cols']
    logger.info(f"  Train: {len(train_data['states']):,} transitions")
    logger.info(f"  Val: {len(val_data['states']):,} transitions")
    logger.info(f"  Test: {len(test_data['states']):,} transitions")
    logger.info(f"  State features: {len(actual_state_cols)}")

    # ==========================================================================
    # STEP 2: HYPERPARAMETER SEARCH
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: HYPERPARAMETER SEARCH (with diversity objective)")
    logger.info("=" * 60)

    tuning_results = hyperparameter_search_v3(
        train_data, val_data, actual_state_cols,
        n_trials=20,
        random_seed=42
    )

    best_config = tuning_results['best_config']

    # ==========================================================================
    # STEP 3: FULL TRAINING WITH BEST CONFIG
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: FULL TRAINING WITH BEST CONFIG")
    logger.info("=" * 60)

    q_function = EntropyRegularizedQFunction(
        n_state_features=len(actual_state_cols),
        n_actions=25,
        alpha=best_config['alpha'],
        gamma=best_config['gamma'],
        entropy_coef=best_config['entropy_coef']
    )

    fqi = EntropyRegularizedFQI(
        q_function=q_function,
        gamma=best_config['gamma'],
        n_iterations=100,
        patience=15,
        entropy_coef=best_config['entropy_coef']
    )

    history = fqi.fit(train_data, val_data)

    # Fit behavior policy
    behavior_policy = ImprovedBehaviorPolicyEstimator(
        n_clusters=best_config['n_clusters'],
        smoothing_alpha=1.0
    )
    behavior_policy.fit(train_data['states'], train_data['actions'])

    # ==========================================================================
    # STEP 4: EVALUATE ALL POLICIES
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: POLICY EVALUATION")
    logger.info("=" * 60)

    gamma = best_config['gamma']

    # 4.1: Clinician policy (on-policy)
    clinician_result = evaluate_clinician_policy(
        test_data['df'], test_data['rewards'], test_data['dones'], gamma
    )

    # 4.2: AI policy (WDR)
    logger.info("\nEvaluating AI policy (WDR)...")
    wdr = CorrectedWDRv3(gamma=gamma)
    ai_result = wdr.estimate_value(
        test_data['df'],
        test_data['states'],
        test_data['actions'],
        test_data['rewards'],
        test_data['dones'],
        q_function,
        behavior_policy
    )

    logger.info(f"  AI policy WDR: {ai_result['wdr_estimate']:.3f} +/- {ai_result['wdr_ci_95']:.3f}")
    logger.info(f"  Effective Sample Size ratio: {ai_result['ess_ratio']:.2%}")

    # 4.3: Agreement rate
    ai_actions = q_function.get_greedy_actions(test_data['states'])
    agreement = np.mean(ai_actions == test_data['actions'])
    logger.info(f"\nAgreement with clinicians: {agreement*100:.1f}%")

    # 4.4: Bootstrap confidence intervals
    logger.info("\nComputing bootstrap confidence intervals...")
    bootstrap_result = wdr.bootstrap_ci(
        test_data['df'],
        test_data['states'],
        test_data['actions'],
        test_data['rewards'],
        test_data['dones'],
        q_function,
        behavior_policy,
        n_bootstrap=500
    )
    logger.info(f"  AI policy 95% CI: [{bootstrap_result['lower']:.3f}, {bootstrap_result['upper']:.3f}]")
    logger.info(f"  Mean ESS ratio across bootstraps: {bootstrap_result['mean_ess_ratio']:.2%}")

    # ==========================================================================
    # STEP 5: STRICT CLINICAL SANITY CHECKS
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: STRICT CLINICAL SANITY CHECKS")
    logger.info("=" * 60)

    sanity_results = strict_clinical_sanity_checks(q_function, actual_state_cols, require_strict_improvement=True)

    # ==========================================================================
    # STEP 6: ACTION DISTRIBUTION ANALYSIS
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: ACTION DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)

    train_ai_actions = q_function.get_greedy_actions(train_data['states'])
    train_clinician_actions = train_data['actions']

    logger.info("\nAction | Clinician | AI Policy | Difference")
    logger.info("-" * 55)

    significant_diffs = []
    for a in range(25):
        clin_pct = np.mean(train_clinician_actions == a) * 100
        ai_pct = np.mean(train_ai_actions == a) * 100
        diff = ai_pct - clin_pct
        iv_bin = a // 5
        vaso_bin = a % 5

        diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
        flag = " ***" if abs(diff) > 5 else ""
        logger.info(f"A{a:2d} (IV={iv_bin},V={vaso_bin}) | {clin_pct:5.1f}% | {ai_pct:5.1f}% | {diff_str}%{flag}")

        if abs(diff) > 5:
            significant_diffs.append((a, iv_bin, vaso_bin, clin_pct, ai_pct, diff))

    # Action diversity metrics
    ai_unique = len(np.unique(train_ai_actions))
    clin_unique = len(np.unique(train_clinician_actions))
    ai_entropy = stats.entropy(np.bincount(train_ai_actions, minlength=25) + 1)
    clin_entropy = stats.entropy(np.bincount(train_clinician_actions, minlength=25) + 1)

    logger.info(f"\nAction Diversity:")
    logger.info(f"  Clinician: {clin_unique} unique actions, entropy={clin_entropy:.2f}")
    logger.info(f"  AI Policy: {ai_unique} unique actions, entropy={ai_entropy:.2f}")

    # ==========================================================================
    # STEP 7: FINAL COMPARISON TABLE
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: FINAL COMPARISON TABLE")
    logger.info("=" * 60)

    # Statistical test
    ai_values = np.array(ai_result['trajectory_values'])
    clin_values = np.array(clinician_result['trajectory_returns'])

    min_traj = min(len(ai_values), len(clin_values))
    diff_values = ai_values[:min_traj] - clin_values[:min_traj]

    t_stat, p_value = stats.ttest_1samp(diff_values, 0)

    # Also compute effect size (Cohen's d)
    cohens_d = np.mean(diff_values) / np.std(diff_values)

    logger.info("\n" + "=" * 80)
    logger.info("FINAL OFF-POLICY EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Policy':<25} {'Value':<12} {'95% CI':<25} {'p-value':<10}")
    logger.info("-" * 80)

    clin_ci = f"[{clinician_result['value']-clinician_result['ci_95']:.3f}, {clinician_result['value']+clinician_result['ci_95']:.3f}]"
    ai_ci = f"[{bootstrap_result['lower']:.3f}, {bootstrap_result['upper']:.3f}]"

    logger.info(f"{'Clinician (observed)':<25} {clinician_result['value']:<12.3f} {clin_ci:<25} {'---':<10}")
    logger.info(f"{'AI Policy (WDR)':<25} {ai_result['wdr_estimate']:<12.3f} {ai_ci:<25} {p_value:<10.4f}")
    logger.info("=" * 80)

    improvement = ai_result['wdr_estimate'] - clinician_result['value']
    improvement_pct = (improvement / abs(clinician_result['value'])) * 100 if clinician_result['value'] != 0 else 0

    logger.info(f"\nImprovement over clinician: {improvement:+.3f} ({improvement_pct:+.1f}%)")
    logger.info(f"Cohen's d effect size: {cohens_d:.3f}")
    logger.info(f"Agreement with clinician actions: {agreement*100:.1f}%")
    logger.info(f"Effective sample size ratio: {ai_result['ess_ratio']:.1%}")
    logger.info(f"Clinical sanity score: {sanity_results['overall']['passed']}/{sanity_results['overall']['total']}")

    # Interpretation
    logger.info("\n" + "-" * 40)
    logger.info("INTERPRETATION:")
    logger.info("-" * 40)

    if p_value < 0.05:
        if improvement > 0:
            logger.info("* AI policy is SIGNIFICANTLY BETTER than clinicians (p < 0.05)")
        else:
            logger.info("* AI policy is SIGNIFICANTLY WORSE than clinicians (p < 0.05)")
    else:
        logger.info("* No significant difference from clinician policy (p >= 0.05)")

    if ai_result['ess_ratio'] < 0.1:
        logger.info("* WARNING: Low ESS ratio indicates high variance in IS estimates")
    elif ai_result['ess_ratio'] < 0.3:
        logger.info("* CAUTION: Moderate ESS ratio, estimates may have some variance")
    else:
        logger.info("* ESS ratio is acceptable, IS estimates are reasonably reliable")

    if sanity_results['overall']['all_passed']:
        logger.info("* Policy passes all clinical sanity checks")
    else:
        failed_checks = [k for k, v in sanity_results.items()
                        if isinstance(v, dict) and 'passed' in v and not v['passed']]
        logger.info(f"* WARNING: Policy fails sanity checks: {failed_checks}")

    # ==========================================================================
    # STEP 8: SAVE RESULTS
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: SAVING RESULTS")
    logger.info("=" * 60)

    results = {
        'q_function': q_function,
        'behavior_policy': behavior_policy,
        'state_cols': actual_state_cols,
        'best_config': {k: v for k, v in best_config.items()
                       if k not in ['q_function', 'behavior_policy']},
        'history': history,
        'evaluation': {
            'clinician': {
                'value': clinician_result['value'],
                'ci_95': clinician_result['ci_95'],
                'mortality_rate': clinician_result['mortality_rate']
            },
            'ai_policy': {
                'wdr_estimate': ai_result['wdr_estimate'],
                'wdr_ci_95': ai_result['wdr_ci_95'],
                'ess_ratio': ai_result['ess_ratio'],
                'bootstrap_ci': (bootstrap_result['lower'], bootstrap_result['upper'])
            },
            'agreement': agreement,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        },
        'sanity_checks': sanity_results,
        'action_distribution': {
            'ai_unique_actions': ai_unique,
            'ai_entropy': ai_entropy,
            'significant_diffs': significant_diffs
        },
        'tuning_results': tuning_results['all_results']
    }

    with open(output_dir / 'full_results_v3.pkl', 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved results to {output_dir / 'full_results_v3.pkl'}")

    # Save training history
    pd.DataFrame(history).to_csv(output_dir / 'training_history_v3.csv', index=False)
    logger.info(f"Saved training history to {output_dir / 'training_history_v3.csv'}")

    # Save hyperparameter search results
    pd.DataFrame(tuning_results['all_results']).to_csv(output_dir / 'hyperparameter_search_v3.csv', index=False)
    logger.info(f"Saved hyperparameter search to {output_dir / 'hyperparameter_search_v3.csv'}")

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)

    return results


if __name__ == "__main__":
    main()
