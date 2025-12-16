"""
Weighted Doubly Robust (WDR) Off-Policy Evaluation

Implements WDR estimator for evaluating RL policies from offline data.

Reference:
- Thomas & Brunskill (2016) "Data-Efficient Off-Policy Policy Evaluation"

Author: AI Clinician Project
Date: 2024-11-16
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class WeightedDoublyRobust:
    """
    Weighted Doubly Robust estimator for off-policy evaluation.

    Combines importance sampling with value function estimation for
    lower-variance policy evaluation.
    """

    def __init__(self, gamma: float = 0.99):
        """
        Initialize WDR estimator.

        Args:
            gamma: Discount factor
        """
        self.gamma = gamma
        logger.info(f"WeightedDoublyRobust initialized (γ={gamma})")

    def estimate_value(
        self,
        trajectories_df: pd.DataFrame,
        target_policy_probs: np.ndarray,
        behavior_policy_probs: np.ndarray,
        q_values: np.ndarray,
        rewards: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Estimate policy value using WDR.

        Args:
            trajectories_df: DataFrame with trajectories (must have stay_id, time_window)
            target_policy_probs: π(a|s) for target policy (n_transitions,)
            behavior_policy_probs: π_b(a|s) for behavior policy (n_transitions,)
            q_values: Q(s, a) estimates (n_transitions,)
            rewards: Observed rewards (n_transitions,)

        Returns:
            Tuple of (estimated_value, info_dict)
        """
        logger.info("Computing WDR estimate...")

        # Sort by stay and time
        df = trajectories_df.copy()
        df['target_prob'] = target_policy_probs
        df['behavior_prob'] = behavior_policy_probs
        df['q_value'] = q_values
        df['reward'] = rewards

        df = df.sort_values(['stay_id', 'time_window']).reset_index(drop=True)

        # Compute importance sampling ratios
        df['rho'] = df['target_prob'] / (df['behavior_prob'] + 1e-10)

        # Compute cumulative importance weights per trajectory
        df['cumulative_rho'] = df.groupby('stay_id')['rho'].cumprod()

        # WDR formula per trajectory
        wdr_values = []
        trajectory_lengths = []

        for stay_id, traj_df in df.groupby('stay_id'):
            traj_df = traj_df.sort_values('time_window').reset_index(drop=True)

            # Initialize value
            V = 0.0

            # Backward pass through trajectory
            for t in range(len(traj_df) - 1, -1, -1):
                row = traj_df.iloc[t]

                # Get cumulative importance ratio up to time t
                rho_t = row['cumulative_rho']

                # Get reward and Q-value
                r_t = row['reward']
                q_t = row['q_value']

                # Get next state value (0 if terminal)
                if t < len(traj_df) - 1:
                    V_next = traj_df.iloc[t + 1]['q_value']
                else:
                    V_next = 0.0

                # WDR term: ρ_t * (r_t + γ * V_next - Q_t) + Q_t
                V = rho_t * (r_t + self.gamma * V_next - q_t) + q_t

            wdr_values.append(V)
            trajectory_lengths.append(len(traj_df))

        # Average over trajectories
        wdr_estimate = np.mean(wdr_values)
        wdr_std = np.std(wdr_values)

        # Confidence interval (95%)
        n_trajectories = len(wdr_values)
        ci_95 = 1.96 * wdr_std / np.sqrt(n_trajectories)

        logger.info(f"✓ WDR estimate: {wdr_estimate:.3f} ± {ci_95:.3f}")
        logger.info(f"  Number of trajectories: {n_trajectories:,}")
        logger.info(f"  Average trajectory length: {np.mean(trajectory_lengths):.1f}")

        # Additional statistics
        info = {
            'wdr_estimate': wdr_estimate,
            'wdr_std': wdr_std,
            'wdr_ci_95': ci_95,
            'n_trajectories': n_trajectories,
            'avg_trajectory_length': np.mean(trajectory_lengths),
            'trajectory_values': wdr_values,
        }

        return wdr_estimate, info

    def bootstrap_confidence_interval(
        self,
        trajectories_df: pd.DataFrame,
        target_policy_probs: np.ndarray,
        behavior_policy_probs: np.ndarray,
        q_values: np.ndarray,
        rewards: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for WDR estimate.

        Args:
            trajectories_df: Trajectories DataFrame
            target_policy_probs: Target policy probabilities
            behavior_policy_probs: Behavior policy probabilities
            q_values: Q-value estimates
            rewards: Rewards
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            random_seed: Random seed

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        logger.info(f"Computing bootstrap CI ({n_bootstrap} samples, {confidence_level*100:.0f}%)...")

        np.random.seed(random_seed)

        # Get unique stay IDs
        stay_ids = trajectories_df['stay_id'].unique()
        n_stays = len(stay_ids)

        bootstrap_estimates = []

        for i in range(n_bootstrap):
            # Resample trajectories (with replacement)
            resampled_stays = np.random.choice(stay_ids, size=n_stays, replace=True)

            # Get indices for resampled trajectories
            resampled_mask = trajectories_df['stay_id'].isin(resampled_stays)
            resampled_df = trajectories_df[resampled_mask].copy()

            # Compute WDR on bootstrap sample
            try:
                wdr_est, _ = self.estimate_value(
                    resampled_df,
                    target_policy_probs[resampled_mask],
                    behavior_policy_probs[resampled_mask],
                    q_values[resampled_mask],
                    rewards[resampled_mask]
                )
                bootstrap_estimates.append(wdr_est)
            except:
                # Skip if bootstrap sample causes issues
                continue

            if (i + 1) % 100 == 0:
                logger.info(f"  Bootstrap iteration {i+1}/{n_bootstrap}")

        # Compute confidence interval
        bootstrap_estimates = np.array(bootstrap_estimates)
        mean_estimate = np.mean(bootstrap_estimates)
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
        upper_bound = np.percentile(bootstrap_estimates, upper_percentile)

        logger.info(f"✓ Bootstrap CI: {mean_estimate:.3f} [{lower_bound:.3f}, {upper_bound:.3f}]")

        return mean_estimate, lower_bound, upper_bound


def main():
    """Example usage."""
    # Create dummy data
    n_samples = 1000
    n_stays = 100

    trajectories_df = pd.DataFrame({
        'stay_id': np.repeat(np.arange(n_stays), n_samples // n_stays),
        'time_window': np.tile(np.arange(n_samples // n_stays), n_stays)
    })

    target_probs = np.random.rand(n_samples) * 0.5 + 0.25
    behavior_probs = np.random.rand(n_samples) * 0.5 + 0.25
    q_values = np.random.randn(n_samples) * 5
    rewards = np.random.randn(n_samples)

    # Estimate value
    wdr = WeightedDoublyRobust(gamma=0.99)
    value, info = wdr.estimate_value(
        trajectories_df,
        target_probs,
        behavior_probs,
        q_values,
        rewards
    )

    logger.info(f"Estimated value: {value:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
