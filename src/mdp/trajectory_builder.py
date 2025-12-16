"""
Trajectory Builder Module

Builds complete MDP trajectories from preprocessed features, actions, and rewards.
Creates (state, action, reward, next_state, done) tuples for RL training.

Author: AI Clinician Project
Date: 2024-11-16
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TrajectoryBuilder:
    """
    Builds MDP trajectories for reinforcement learning.

    A trajectory is a sequence of (s, a, r, s', done) tuples representing
    one patient's ICU stay.
    """

    def __init__(self, config: Dict):
        """
        Initialize TrajectoryBuilder.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.gamma = config.get('mdp', {}).get('gamma', 0.99)

        logger.info(f"TrajectoryBuilder initialized (γ={self.gamma})")

    def build_trajectories(
        self,
        features: pd.DataFrame,
        actions: pd.DataFrame,
        rewards: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build complete MDP trajectories.

        Args:
            features: DataFrame with state features (stay_id, time_window, features...)
            actions: DataFrame with actions (stay_id, time_window, action)
            rewards: DataFrame with rewards (stay_id, time_window, reward)

        Returns:
            DataFrame with columns: stay_id, time_window, state_features, action,
                                   reward, next_state_features, done
        """
        logger.info("Building MDP trajectories...")
        logger.info(f"  Features: {len(features):,} observations")
        logger.info(f"  Actions: {len(actions):,} observations")
        logger.info(f"  Rewards: {len(rewards):,} observations")

        # Merge features, actions, and rewards
        logger.info("  Merging data...")
        trajectories = features.copy()

        # Merge actions
        trajectories = trajectories.merge(
            actions[['stay_id', 'time_window', 'action', 'iv_bin', 'vaso_bin']],
            on=['stay_id', 'time_window'],
            how='left'
        )

        # Merge rewards
        if 'reward' in rewards.columns:
            trajectories = trajectories.merge(
                rewards[['stay_id', 'time_window', 'reward']],
                on=['stay_id', 'time_window'],
                how='left'
            )
        else:
            logger.warning("  No rewards found - setting to 0")
            trajectories['reward'] = 0

        # Sort by stay and time
        trajectories = trajectories.sort_values(['stay_id', 'time_window']).reset_index(drop=True)

        # Create next_state columns
        logger.info("  Creating next-state features...")
        state_features = [col for col in features.columns
                         if col not in ['stay_id', 'time_window']]

        for feature in state_features:
            trajectories[f'next_{feature}'] = trajectories.groupby('stay_id')[feature].shift(-1)

        # Mark terminal states
        trajectories['done'] = False
        last_time_windows = trajectories.groupby('stay_id')['time_window'].transform('max')
        trajectories.loc[trajectories['time_window'] == last_time_windows, 'done'] = True

        # Remove trajectories with missing actions
        missing_actions = trajectories['action'].isna().sum()
        if missing_actions > 0:
            logger.warning(f"  Removing {missing_actions:,} observations with missing actions")
            trajectories = trajectories[trajectories['action'].notna()]

        # Remove last state of each trajectory (no next state for terminal states in training)
        # Keep them only for evaluation
        logger.info("  Filtering complete transitions...")
        complete_transitions = trajectories[~trajectories['done']].copy()

        logger.info(f"✓ Built {len(complete_transitions):,} complete transitions")
        logger.info(f"  From {trajectories['stay_id'].nunique():,} unique ICU stays")
        logger.info(f"  Average trajectory length: {len(trajectories) / trajectories['stay_id'].nunique():.1f} time steps")

        return complete_transitions

    def create_feature_matrix(
        self,
        trajectories: pd.DataFrame,
        include_next_state: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Create feature matrices for training.

        Args:
            trajectories: DataFrame with complete trajectories
            include_next_state: Whether to include next-state features

        Returns:
            Tuple of (states, actions, rewards, next_states)
            - states: (n_samples, n_features) array
            - actions: (n_samples,) array
            - rewards: (n_samples,) array
            - next_states: (n_samples, n_features) array or None
        """
        logger.info("Creating feature matrices...")

        # Identify state feature columns
        state_cols = [col for col in trajectories.columns
                     if col not in ['stay_id', 'time_window', 'action', 'reward', 'done',
                                   'iv_bin', 'vaso_bin', 'hospital_expire_flag', 'is_terminal',
                                   'sofa_prev', 'sofa_change', 'return']
                     and not col.startswith('next_')]

        # Extract states
        states = trajectories[state_cols].values
        logger.info(f"  State features: {states.shape}")

        # Extract actions
        actions = trajectories['action'].values
        logger.info(f"  Actions: {actions.shape}")

        # Extract rewards
        rewards = trajectories['reward'].values if 'reward' in trajectories.columns else np.zeros(len(trajectories))
        logger.info(f"  Rewards: {rewards.shape}")

        # Extract next states
        next_states = None
        if include_next_state:
            next_state_cols = [f'next_{col}' for col in state_cols]
            available_next_cols = [col for col in next_state_cols if col in trajectories.columns]

            if len(available_next_cols) > 0:
                next_states = trajectories[available_next_cols].values
                logger.info(f"  Next states: {next_states.shape}")
            else:
                logger.warning("  No next-state features found")

        logger.info("✓ Feature matrices created")

        return states, actions, rewards, next_states

    def split_trajectories(
        self,
        trajectories: pd.DataFrame,
        train_stays: List[int],
        val_stays: List[int],
        test_stays: List[int]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split trajectories into train/val/test sets based on stay IDs.

        Args:
            trajectories: Complete trajectories DataFrame
            train_stays: List of stay_ids for training
            val_stays: List of stay_ids for validation
            test_stays: List of stay_ids for testing

        Returns:
            Tuple of (train_trajectories, val_trajectories, test_trajectories)
        """
        logger.info("Splitting trajectories...")

        train_traj = trajectories[trajectories['stay_id'].isin(train_stays)].copy()
        val_traj = trajectories[trajectories['stay_id'].isin(val_stays)].copy()
        test_traj = trajectories[trajectories['stay_id'].isin(test_stays)].copy()

        logger.info(f"  Train: {len(train_traj):,} transitions from {len(train_stays):,} stays")
        logger.info(f"  Val:   {len(val_traj):,} transitions from {len(val_stays):,} stays")
        logger.info(f"  Test:  {len(test_traj):,} transitions from {len(test_stays):,} stays")

        return train_traj, val_traj, test_traj

    def get_trajectory_statistics(self, trajectories: pd.DataFrame) -> Dict:
        """
        Compute statistics about trajectories.

        Args:
            trajectories: Trajectories DataFrame

        Returns:
            Dictionary with statistics
        """
        stats = {}

        # Basic counts
        stats['n_transitions'] = len(trajectories)
        stats['n_stays'] = trajectories['stay_id'].nunique()
        stats['n_time_windows'] = trajectories['time_window'].nunique()

        # Trajectory lengths
        traj_lengths = trajectories.groupby('stay_id').size()
        stats['mean_trajectory_length'] = traj_lengths.mean()
        stats['median_trajectory_length'] = traj_lengths.median()
        stats['min_trajectory_length'] = traj_lengths.min()
        stats['max_trajectory_length'] = traj_lengths.max()

        # Actions
        if 'action' in trajectories.columns:
            stats['n_unique_actions'] = trajectories['action'].nunique()
            action_counts = trajectories['action'].value_counts()
            stats['most_common_action'] = action_counts.index[0]
            stats['most_common_action_freq'] = action_counts.iloc[0] / len(trajectories)

        # Rewards
        if 'reward' in trajectories.columns:
            stats['mean_reward'] = trajectories['reward'].mean()
            stats['std_reward'] = trajectories['reward'].std()
            stats['min_reward'] = trajectories['reward'].min()
            stats['max_reward'] = trajectories['reward'].max()

        # Returns
        if 'return' in trajectories.columns:
            stats['mean_return'] = trajectories['return'].mean()
            stats['std_return'] = trajectories['return'].std()

        # Mortality (if available)
        if 'hospital_expire_flag' in trajectories.columns:
            mortality_per_stay = trajectories.groupby('stay_id')['hospital_expire_flag'].first()
            stats['mortality_rate'] = mortality_per_stay.mean()

        return stats

    def save_trajectories(self, trajectories: pd.DataFrame, output_path: str):
        """
        Save trajectories to CSV file.

        Args:
            trajectories: Trajectories DataFrame
            output_path: Path to save file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        trajectories.to_csv(output_path, index=False)

        logger.info(f"✓ Trajectories saved to {output_path}")
        logger.info(f"  Size: {Path(output_path).stat().st_size / (1024**2):.1f} MB")

    def load_trajectories(self, input_path: str) -> pd.DataFrame:
        """
        Load trajectories from CSV file.

        Args:
            input_path: Path to trajectories file

        Returns:
            Trajectories DataFrame
        """
        logger.info(f"Loading trajectories from {input_path}...")

        trajectories = pd.read_csv(input_path)

        logger.info(f"✓ Loaded {len(trajectories):,} transitions")
        logger.info(f"  From {trajectories['stay_id'].nunique():,} ICU stays")

        return trajectories


def main():
    """Example usage of TrajectoryBuilder."""
    from src.utils.config_loader import ConfigLoader

    # Load config
    config = ConfigLoader('configs/config.yaml').config

    # Initialize builder
    builder = TrajectoryBuilder(config)

    logger.info("TrajectoryBuilder initialized")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
