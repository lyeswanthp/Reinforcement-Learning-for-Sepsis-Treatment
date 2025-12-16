#!/usr/bin/env python3
"""
MDP Pipeline: Actions, Rewards, and Trajectories
=================================================

This script performs the following steps:
1. Discretizes actions (input_4hourly -> IV bins, max_dose_vaso -> vaso bins)
2. Computes rewards (SOFA-based intermediate + terminal mortality rewards)
3. Builds complete MDP trajectories for RL training

Usage:
    python run_mdp_pipeline.py

Output files in data/processed/:
    - train_trajectories.csv
    - val_trajectories.csv
    - test_trajectories.csv
    - action_bins.pkl

Author: AI Clinician Project
"""

import pandas as pd
import numpy as np
import pickle
import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/mdp_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import ConfigLoader


class SimplifiedActionExtractor:
    """
    Discretizes continuous actions into 5x5=25 discrete action bins.

    Uses existing columns:
    - input_4hourly: IV fluids (mL per 4-hour window)
    - max_dose_vaso: Maximum vasopressor dose (μg/kg/min NE-equivalent)
    """

    def __init__(self, n_iv_bins: int = 5, n_vaso_bins: int = 5):
        self.n_iv_bins = n_iv_bins
        self.n_vaso_bins = n_vaso_bins
        self.n_actions = n_iv_bins * n_vaso_bins

        self.iv_bins = None
        self.vaso_bins = None
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> 'SimplifiedActionExtractor':
        """Fit action bins using percentile-based discretization on training data."""
        logger.info("Fitting action bins on training data...")

        # IV fluids: Create bins based on non-zero values
        iv_values = df[df['input_4hourly'] > 0]['input_4hourly'].values
        vaso_values = df[df['max_dose_vaso'] > 0]['max_dose_vaso'].values

        logger.info(f"  Non-zero IV fluid observations: {len(iv_values):,}")
        logger.info(f"  Non-zero vasopressor observations: {len(vaso_values):,}")

        # Percentile-based bins: [0, 25th, 50th, 75th, 100th]
        # Bin 0 = zero/very low, Bins 1-4 = quartiles of non-zero values
        if len(iv_values) > 0:
            iv_percentiles = np.percentile(iv_values, [25, 50, 75, 100])
            self.iv_bins = np.array([0, iv_percentiles[0], iv_percentiles[1],
                                      iv_percentiles[2], iv_percentiles[3], np.inf])
        else:
            self.iv_bins = np.array([0, 100, 250, 500, 1000, np.inf])

        if len(vaso_values) > 0:
            vaso_percentiles = np.percentile(vaso_values, [25, 50, 75, 100])
            self.vaso_bins = np.array([0, vaso_percentiles[0], vaso_percentiles[1],
                                        vaso_percentiles[2], vaso_percentiles[3], np.inf])
        else:
            self.vaso_bins = np.array([0, 0.05, 0.1, 0.3, 0.5, np.inf])

        logger.info(f"  IV bins: {self.iv_bins[:-1]}")
        logger.info(f"  Vaso bins: {self.vaso_bins[:-1]}")

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform continuous actions to discrete bins."""
        if not self.fitted:
            raise ValueError("ActionExtractor must be fitted before transform")

        df = df.copy()

        # Discretize IV fluids (0 = no/minimal fluids, 1-4 = increasing amounts)
        df['iv_bin'] = np.digitize(df['input_4hourly'].fillna(0), self.iv_bins[1:-1], right=False)
        df['iv_bin'] = df['iv_bin'].clip(0, self.n_iv_bins - 1)

        # Discretize vasopressors
        df['vaso_bin'] = np.digitize(df['max_dose_vaso'].fillna(0), self.vaso_bins[1:-1], right=False)
        df['vaso_bin'] = df['vaso_bin'].clip(0, self.n_vaso_bins - 1)

        # Combine into single action index: action = iv_bin * n_vaso_bins + vaso_bin
        df['action'] = df['iv_bin'] * self.n_vaso_bins + df['vaso_bin']

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str):
        """Save action bins to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'iv_bins': self.iv_bins,
                'vaso_bins': self.vaso_bins,
                'n_iv_bins': self.n_iv_bins,
                'n_vaso_bins': self.n_vaso_bins
            }, f)
        logger.info(f"  ✓ Saved action bins to {path}")

    def get_action_description(self, action: int) -> str:
        """Get human-readable description of an action."""
        iv_bin = action // self.n_vaso_bins
        vaso_bin = action % self.n_vaso_bins
        return f"Action {action}: IV bin={iv_bin}, Vaso bin={vaso_bin}"


class SimplifiedRewardComputer:
    """
    Computes rewards based on SOFA score changes and mortality.

    Reward structure (from Komorowski et al.):
    - Terminal survival: +15
    - Terminal death: -15
    - SOFA decrease (improvement): +0.1 per point
    - SOFA increase (worsening): -0.25 per point
    """

    def __init__(
        self,
        terminal_survival: float = 15.0,
        terminal_death: float = -15.0,
        sofa_decrease_reward: float = 0.1,
        sofa_increase_penalty: float = -0.25
    ):
        self.terminal_survival = terminal_survival
        self.terminal_death = terminal_death
        self.sofa_decrease_reward = sofa_decrease_reward
        self.sofa_increase_penalty = sofa_increase_penalty

    def compute_rewards(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rewards for all transitions."""
        logger.info("Computing rewards...")

        df = df.copy()

        # Sort by stay and time
        df = df.sort_values(['stay_id', 'time_window']).reset_index(drop=True)

        # Compute SOFA change (current - previous)
        df['sofa_prev'] = df.groupby('stay_id')['SOFA'].shift(1)
        df['sofa_change'] = df['SOFA'] - df['sofa_prev']

        # Identify terminal states (last time window for each stay)
        df['is_terminal'] = False
        last_tw = df.groupby('stay_id')['time_window'].transform('max')
        df.loc[df['time_window'] == last_tw, 'is_terminal'] = True

        # Initialize rewards
        df['reward'] = 0.0

        # Terminal rewards
        terminal_survived = df['is_terminal'] & (df['hospital_expire_flag'] == 0)
        terminal_died = df['is_terminal'] & (df['hospital_expire_flag'] == 1)

        df.loc[terminal_survived, 'reward'] = self.terminal_survival
        df.loc[terminal_died, 'reward'] = self.terminal_death

        logger.info(f"  Terminal rewards: {terminal_survived.sum():,} survived, {terminal_died.sum():,} died")

        # Intermediate rewards (based on SOFA change)
        intermediate = ~df['is_terminal'] & df['sofa_prev'].notna()

        # SOFA decreased (improvement) -> positive reward
        improved = intermediate & (df['sofa_change'] < 0)
        df.loc[improved, 'reward'] = -df.loc[improved, 'sofa_change'] * self.sofa_decrease_reward

        # SOFA increased (worsening) -> negative reward
        worsened = intermediate & (df['sofa_change'] > 0)
        df.loc[worsened, 'reward'] = -df.loc[worsened, 'sofa_change'] * abs(self.sofa_increase_penalty)

        logger.info(f"  Intermediate: {improved.sum():,} improved, {worsened.sum():,} worsened, "
                   f"{(intermediate & (df['sofa_change'] == 0)).sum():,} no change")
        logger.info(f"  Reward range: [{df['reward'].min():.2f}, {df['reward'].max():.2f}]")
        logger.info(f"  Mean reward: {df['reward'].mean():.4f}")

        return df


def build_trajectories(df: pd.DataFrame, state_columns: list) -> pd.DataFrame:
    """
    Build complete (s, a, r, s', done) trajectories.

    Args:
        df: DataFrame with features, actions, and rewards
        state_columns: List of state feature column names

    Returns:
        DataFrame with next-state features added
    """
    logger.info("Building trajectories...")

    df = df.copy()
    df = df.sort_values(['stay_id', 'time_window']).reset_index(drop=True)

    # Create next-state columns
    logger.info(f"  Creating next-state features for {len(state_columns)} columns...")
    for col in state_columns:
        df[f'next_{col}'] = df.groupby('stay_id')[col].shift(-1)

    # Mark terminal states (done flag)
    df['done'] = df['is_terminal'].astype(int)

    logger.info(f"  ✓ Built trajectories: {len(df):,} transitions")
    logger.info(f"  Terminal states: {df['done'].sum():,}")

    return df


def compute_returns(df: pd.DataFrame, gamma: float = 0.99) -> pd.DataFrame:
    """Compute discounted returns for each state."""
    logger.info(f"Computing discounted returns (γ={gamma})...")

    df = df.copy()
    df = df.sort_values(['stay_id', 'time_window']).reset_index(drop=True)

    def compute_stay_returns(stay_df):
        rewards = stay_df['reward'].values
        n = len(rewards)
        returns = np.zeros(n)

        # Backward pass
        returns[-1] = rewards[-1]
        for t in range(n - 2, -1, -1):
            returns[t] = rewards[t] + gamma * returns[t + 1]

        stay_df = stay_df.copy()
        stay_df['return'] = returns
        return stay_df

    df = df.groupby('stay_id', group_keys=False).apply(compute_stay_returns)

    logger.info(f"  Return range: [{df['return'].min():.2f}, {df['return'].max():.2f}]")
    logger.info(f"  Mean return: {df['return'].mean():.2f}")

    return df


def main():
    """Run the complete MDP pipeline."""
    logger.info("="*80)
    logger.info("MDP PIPELINE: ACTIONS, REWARDS, TRAJECTORIES")
    logger.info("="*80)

    # Paths
    processed_dir = Path("data/processed")
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Load config
    config = ConfigLoader("configs/config.yaml").config
    gamma = config.get('mdp', {}).get('gamma', 0.99)

    # Load data
    logger.info("\n" + "="*40)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*40)

    train = pd.read_csv(processed_dir / "train_features.csv")
    val = pd.read_csv(processed_dir / "val_features.csv")
    test = pd.read_csv(processed_dir / "test_features.csv")

    logger.info(f"  Train: {len(train):,} observations, {train['stay_id'].nunique():,} stays")
    logger.info(f"  Val: {len(val):,} observations, {val['stay_id'].nunique():,} stays")
    logger.info(f"  Test: {len(test):,} observations, {test['stay_id'].nunique():,} stays")

    # Define state columns (exclude identifiers, actions, and outcome)
    exclude_cols = ['stay_id', 'time_window', 'input_4hourly', 'input_total',
                    'output_4hourly', 'output_total', 'cumulated_balance',
                    'max_dose_vaso', 'hospital_expire_flag']
    state_columns = [col for col in train.columns if col not in exclude_cols]
    logger.info(f"  State features: {len(state_columns)}")

    # ============================================================
    # STEP 2: DISCRETIZE ACTIONS
    # ============================================================
    logger.info("\n" + "="*40)
    logger.info("STEP 2: DISCRETIZING ACTIONS")
    logger.info("="*40)

    action_extractor = SimplifiedActionExtractor(n_iv_bins=5, n_vaso_bins=5)

    # Fit on training data, transform all
    train = action_extractor.fit_transform(train)
    val = action_extractor.transform(val)
    test = action_extractor.transform(test)

    # Save action bins
    action_extractor.save(str(processed_dir / "action_bins.pkl"))

    # Log action distribution
    logger.info("\nAction distribution (Train):")
    action_counts = train['action'].value_counts().sort_index()
    for action_id in range(min(10, len(action_counts))):
        if action_id in action_counts.index:
            count = action_counts[action_id]
            iv_bin = action_id // 5
            vaso_bin = action_id % 5
            pct = count / len(train) * 100
            logger.info(f"  Action {action_id:2d} (IV={iv_bin}, Vaso={vaso_bin}): {count:>8,} ({pct:>5.1f}%)")
    logger.info(f"  ... total {len(action_counts)} unique actions")

    # ============================================================
    # STEP 3: COMPUTE REWARDS
    # ============================================================
    logger.info("\n" + "="*40)
    logger.info("STEP 3: COMPUTING REWARDS")
    logger.info("="*40)

    reward_config = config.get('reward', {})
    reward_computer = SimplifiedRewardComputer(
        terminal_survival=reward_config.get('terminal_survival', 15),
        terminal_death=reward_config.get('terminal_death', -15),
        sofa_decrease_reward=reward_config.get('intermediate_sofa_decrease', 0.1),
        sofa_increase_penalty=reward_config.get('intermediate_sofa_increase', -0.25)
    )

    train = reward_computer.compute_rewards(train)
    val = reward_computer.compute_rewards(val)
    test = reward_computer.compute_rewards(test)

    # ============================================================
    # STEP 4: BUILD TRAJECTORIES
    # ============================================================
    logger.info("\n" + "="*40)
    logger.info("STEP 4: BUILDING TRAJECTORIES")
    logger.info("="*40)

    train = build_trajectories(train, state_columns)
    val = build_trajectories(val, state_columns)
    test = build_trajectories(test, state_columns)

    # ============================================================
    # STEP 5: COMPUTE RETURNS
    # ============================================================
    logger.info("\n" + "="*40)
    logger.info("STEP 5: COMPUTING RETURNS")
    logger.info("="*40)

    train = compute_returns(train, gamma=gamma)
    val = compute_returns(val, gamma=gamma)
    test = compute_returns(test, gamma=gamma)

    # ============================================================
    # STEP 6: SAVE TRAJECTORIES
    # ============================================================
    logger.info("\n" + "="*40)
    logger.info("STEP 6: SAVING TRAJECTORIES")
    logger.info("="*40)

    train.to_csv(processed_dir / "train_trajectories.csv", index=False)
    val.to_csv(processed_dir / "val_trajectories.csv", index=False)
    test.to_csv(processed_dir / "test_trajectories.csv", index=False)

    logger.info(f"  ✓ Saved train_trajectories.csv ({len(train):,} rows)")
    logger.info(f"  ✓ Saved val_trajectories.csv ({len(val):,} rows)")
    logger.info(f"  ✓ Saved test_trajectories.csv ({len(test):,} rows)")

    # ============================================================
    # SUMMARY STATISTICS
    # ============================================================
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info("="*80)

    for name, df in [('Train', train), ('Val', val), ('Test', test)]:
        n_stays = df['stay_id'].nunique()
        n_transitions = len(df)
        avg_traj_len = n_transitions / n_stays
        mortality_rate = df.groupby('stay_id')['hospital_expire_flag'].first().mean() * 100

        logger.info(f"\n{name}:")
        logger.info(f"  Patients: {n_stays:,}")
        logger.info(f"  Transitions: {n_transitions:,}")
        logger.info(f"  Avg trajectory length: {avg_traj_len:.1f}")
        logger.info(f"  Mortality rate: {mortality_rate:.1f}%")
        logger.info(f"  Mean reward: {df['reward'].mean():.4f}")
        logger.info(f"  Mean return: {df['return'].mean():.2f}")

    logger.info("\n" + "="*80)
    logger.info("MDP PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info("\nNext step: Run Q-learning training with:")
    logger.info("  python run_qlearning.py")


if __name__ == "__main__":
    main()
