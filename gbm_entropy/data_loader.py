"""Data loading and action simplification."""

import numpy as np
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ActionSimplifier:
    """Maps 5-bin actions to 3-bin actions."""

    def __init__(self, n_iv_bins: int = 3, n_vaso_bins: int = 3):
        self.n_iv_bins = n_iv_bins
        self.n_vaso_bins = n_vaso_bins
        self.n_actions = n_iv_bins * n_vaso_bins
        self.iv_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
        self.vaso_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}

    def simplify(self, actions: np.ndarray) -> np.ndarray:
        """Convert 25-action space to 9-action space."""
        iv_bins = actions // 5
        vaso_bins = actions % 5
        iv_simplified = np.array([self.iv_mapping[iv] for iv in iv_bins])
        vaso_simplified = np.array([self.vaso_mapping[v] for v in vaso_bins])
        return iv_simplified * self.n_vaso_bins + vaso_simplified

    def get_action_components(self, action: int) -> tuple:
        """Get IV and vaso components of simplified action."""
        return action // self.n_vaso_bins, action % self.n_vaso_bins


class DataLoader:
    """Load and process data."""

    def __init__(self, config):
        self.config = config
        self.action_simplifier = ActionSimplifier(
            config.action.n_iv_bins,
            config.action.n_vaso_bins
        )
        self.state_columns = None

    def _load_and_process(self, path: str) -> Dict:
        """Load and process single dataset."""
        df = pd.read_csv(path)

        action_col = 'action' if 'action' in df.columns else 'action_discrete'
        reward_col = 'reward' if 'reward' in df.columns else 'reward_terminal'
        done_col = 'done' if 'done' in df.columns else 'terminal'

        exclude_cols = [
            'stay_id', 'time_window', 'action', 'action_discrete',
            'reward', 'reward_terminal', 'done', 'terminal',
            'admittime', 'intime', 'outtime', 'mortality_90d'
        ]

        if self.state_columns is None:
            self.state_columns = [
                c for c in df.columns
                if c not in exclude_cols and not c.startswith('next_')
            ]

        states = df[self.state_columns].values.astype(np.float32)
        states = np.nan_to_num(states, nan=0.0)

        actions_orig = df[action_col].values.astype(np.int32)
        actions = self.action_simplifier.simplify(actions_orig)

        rewards = df[reward_col].values.astype(np.float32)
        dones = df[done_col].values.astype(bool)

        next_cols = [f'next_{c}' for c in self.state_columns]
        if all(c in df.columns for c in next_cols):
            next_states = df[next_cols].values.astype(np.float32)
        else:
            next_states = np.roll(states, -1, axis=0)
            for stay_id in df['stay_id'].unique():
                mask = (df['stay_id'] == stay_id).values
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    next_states[indices[-1]] = states[indices[-1]]

        next_states = np.nan_to_num(next_states, nan=0.0)

        return {
            'df': df,
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'next_states': next_states
        }

    def load_all_splits(self) -> Dict[str, Dict]:
        """Load all data splits."""
        logger.info("=" * 60)
        logger.info("LOADING DATA")
        logger.info("=" * 60)

        datasets = {}
        for split_name, path in [
            ('train', self.config.data.train_path),
            ('val', self.config.data.val_path),
            ('test', self.config.data.test_path)
        ]:
            logger.info(f"Loading {split_name} split...")
            data = self._load_and_process(path)
            n_traj = data['df']['stay_id'].nunique()
            logger.info(f"  {len(data['states']):,} transitions, {n_traj:,} trajectories")
            datasets[split_name] = data

        logger.info(f"State features: {len(self.state_columns)}")
        logger.info(f"Actions: {self.config.action.n_actions}")

        return datasets
