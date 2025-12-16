"""Data loading with temporal split and action simplification."""

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
        iv = action // self.n_vaso_bins
        vaso = action % self.n_vaso_bins
        return iv, vaso


class DataLoader:
    """Load and split data temporally."""

    def __init__(self, config):
        self.config = config
        self.action_simplifier = ActionSimplifier(
            config.action.n_iv_bins,
            config.action.n_vaso_bins
        )
        self.state_columns = None

    def _temporal_split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data by admission time."""
        if 'admittime' not in df.columns:
            logger.warning("No admittime column, using icustay_id for splitting")
            stay_ids = df['stay_id'].unique()
            np.random.shuffle(stay_ids)

            n_train = int(len(stay_ids) * self.config.data.train_quantile)
            n_val = int(len(stay_ids) * self.config.data.val_quantile)

            train_ids = stay_ids[:n_train]
            val_ids = stay_ids[n_train:n_val]
            test_ids = stay_ids[n_val:]

            return {
                'train': df[df['stay_id'].isin(train_ids)],
                'val': df[df['stay_id'].isin(val_ids)],
                'test': df[df['stay_id'].isin(test_ids)]
            }

        df['admittime'] = pd.to_datetime(df['admittime'])
        train_cutoff = df['admittime'].quantile(self.config.data.train_quantile)
        val_cutoff = df['admittime'].quantile(self.config.data.val_quantile)

        logger.info(f"Temporal split:")
        logger.info(f"  Train: admittime < {train_cutoff}")
        logger.info(f"  Val:   {train_cutoff} <= admittime < {val_cutoff}")
        logger.info(f"  Test:  admittime >= {val_cutoff}")

        return {
            'train': df[df['admittime'] < train_cutoff],
            'val': df[(df['admittime'] >= train_cutoff) & (df['admittime'] < val_cutoff)],
            'test': df[df['admittime'] >= val_cutoff]
        }

    def _load_and_process(self, df: pd.DataFrame) -> Dict:
        """Process single dataset."""
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

        bp_idx = self._get_feature_index('MeanBP')
        lac_idx = self._get_feature_index('Arterial_lactate')

        next_bp = next_states[:, bp_idx] if bp_idx is not None else np.zeros(len(states))
        next_lactate = next_states[:, lac_idx] if lac_idx is not None else np.zeros(len(states))

        return {
            'df': df,
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'next_states': next_states,
            'next_bp': next_bp,
            'next_lactate': next_lactate
        }

    def _get_feature_index(self, feature_name: str) -> int:
        """Get index of feature in state columns."""
        if feature_name in self.state_columns:
            return self.state_columns.index(feature_name)
        return None

    def load_all_splits(self) -> Dict[str, Dict]:
        """Load and split all data."""
        logger.info("=" * 60)
        logger.info("LOADING DATA")
        logger.info("=" * 60)

        if self.config.data.temporal_split:
            logger.info("Using temporal split")
            train_df = pd.read_csv(self.config.data.train_path)
            val_df = pd.read_csv(self.config.data.val_path)
            test_df = pd.read_csv(self.config.data.test_path)

            all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            splits = self._temporal_split(all_df)
        else:
            logger.info("Using existing split")
            splits = {
                'train': pd.read_csv(self.config.data.train_path),
                'val': pd.read_csv(self.config.data.val_path),
                'test': pd.read_csv(self.config.data.test_path)
            }

        datasets = {}
        for split_name, split_df in splits.items():
            logger.info(f"Processing {split_name} split...")
            data = self._load_and_process(split_df)
            n_traj = split_df['stay_id'].nunique()
            logger.info(f"  {len(data['states']):,} transitions, {n_traj:,} trajectories")
            datasets[split_name] = data

        logger.info(f"State features: {len(self.state_columns)}")
        logger.info(f"Actions: {self.config.action.n_actions}")

        return datasets
