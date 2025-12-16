"""
Data loading and action space simplification.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from config import Config

logger = logging.getLogger(__name__)


class ActionSimplifier:
    """Simplifies 5x5 action space to 3x3."""

    def __init__(self, config: Config):
        self.config = config
        self.iv_bin_mapping = None
        self.vaso_bin_mapping = None
        self.fitted = False

    def fit(self, iv_bins: np.ndarray, vaso_bins: np.ndarray):
        """
        Learn mapping from 5-bin to 3-bin space based on data distribution.

        Args:
            iv_bins: Original IV bin assignments (0-4)
            vaso_bins: Original vaso bin assignments (0-4)
        """
        logger.info("Fitting action simplifier...")

        self.iv_bin_mapping = self._compute_mapping(iv_bins, 'IV')
        self.vaso_bin_mapping = self._compute_mapping(vaso_bins, 'Vaso')

        self.fitted = True
        logger.info("Action simplifier fitted")

    def _compute_mapping(self, bins: np.ndarray, name: str) -> Dict[int, int]:
        """
        Map 5 bins to 3 bins: [0] -> 0 (None), [1,2] -> 1 (Low), [3,4] -> 2 (High)
        """
        mapping = {
            0: 0,
            1: 1,
            2: 1,
            3: 2,
            4: 2
        }

        counts = np.bincount(bins, minlength=5)
        logger.info(f"  {name} bin distribution (5-bin): {counts}")

        new_counts = np.zeros(3, dtype=int)
        for old_bin, new_bin in mapping.items():
            if old_bin < len(counts):
                new_counts[new_bin] += counts[old_bin]

        logger.info(f"  {name} bin distribution (3-bin): {new_counts}")
        return mapping

    def transform(self, iv_bins: np.ndarray, vaso_bins: np.ndarray) -> np.ndarray:
        """
        Transform 5x5 actions to 3x3 actions.

        Args:
            iv_bins: Original IV bins (0-4)
            vaso_bins: Original vaso bins (0-4)

        Returns:
            actions: Simplified action indices (0-8)
        """
        if not self.fitted:
            raise RuntimeError("ActionSimplifier not fitted")

        iv_new = np.array([self.iv_bin_mapping[int(b)] for b in iv_bins])
        vaso_new = np.array([self.vaso_bin_mapping[int(b)] for b in vaso_bins])

        actions = iv_new * 3 + vaso_new
        return actions

    def get_action_components(self, action: int) -> Tuple[int, int]:
        """
        Decompose action index into (IV bin, Vaso bin).

        Args:
            action: Action index (0-8)

        Returns:
            (iv_bin, vaso_bin): Both in range 0-2
        """
        iv_bin = action // 3
        vaso_bin = action % 3
        return iv_bin, vaso_bin


class DataLoader:
    """Load and preprocess trajectory data."""

    def __init__(self, config: Config):
        self.config = config
        self.action_simplifier = ActionSimplifier(config)
        self.state_columns = None

    def load_all_splits(self) -> Dict[str, Dict]:
        """
        Load train/val/test splits and simplify actions.

        Returns:
            Dictionary with 'train', 'val', 'test' data
        """
        logger.info("="*60)
        logger.info("LOADING DATA")
        logger.info("="*60)

        datasets = {}

        for split, path in [
            ('train', self.config.data.train_path),
            ('val', self.config.data.val_path),
            ('test', self.config.data.test_path)
        ]:
            logger.info(f"Loading {split} from {path}...")
            df = pd.read_csv(path)

            if split == 'train':
                data = self._prepare_split(df, fit_simplifier=True)
            else:
                data = self._prepare_split(df, fit_simplifier=False)

            datasets[split] = data

            logger.info(f"  {split}: {len(data['states']):,} transitions, "
                       f"{data['df']['stay_id'].nunique():,} trajectories")

        return datasets

    def _prepare_split(self, df: pd.DataFrame, fit_simplifier: bool) -> Dict:
        """
        Prepare a single data split.

        Args:
            df: Raw trajectory dataframe
            fit_simplifier: Whether to fit action simplifier

        Returns:
            Dictionary with states, actions, rewards, etc.
        """
        if self.state_columns is None:
            self.state_columns = self._identify_state_columns(df)

        states = df[self.state_columns].values.astype(np.float32)
        next_states = df[[f'next_{c}' for c in self.state_columns]].values.astype(np.float32)

        iv_bins = df['iv_bin'].values.astype(int)
        vaso_bins = df['vaso_bin'].values.astype(int)

        if fit_simplifier:
            self.action_simplifier.fit(iv_bins, vaso_bins)

        actions = self.action_simplifier.transform(iv_bins, vaso_bins)

        rewards = df['reward'].values.astype(np.float32)
        dones = df['done'].values.astype(bool)

        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        next_states = np.nan_to_num(next_states, nan=0.0, posinf=0.0, neginf=0.0)
        rewards = np.nan_to_num(rewards, nan=0.0)

        return {
            'df': df,
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'next_states': next_states,
            'iv_bins_original': iv_bins,
            'vaso_bins_original': vaso_bins
        }

    def _identify_state_columns(self, df: pd.DataFrame) -> list:
        """Identify state feature columns."""
        exclude = [
            'stay_id', 'time_window', 'action', 'reward', 'done',
            'is_terminal', 'hospital_expire_flag',
            'iv_bin', 'vaso_bin', 'sofa_prev', 'sofa_change',
            'input_4hourly', 'input_total', 'output_4hourly',
            'output_total', 'cumulated_balance', 'max_dose_vaso',
            'return'
        ]

        state_cols = [
            c for c in df.columns
            if c not in exclude and not c.startswith('next_')
        ]

        logger.info(f"Identified {len(state_cols)} state features")
        return state_cols
