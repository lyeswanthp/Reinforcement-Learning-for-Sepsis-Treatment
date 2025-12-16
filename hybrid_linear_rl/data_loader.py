"""
Data loading with action simplification.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from config import Config

logger = logging.getLogger(__name__)


class ActionSimplifier:
    """Simplifies 5x5 to 3x3 action space."""

    def __init__(self, config: Config):
        self.config = config
        self.iv_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
        self.vaso_mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
        self.fitted = False

    def fit(self, iv_bins: np.ndarray, vaso_bins: np.ndarray):
        logger.info("Action simplifier: using fixed 5→3 mapping")
        logger.info(f"  IV mapping: {self.iv_mapping}")
        logger.info(f"  Vaso mapping: {self.vaso_mapping}")
        self.fitted = True

    def transform(self, iv_bins: np.ndarray, vaso_bins: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ActionSimplifier not fitted")

        iv_new = np.array([self.iv_mapping[int(b)] for b in iv_bins])
        vaso_new = np.array([self.vaso_mapping[int(b)] for b in vaso_bins])
        actions = iv_new * 3 + vaso_new
        return actions

    def get_action_components(self, action: int) -> Tuple[int, int]:
        return action // 3, action % 3


class DataLoader:
    """Load and prepare trajectory data."""

    def __init__(self, config: Config):
        self.config = config
        self.action_simplifier = ActionSimplifier(config)
        self.state_columns = None

    def load_all_splits(self) -> Dict[str, Dict]:
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
            'next_states': next_states
        }

    def _identify_state_columns(self, df: pd.DataFrame) -> list:
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
