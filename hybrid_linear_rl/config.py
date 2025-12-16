"""
Configuration for Hybrid Linear RL (Option 5).
Combines linear Q-learning and gradient boosting Q-learning.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class ActionConfig:
    n_iv_bins: int = 3
    n_vaso_bins: int = 3
    n_actions: int = 9


@dataclass
class LinearQLearningConfig:
    learning_rate: float = 0.001
    n_epochs: int = 500
    gamma: float = 0.99
    l2_lambda: float = 0.01
    batch_size: int = 1024
    patience: int = 30
    min_delta: float = 0.0001


@dataclass
class GradientBoostingConfig:
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    random_state: int = 42


@dataclass
class OPEConfig:
    n_bootstrap: int = 1000
    max_importance_weight: float = 100.0
    behavior_softening: float = 0.01


@dataclass
class DataConfig:
    base_dir: str = '/scratch/lpanch2/RL'
    processed_dir: str = 'data/processed'
    output_dir: str = 'hybrid_linear_rl/outputs'

    train_file: str = 'train_trajectories.csv'
    val_file: str = 'val_trajectories.csv'
    test_file: str = 'test_trajectories.csv'

    @property
    def train_path(self):
        return f'{self.base_dir}/{self.processed_dir}/{self.train_file}'

    @property
    def val_path(self):
        return f'{self.base_dir}/{self.processed_dir}/{self.val_file}'

    @property
    def test_path(self):
        return f'{self.base_dir}/{self.processed_dir}/{self.test_file}'

    @property
    def output_path(self):
        return f'{self.base_dir}/{self.output_dir}'


@dataclass
class Config:
    action: ActionConfig = None
    linear: LinearQLearningConfig = None
    gbm: GradientBoostingConfig = None
    ope: OPEConfig = None
    data: DataConfig = None
    random_seed: int = 42

    def __post_init__(self):
        if self.action is None:
            self.action = ActionConfig()
        if self.linear is None:
            self.linear = LinearQLearningConfig()
        if self.gbm is None:
            self.gbm = GradientBoostingConfig()
        if self.ope is None:
            self.ope = OPEConfig()
        if self.data is None:
            self.data = DataConfig()
