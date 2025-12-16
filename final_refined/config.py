"""Configuration for refined approaches."""

from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    train_path: str = '/scratch/lpanch2/RL/data/processed/train_trajectories.csv'
    val_path: str = '/scratch/lpanch2/RL/data/processed/val_trajectories.csv'
    test_path: str = '/scratch/lpanch2/RL/data/processed/test_trajectories.csv'
    output_path: str = '/scratch/lpanch2/RL/final_refined/outputs'

    temporal_split: bool = True
    train_quantile: float = 0.70
    val_quantile: float = 0.85
    admittime_columns: List[str] = None

    def __post_init__(self):
        if self.admittime_columns is None:
            self.admittime_columns = ['admittime', 'intime', 'icu_intime']


@dataclass
class ActionConfig:
    n_iv_bins: int = 3
    n_vaso_bins: int = 3
    n_actions: int = 9


@dataclass
class GBMConfig:
    n_estimators: int = 150
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 10
    gamma: float = 0.99
    n_iterations: int = 5
    ensemble_size: int = 5

    state_noise_std: float = 0.15
    use_state_noise: bool = True


@dataclass
class IQLConfig:
    hidden_dims: List[int] = None
    dropout_rate: float = 0.1
    use_layer_norm: bool = True

    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 1024
    n_epochs: int = 300

    expectile: float = 0.7
    temperature: float = 3.0

    gamma: float = 0.99
    grad_clip: float = 1.0

    entropy_weight: float = 0.3
    bp_weight: float = 0.5
    lactate_weight: float = 0.5

    patience: int = 30
    min_delta: float = 0.0001

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]


@dataclass
class OPEConfig:
    max_importance_weight: float = 10.0
    behavior_softening: float = 0.01
    n_bootstrap: int = 500


@dataclass
class Config:
    data: DataConfig = None
    action: ActionConfig = None
    gbm: GBMConfig = None
    iql: IQLConfig = None
    ope: OPEConfig = None
    random_seed: int = 42

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.action is None:
            self.action = ActionConfig()
        if self.gbm is None:
            self.gbm = GBMConfig()
        if self.iql is None:
            self.iql = IQLConfig()
        if self.ope is None:
            self.ope = OPEConfig()
