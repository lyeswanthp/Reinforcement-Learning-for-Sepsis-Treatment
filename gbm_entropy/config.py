"""Configuration for GBM with entropy regularization."""

from dataclasses import dataclass


@dataclass
class DataConfig:
    train_path: str = '/scratch/lpanch2/RL/data/processed/train_trajectories.csv'
    val_path: str = '/scratch/lpanch2/RL/data/processed/val_trajectories.csv'
    test_path: str = '/scratch/lpanch2/RL/data/processed/test_trajectories.csv'
    output_path: str = '/scratch/lpanch2/RL/gbm_entropy/outputs'


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


@dataclass
class EntropyConfig:
    temperature: float = 2.0
    min_temperature: float = 0.5
    decay_rate: float = 0.95


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
    entropy: EntropyConfig = None
    ope: OPEConfig = None
    random_seed: int = 42

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.action is None:
            self.action = ActionConfig()
        if self.gbm is None:
            self.gbm = GBMConfig()
        if self.entropy is None:
            self.entropy = EntropyConfig()
        if self.ope is None:
            self.ope = OPEConfig()
