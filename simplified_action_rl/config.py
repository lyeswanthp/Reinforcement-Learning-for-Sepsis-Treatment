"""
Configuration for Simplified Action Space RL
"""
from dataclasses import dataclass
from typing import List


@dataclass
class ActionConfig:
    """Action space configuration."""
    n_iv_bins: int = 3
    n_vaso_bins: int = 3
    n_actions: int = 9

    iv_labels: List[str] = None
    vaso_labels: List[str] = None

    def __post_init__(self):
        if self.iv_labels is None:
            self.iv_labels = ['None', 'Low', 'High']
        if self.vaso_labels is None:
            self.vaso_labels = ['None', 'Low', 'High']


@dataclass
class ModelConfig:
    """Model training configuration."""
    hidden_dims: List[int] = None
    learning_rate: float = 1e-4
    batch_size: int = 256
    n_epochs: int = 100
    gamma: float = 0.99

    target_update_freq: int = 500
    soft_update_tau: float = 0.005
    use_soft_update: bool = True

    dropout_rate: float = 0.1
    gradient_clip: float = 10.0

    patience: int = 15
    min_delta: float = 0.001

    random_seed: int = 42
    device: str = 'cuda'

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]


@dataclass
class OPEConfig:
    """Off-policy evaluation configuration."""
    n_bootstrap: int = 1000
    max_importance_weight: float = 100.0
    behavior_softening: float = 0.01


@dataclass
class DataConfig:
    """Data paths configuration."""
    base_dir: str = '/scratch/lpanch2/RL'
    processed_dir: str = 'data/processed'
    output_dir: str = 'simplified_action_rl/outputs'

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
    """Main configuration."""
    action: ActionConfig = None
    model: ModelConfig = None
    ope: OPEConfig = None
    data: DataConfig = None

    def __post_init__(self):
        if self.action is None:
            self.action = ActionConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.ope is None:
            self.ope = OPEConfig()
        if self.data is None:
            self.data = DataConfig()
