"""
Configuration Loader
Loads and validates configuration from YAML file
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate configuration settings"""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader

        Args:
            config_path: Path to config.yaml file
        """
        if config_path is None:
            # Default to configs/config.yaml
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {self.config_path}")
        return config

    def _validate_config(self):
        """Validate required configuration sections exist"""
        required_sections = [
            'database',
            'cohort',
            'mdp',
            'state_features',
            'action_space',
            'reward',
            'data_split',
            'q_learning',
            'ope'
        ]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

        logger.info("Configuration validation passed")

    def get(self, key: str, default=None):
        """
        Get configuration value by key (supports nested keys with dots)

        Args:
            key: Configuration key (e.g., 'database.host')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.config['database']

    def get_mdp_params(self) -> Dict[str, Any]:
        """Get MDP parameters"""
        return self.config['mdp']

    def get_feature_list(self) -> list:
        """Get list of all state features (48 features)"""
        features = []
        state_config = self.config['state_features']

        for category in state_config.values():
            if isinstance(category, list):
                features.extend(category)

        return features

    def __repr__(self):
        return f"ConfigLoader(config_path={self.config_path})"


if __name__ == "__main__":
    # Test configuration loader
    logging.basicConfig(level=logging.INFO)
    config = ConfigLoader()
    print(f"Database host: {config.get('database.host')}")
    print(f"Gamma: {config.get('mdp.gamma')}")
    print(f"Feature count: {len(config.get_feature_list())}")
