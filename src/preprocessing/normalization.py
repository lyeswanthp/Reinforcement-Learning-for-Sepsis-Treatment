"""
Feature Normalization
Z-score normalization with log transforms for skewed features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """
    Normalize features using z-score (standardization)

    - Log transform for skewed features
    - Binary features: subtract 0.5
    - Continuous features: z-score normalization
    - Save scalers for inverse transform
    """

    def __init__(self, config: Dict):
        """
        Initialize feature normalizer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.norm_config = config.get('normalization', {})

        # Features to log-transform
        self.log_features = self.norm_config.get('continuous_features', {}).get(
            'log_transform', []
        )

        # Binary features
        self.binary_features = self.norm_config.get('binary_features', [])

        # Scalers (fitted on training data)
        self.scalers = {}
        self.fitted = False

        logger.info(f"FeatureNormalizer initialized")
        logger.info(f"  Log transform: {len(self.log_features)} features")
        logger.info(f"  Binary features: {len(self.binary_features)} features")

    def fit(
        self,
        data: pd.DataFrame,
        exclude_cols: List[str] = ['stay_id', 'time_window', 'subject_id', 'hadm_id']
    ) -> 'FeatureNormalizer':
        """
        Fit normalizers on training data

        Args:
            data: Training data
            exclude_cols: Columns to exclude from normalization

        Returns:
            self
        """
        logger.info("Fitting normalizers on training data...")

        # Get feature columns (exclude IDs)
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        # Step 1: Log transform
        logger.info(f"  Step 1: Preparing log transforms for {len(self.log_features)} features")
        self.log_transform_params = {}
        for feature in self.log_features:
            if feature in data.columns:
                # Add small constant to avoid log(0)
                min_val = data[feature].min()
                offset = 0.01 if min_val <= 0 else 0
                self.log_transform_params[feature] = offset

        # Step 2: Binary features
        logger.info(f"  Step 2: Identified {len(self.binary_features)} binary features")

        # Step 3: Fit scalers for continuous features
        continuous_features = [
            col for col in feature_cols
            if col not in self.binary_features and data[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]

        logger.info(f"  Step 3: Fitting scalers for {len(continuous_features)} continuous features")

        for feature in continuous_features:
            if feature in data.columns:
                # Apply log transform if needed
                if feature in self.log_features:
                    offset = self.log_transform_params.get(feature, 0)
                    feature_data = np.log(data[feature] + offset + 1e-8)
                else:
                    feature_data = data[feature]

                # Fit scaler
                scaler = StandardScaler()
                scaler.fit(feature_data.values.reshape(-1, 1))
                self.scalers[feature] = scaler

        self.fitted = True
        logger.info(f"✓ Fitted {len(self.scalers)} scalers")

        return self

    def transform(
        self,
        data: pd.DataFrame,
        exclude_cols: List[str] = ['stay_id', 'time_window', 'subject_id', 'hadm_id']
    ) -> pd.DataFrame:
        """
        Transform data using fitted normalizers

        Args:
            data: Data to transform
            exclude_cols: Columns to exclude from normalization

        Returns:
            Normalized data
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        logger.info("Transforming data...")

        data_norm = data.copy()

        # Transform binary features
        for feature in self.binary_features:
            if feature in data_norm.columns:
                data_norm[feature] = data_norm[feature] - 0.5

        # Transform continuous features
        for feature, scaler in self.scalers.items():
            if feature in data_norm.columns:
                # Apply log transform if needed
                if feature in self.log_features:
                    offset = self.log_transform_params.get(feature, 0)
                    feature_data = np.log(data_norm[feature] + offset + 1e-8)
                else:
                    feature_data = data_norm[feature]

                # Apply z-score normalization
                data_norm[feature] = scaler.transform(feature_data.values.reshape(-1, 1)).flatten()

        logger.info("✓ Data transformation complete")

        return data_norm

    def fit_transform(
        self,
        data: pd.DataFrame,
        exclude_cols: List[str] = ['stay_id', 'time_window', 'subject_id', 'hadm_id']
    ) -> pd.DataFrame:
        """
        Fit and transform in one step

        Args:
            data: Data to fit and transform
            exclude_cols: Columns to exclude from normalization

        Returns:
            Normalized data
        """
        self.fit(data, exclude_cols)
        return self.transform(data, exclude_cols)

    def inverse_transform(
        self,
        data: pd.DataFrame,
        exclude_cols: List[str] = ['stay_id', 'time_window', 'subject_id', 'hadm_id']
    ) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale

        Args:
            data: Normalized data
            exclude_cols: Columns to exclude from inverse transform

        Returns:
            Data in original scale
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        logger.info("Inverse transforming data...")

        data_orig = data.copy()

        # Inverse transform continuous features
        for feature, scaler in self.scalers.items():
            if feature in data_orig.columns:
                # Inverse z-score
                feature_data = scaler.inverse_transform(
                    data_orig[feature].values.reshape(-1, 1)
                ).flatten()

                # Inverse log transform if needed
                if feature in self.log_features:
                    offset = self.log_transform_params.get(feature, 0)
                    feature_data = np.exp(feature_data) - offset - 1e-8

                data_orig[feature] = feature_data

        # Inverse transform binary features
        for feature in self.binary_features:
            if feature in data_orig.columns:
                data_orig[feature] = data_orig[feature] + 0.5

        logger.info("✓ Inverse transformation complete")

        return data_orig

    def save(self, filepath: str):
        """
        Save fitted normalizer to disk

        Args:
            filepath: Path to save normalizer
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        normalizer_state = {
            'scalers': self.scalers,
            'log_transform_params': self.log_transform_params,
            'log_features': self.log_features,
            'binary_features': self.binary_features,
            'fitted': self.fitted
        }

        with open(save_path, 'wb') as f:
            pickle.dump(normalizer_state, f)

        logger.info(f"✓ Saved normalizer to {filepath}")

    def load(self, filepath: str):
        """
        Load fitted normalizer from disk

        Args:
            filepath: Path to load normalizer from
        """
        load_path = Path(filepath)

        if not load_path.exists():
            raise FileNotFoundError(f"Normalizer file not found: {filepath}")

        with open(load_path, 'rb') as f:
            normalizer_state = pickle.load(f)

        self.scalers = normalizer_state['scalers']
        self.log_transform_params = normalizer_state['log_transform_params']
        self.log_features = normalizer_state['log_features']
        self.binary_features = normalizer_state['binary_features']
        self.fitted = normalizer_state['fitted']

        logger.info(f"✓ Loaded normalizer from {filepath}")

    def get_normalization_summary(self) -> pd.DataFrame:
        """
        Get summary of normalization parameters

        Returns:
            DataFrame with normalization statistics
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        summary_data = []

        for feature, scaler in self.scalers.items():
            row = {
                'Feature': feature,
                'Type': 'Binary' if feature in self.binary_features else 'Continuous',
                'Log Transform': feature in self.log_features,
                'Mean': scaler.mean_[0],
                'Std': scaler.scale_[0]
            }
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        return summary_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Example usage
    from utils.config_loader import ConfigLoader

    config = ConfigLoader().config
    normalizer = FeatureNormalizer(config)

    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'stay_id': range(1000),
        'time_window': np.repeat(range(10), 100),
        'HR': np.random.normal(80, 15, 1000),
        'SysBP': np.random.normal(120, 20, 1000),
        'Temp_C': np.random.normal(37, 1, 1000),
        'SpO2': np.random.exponential(2, 1000),  # Skewed
        'gender': np.random.choice([0, 1], 1000),  # Binary
        'mechvent': np.random.choice([0, 1], 1000),  # Binary
    })

    print("\nBefore normalization:")
    print(sample_data[['HR', 'SysBP', 'Temp_C', 'SpO2', 'gender']].describe())

    # Fit and transform
    normalized_data = normalizer.fit_transform(sample_data)

    print("\nAfter normalization:")
    print(normalized_data[['HR', 'SysBP', 'Temp_C', 'SpO2', 'gender']].describe())

    # Inverse transform
    original_data = normalizer.inverse_transform(normalized_data)

    print("\nAfter inverse transform:")
    print(original_data[['HR', 'SysBP', 'Temp_C', 'SpO2', 'gender']].describe())

    # Get summary
    print("\nNormalization Summary:")
    print(normalizer.get_normalization_summary())
