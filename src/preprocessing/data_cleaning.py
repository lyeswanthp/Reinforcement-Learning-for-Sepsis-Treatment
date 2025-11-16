"""
Data Cleaning and Imputation
Handles missing values, outliers, and data quality issues
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clean and impute missing values in MIMIC-IV features

    Strategies:
    - Forward fill for time-series data
    - Median imputation for static features
    - KNN imputation for complex patterns
    - Outlier detection and handling
    """

    # Physiological ranges for outlier detection
    VALID_RANGES = {
        # Vital signs
        'HR': (0, 250),
        'SysBP': (40, 250),
        'MeanBP': (30, 200),
        'DiaBP': (20, 180),
        'RR': (0, 80),
        'Temp_C': (25, 45),
        'SpO2': (0, 100),
        'FiO2_1': (21, 100),
        'GCS': (3, 15),

        # Labs - Chemistry
        'Potassium': (1.5, 10),
        'Sodium': (100, 180),
        'Chloride': (60, 140),
        'Glucose': (20, 1000),
        'BUN': (1, 300),
        'Creatinine': (0.1, 25),
        'Magnesium': (0.5, 5),
        'Calcium': (4, 16),
        'SGOT': (1, 10000),
        'SGPT': (1, 10000),
        'Total_bili': (0.1, 50),

        # Labs - Hematology
        'Hb': (2, 25),
        'WBC_count': (0, 500),
        'Platelets_count': (1, 2000),
        'PTT': (10, 200),
        'PT': (5, 100),
        'INR': (0.5, 20),

        # Labs - Blood Gas
        'Arterial_pH': (6.5, 8.0),
        'paO2': (20, 700),
        'paCO2': (10, 200),
        'Arterial_BE': (-30, 30),
        'HCO3': (5, 60),
        'Arterial_lactate': (0.1, 30),

        # Fluid balance
        'input_total': (0, 100000),
        'input_4hourly': (0, 20000),
        'output_total': (0, 50000),
        'output_4hourly': (0, 10000),

        # Derived
        'SOFA': (0, 24),
        'SIRS': (0, 4),
        'Shock_Index': (0, 10),
        'PaO2_FiO2': (0, 1000),
        'cumulated_balance': (-50000, 100000),

        # Demographics
        'age': (18, 120),
        'Weight_kg': (20, 300),
    }

    def __init__(self, config: Dict):
        """
        Initialize data cleaner

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})

        # Imputation strategy
        self.strategy = self.preprocessing_config.get('missing_data', {}).get(
            'strategy', 'forward_fill_then_median'
        )
        self.max_missing_ratio = self.preprocessing_config.get('missing_data', {}).get(
            'max_missing_ratio', 0.5
        )

        # Outlier detection
        self.outlier_method = self.preprocessing_config.get('outliers', {}).get(
            'method', 'iqr'
        )
        self.iqr_multiplier = self.preprocessing_config.get('outliers', {}).get(
            'iqr_multiplier', 3.0
        )
        self.zscore_threshold = self.preprocessing_config.get('outliers', {}).get(
            'zscore_threshold', 5.0
        )

        logger.info(f"DataCleaner initialized (strategy: {self.strategy}, "
                   f"outlier method: {self.outlier_method})")

    def clean_data(
        self,
        data: pd.DataFrame,
        temporal: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean data: handle outliers and missing values

        Args:
            data: Input dataframe
            temporal: If True, use forward fill for time-series data

        Returns:
            Tuple of (cleaned_data, cleaning_stats)
        """
        logger.info("="*80)
        logger.info("Starting Data Cleaning")
        logger.info("="*80)

        stats = {
            'initial_rows': len(data),
            'initial_columns': len(data.columns),
            'outliers_removed': 0,
            'features_dropped': 0,
            'missing_imputed': 0,
        }

        data_clean = data.copy()

        # Step 1: Remove outliers
        logger.info("\nStep 1: Detecting and removing outliers...")
        data_clean, outlier_stats = self._remove_outliers(data_clean)
        stats['outliers_removed'] = outlier_stats['total_outliers']

        # Step 2: Drop features with too many missing values
        logger.info(f"\nStep 2: Dropping features with >{self.max_missing_ratio*100}% missing...")
        data_clean, dropped_features = self._drop_high_missing_features(data_clean)
        stats['features_dropped'] = len(dropped_features)

        if dropped_features:
            logger.info(f"  Dropped features: {', '.join(dropped_features)}")

        # Step 3: Impute missing values
        logger.info("\nStep 3: Imputing missing values...")
        data_clean, imputation_stats = self._impute_missing(data_clean, temporal=temporal)
        stats['missing_imputed'] = imputation_stats['total_imputed']

        # Step 4: Final validation
        logger.info("\nStep 4: Final validation...")
        remaining_missing = data_clean.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"  Warning: {remaining_missing:,} missing values remain")
            stats['remaining_missing'] = remaining_missing
        else:
            logger.info("  ✓ No missing values remain")

        stats['final_rows'] = len(data_clean)
        stats['final_columns'] = len(data_clean.columns)

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("Data Cleaning Summary")
        logger.info("="*80)
        logger.info(f"Initial shape:          {stats['initial_rows']:>10,} rows × {stats['initial_columns']:>3} columns")
        logger.info(f"Outliers removed:       {stats['outliers_removed']:>10,} values")
        logger.info(f"Features dropped:       {stats['features_dropped']:>10,}")
        logger.info(f"Missing values imputed: {stats['missing_imputed']:>10,}")
        logger.info(f"Final shape:            {stats['final_rows']:>10,} rows × {stats['final_columns']:>3} columns")
        logger.info("="*80)

        return data_clean, stats

    def _remove_outliers(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Remove outliers based on physiological ranges and statistical methods

        Args:
            data: Input dataframe

        Returns:
            Tuple of (cleaned_data, outlier_stats)
        """
        data_clean = data.copy()
        total_outliers = 0

        # Method 1: Physiological ranges
        for feature, (min_val, max_val) in self.VALID_RANGES.items():
            if feature in data_clean.columns:
                before = data_clean[feature].notna().sum()

                # Replace outliers with NaN
                data_clean.loc[
                    (data_clean[feature] < min_val) | (data_clean[feature] > max_val),
                    feature
                ] = np.nan

                after = data_clean[feature].notna().sum()
                outliers_found = before - after

                if outliers_found > 0:
                    total_outliers += outliers_found
                    logger.debug(f"    {feature}: {outliers_found} outliers")

        logger.info(f"  ✓ Removed {total_outliers:,} outliers (physiological ranges)")

        # Method 2: Statistical outlier detection (IQR or Z-score)
        if self.outlier_method == 'iqr':
            data_clean, iqr_outliers = self._remove_outliers_iqr(data_clean)
            total_outliers += iqr_outliers
            logger.info(f"  ✓ Removed {iqr_outliers:,} outliers (IQR method)")
        elif self.outlier_method == 'zscore':
            data_clean, z_outliers = self._remove_outliers_zscore(data_clean)
            total_outliers += z_outliers
            logger.info(f"  ✓ Removed {z_outliers:,} outliers (Z-score method)")

        stats = {'total_outliers': total_outliers}
        return data_clean, stats

    def _remove_outliers_iqr(
        self,
        data: pd.DataFrame,
        exclude_cols: List[str] = ['stay_id', 'time_window', 'subject_id', 'hadm_id']
    ) -> Tuple[pd.DataFrame, int]:
        """Remove outliers using IQR method"""
        data_clean = data.copy()
        total_outliers = 0

        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        for col in numeric_cols:
            Q1 = data_clean[col].quantile(0.25)
            Q3 = data_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR

            before = data_clean[col].notna().sum()
            data_clean.loc[
                (data_clean[col] < lower_bound) | (data_clean[col] > upper_bound),
                col
            ] = np.nan
            after = data_clean[col].notna().sum()

            total_outliers += (before - after)

        return data_clean, total_outliers

    def _remove_outliers_zscore(
        self,
        data: pd.DataFrame,
        exclude_cols: List[str] = ['stay_id', 'time_window', 'subject_id', 'hadm_id']
    ) -> Tuple[pd.DataFrame, int]:
        """Remove outliers using Z-score method"""
        data_clean = data.copy()
        total_outliers = 0

        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        for col in numeric_cols:
            mean = data_clean[col].mean()
            std = data_clean[col].std()

            if std > 0:
                z_scores = np.abs((data_clean[col] - mean) / std)

                before = data_clean[col].notna().sum()
                data_clean.loc[z_scores > self.zscore_threshold, col] = np.nan
                after = data_clean[col].notna().sum()

                total_outliers += (before - after)

        return data_clean, total_outliers

    def _drop_high_missing_features(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Drop features with more than max_missing_ratio missing values

        Args:
            data: Input dataframe

        Returns:
            Tuple of (cleaned_data, dropped_features)
        """
        missing_ratios = data.isnull().sum() / len(data)
        high_missing_features = missing_ratios[missing_ratios > self.max_missing_ratio].index.tolist()

        # Don't drop ID columns
        id_cols = ['stay_id', 'time_window', 'subject_id', 'hadm_id']
        high_missing_features = [f for f in high_missing_features if f not in id_cols]

        if high_missing_features:
            data_clean = data.drop(columns=high_missing_features)
        else:
            data_clean = data.copy()

        return data_clean, high_missing_features

    def _impute_missing(
        self,
        data: pd.DataFrame,
        temporal: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Impute missing values using configured strategy

        Args:
            data: Input dataframe
            temporal: If True, use forward fill for time-series

        Returns:
            Tuple of (imputed_data, imputation_stats)
        """
        data_clean = data.copy()
        initial_missing = data_clean.isnull().sum().sum()

        # Separate ID columns
        id_cols = ['stay_id', 'time_window', 'subject_id', 'hadm_id']
        id_cols_present = [col for col in id_cols if col in data_clean.columns]

        if id_cols_present:
            ids = data_clean[id_cols_present]
            features = data_clean.drop(columns=id_cols_present)
        else:
            ids = None
            features = data_clean

        # Strategy 1: Forward fill (for temporal data)
        if temporal and 'stay_id' in data_clean.columns and 'time_window' in data_clean.columns:
            logger.info("  Using forward fill for temporal data...")
            features = features.groupby('stay_id').fillna(method='ffill')

        # Strategy 2: Median imputation
        if self.strategy in ['median', 'forward_fill_then_median']:
            logger.info("  Using median imputation...")
            numeric_cols = features.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if features[col].isnull().any():
                    median_val = features[col].median()
                    features[col].fillna(median_val, inplace=True)

        # Strategy 3: KNN imputation (more sophisticated)
        elif self.strategy == 'knn':
            logger.info("  Using KNN imputation...")
            numeric_cols = features.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                features[numeric_cols] = imputer.fit_transform(features[numeric_cols])

        # Combine back with IDs
        if ids is not None:
            data_clean = pd.concat([ids, features], axis=1)
        else:
            data_clean = features

        final_missing = data_clean.isnull().sum().sum()
        total_imputed = initial_missing - final_missing

        stats = {
            'initial_missing': initial_missing,
            'final_missing': final_missing,
            'total_imputed': total_imputed
        }

        logger.info(f"  ✓ Imputed {total_imputed:,} missing values")
        logger.info(f"    Remaining missing: {final_missing:,}")

        return data_clean, stats

    def get_missing_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of missing values

        Args:
            data: Input dataframe

        Returns:
            DataFrame with missing value statistics
        """
        missing_counts = data.isnull().sum()
        missing_ratios = missing_counts / len(data)

        summary = pd.DataFrame({
            'Feature': missing_counts.index,
            'Missing Count': missing_counts.values,
            'Missing Ratio': missing_ratios.values
        })

        summary = summary[summary['Missing Count'] > 0].sort_values(
            'Missing Ratio', ascending=False
        )

        return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Example usage
    from utils.config_loader import ConfigLoader

    config = ConfigLoader().config
    cleaner = DataCleaner(config)

    # Create sample data with outliers and missing values
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'stay_id': range(100),
        'time_window': np.repeat(range(10), 10),
        'HR': np.random.normal(80, 15, 100),
        'SysBP': np.random.normal(120, 20, 100),
        'Temp_C': np.random.normal(37, 1, 100),
    })

    # Add some outliers
    sample_data.loc[5, 'HR'] = 500  # Outlier
    sample_data.loc[10, 'SysBP'] = 300  # Outlier

    # Add some missing values
    sample_data.loc[15:20, 'Temp_C'] = np.nan

    print("\nBefore cleaning:")
    print(sample_data.describe())
    print(f"\nMissing values:\n{sample_data.isnull().sum()}")

    # Clean data
    cleaned_data, stats = cleaner.clean_data(sample_data, temporal=True)

    print("\nAfter cleaning:")
    print(cleaned_data.describe())
    print(f"\nMissing values:\n{cleaned_data.isnull().sum()}")
