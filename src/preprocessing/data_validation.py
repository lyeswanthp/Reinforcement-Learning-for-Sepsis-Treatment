"""
Data Validation Module
Validates data quality and checks for issues
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validate data quality throughout the preprocessing pipeline

    Checks:
    - Missing values
    - Data types
    - Value ranges
    - Duplicates
    - Temporal consistency
    - Feature completeness
    """

    def __init__(self, config: Dict):
        """
        Initialize data validator

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.expected_features = self._get_expected_features()

        logger.info(f"DataValidator initialized")
        logger.info(f"  Expected features: {len(self.expected_features)}")

    def _get_expected_features(self) -> List[str]:
        """Get list of expected state features from config"""
        expected = []
        state_config = self.config.get('state_features', {})

        for category in state_config.values():
            if isinstance(category, list):
                expected.extend(category)

        return expected

    def validate_cohort(
        self,
        cohort: pd.DataFrame,
        min_stays: int = 100
    ) -> Tuple[bool, Dict]:
        """
        Validate cohort selection

        Args:
            cohort: Cohort dataframe
            min_stays: Minimum number of ICU stays required

        Returns:
            Tuple of (is_valid, validation_report)
        """
        logger.info("Validating cohort...")

        report = {
            'total_stays': len(cohort),
            'total_patients': cohort['subject_id'].nunique(),
            'total_admissions': cohort['hadm_id'].nunique(),
            'issues': []
        }

        is_valid = True

        # Check 1: Sufficient sample size
        if len(cohort) < min_stays:
            report['issues'].append(f"Insufficient sample size: {len(cohort)} < {min_stays}")
            is_valid = False

        # Check 2: Required columns present
        required_cols = ['stay_id', 'subject_id', 'hadm_id', 'intime', 'outtime']
        missing_cols = [col for col in required_cols if col not in cohort.columns]
        if missing_cols:
            report['issues'].append(f"Missing required columns: {missing_cols}")
            is_valid = False

        # Check 3: No duplicate stay_ids
        if cohort['stay_id'].duplicated().any():
            n_duplicates = cohort['stay_id'].duplicated().sum()
            report['issues'].append(f"Duplicate stay_ids: {n_duplicates}")
            is_valid = False

        # Check 4: Valid time ranges
        if 'intime' in cohort.columns and 'outtime' in cohort.columns:
            invalid_times = cohort[cohort['outtime'] <= cohort['intime']]
            if len(invalid_times) > 0:
                report['issues'].append(f"Invalid time ranges: {len(invalid_times)} stays")
                is_valid = False

        # Check 5: Age distribution
        if 'anchor_age' in cohort.columns:
            report['age_mean'] = cohort['anchor_age'].mean()
            report['age_median'] = cohort['anchor_age'].median()
            report['age_std'] = cohort['anchor_age'].std()

        if is_valid:
            logger.info("✓ Cohort validation passed")
        else:
            logger.warning(f"✗ Cohort validation failed: {len(report['issues'])} issues")
            for issue in report['issues']:
                logger.warning(f"  - {issue}")

        return is_valid, report

    def validate_features(
        self,
        data: pd.DataFrame,
        stage: str = "extraction"
    ) -> Tuple[bool, Dict]:
        """
        Validate feature data

        Args:
            data: Feature dataframe
            stage: Pipeline stage (extraction, cleaning, normalization)

        Returns:
            Tuple of (is_valid, validation_report)
        """
        logger.info(f"Validating features ({stage} stage)...")

        report = {
            'stage': stage,
            'n_rows': len(data),
            'n_columns': len(data.columns),
            'missing_features': [],
            'extra_features': [],
            'missing_values': {},
            'infinite_values': {},
            'issues': []
        }

        is_valid = True

        # Check 1: Required columns
        id_cols = ['stay_id', 'time_window']
        for col in id_cols:
            if col not in data.columns:
                report['issues'].append(f"Missing required column: {col}")
                is_valid = False

        # Check 2: Feature completeness (after extraction)
        if stage == "extraction":
            feature_cols = [col for col in data.columns if col not in ['stay_id', 'time_window', 'subject_id', 'hadm_id']]
            missing_features = [f for f in self.expected_features if f not in feature_cols]
            extra_features = [f for f in feature_cols if f not in self.expected_features and f not in ['num_diagnoses', 'sofa_coag', 'sofa_liver', 'sofa_renal']]

            report['missing_features'] = missing_features
            report['extra_features'] = extra_features

            if missing_features:
                logger.info(f"  Note: {len(missing_features)} expected features not extracted (may be normal)")
                logger.debug(f"    Missing: {missing_features}")

        # Check 3: Missing values
        missing_counts = data.isnull().sum()
        missing_features = missing_counts[missing_counts > 0]

        if len(missing_features) > 0:
            report['missing_values'] = missing_features.to_dict()
            total_missing = missing_counts.sum()

            if stage == "normalization":
                # Should have no missing values after cleaning
                report['issues'].append(f"Missing values after cleaning: {total_missing}")
                is_valid = False
            else:
                logger.info(f"  Missing values: {total_missing:,} ({len(missing_features)} features)")

        # Check 4: Infinite values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            n_inf = np.isinf(data[col]).sum()
            if n_inf > 0:
                report['infinite_values'][col] = n_inf
                report['issues'].append(f"Infinite values in {col}: {n_inf}")
                is_valid = False

        # Check 5: Data type consistency
        for col in numeric_cols:
            if data[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                report['issues'].append(f"Unexpected data type for {col}: {data[col].dtype}")

        if is_valid:
            logger.info(f"✓ Feature validation passed ({stage})")
        else:
            logger.warning(f"✗ Feature validation failed ({stage}): {len(report['issues'])} issues")
            for issue in report['issues']:
                logger.warning(f"  - {issue}")

        return is_valid, report

    def validate_temporal_consistency(
        self,
        data: pd.DataFrame
    ) -> Tuple[bool, Dict]:
        """
        Validate temporal consistency (monotonic time windows, etc.)

        Args:
            data: Feature dataframe with time_window column

        Returns:
            Tuple of (is_valid, validation_report)
        """
        logger.info("Validating temporal consistency...")

        report = {
            'issues': []
        }

        is_valid = True

        if 'stay_id' not in data.columns or 'time_window' not in data.columns:
            report['issues'].append("Missing stay_id or time_window columns")
            return False, report

        # Check for gaps in time windows
        for stay_id in data['stay_id'].unique():
            stay_data = data[data['stay_id'] == stay_id].sort_values('time_window')
            time_windows = stay_data['time_window'].values

            # Check if monotonically increasing
            if not all(time_windows[i] <= time_windows[i+1] for i in range(len(time_windows)-1)):
                report['issues'].append(f"Non-monotonic time windows for stay {stay_id}")
                is_valid = False
                break

        if is_valid:
            logger.info("✓ Temporal consistency validated")
        else:
            logger.warning(f"✗ Temporal consistency issues: {len(report['issues'])} problems")

        return is_valid, report

    def generate_validation_report(
        self,
        cohort_report: Dict,
        extraction_report: Dict,
        cleaning_report: Dict,
        normalization_report: Dict
    ) -> str:
        """
        Generate comprehensive validation report

        Args:
            cohort_report: Cohort validation results
            extraction_report: Feature extraction validation results
            cleaning_report: Data cleaning validation results
            normalization_report: Normalization validation results

        Returns:
            Formatted validation report string
        """
        report = []
        report.append("="*80)
        report.append("DATA VALIDATION REPORT")
        report.append("="*80)

        # Cohort
        report.append("\n1. COHORT VALIDATION")
        report.append("-"*40)
        report.append(f"Total ICU stays:       {cohort_report['total_stays']:>10,}")
        report.append(f"Total patients:        {cohort_report['total_patients']:>10,}")
        report.append(f"Total admissions:      {cohort_report['total_admissions']:>10,}")
        if cohort_report['issues']:
            report.append(f"\nIssues found: {len(cohort_report['issues'])}")
            for issue in cohort_report['issues']:
                report.append(f"  - {issue}")
        else:
            report.append("\n✓ No issues found")

        # Feature Extraction
        report.append("\n2. FEATURE EXTRACTION VALIDATION")
        report.append("-"*40)
        report.append(f"Rows extracted:        {extraction_report['n_rows']:>10,}")
        report.append(f"Columns extracted:     {extraction_report['n_columns']:>10}")
        if extraction_report['missing_features']:
            report.append(f"Missing features:      {len(extraction_report['missing_features']):>10}")
        if extraction_report['issues']:
            report.append(f"\nIssues found: {len(extraction_report['issues'])}")
            for issue in extraction_report['issues']:
                report.append(f"  - {issue}")
        else:
            report.append("\n✓ No issues found")

        # Data Cleaning
        report.append("\n3. DATA CLEANING VALIDATION")
        report.append("-"*40)
        report.append(f"Rows after cleaning:   {cleaning_report['n_rows']:>10,}")
        report.append(f"Columns after cleaning:{cleaning_report['n_columns']:>10}")
        if cleaning_report['issues']:
            report.append(f"\nIssues found: {len(cleaning_report['issues'])}")
            for issue in cleaning_report['issues']:
                report.append(f"  - {issue}")
        else:
            report.append("\n✓ No issues found")

        # Normalization
        report.append("\n4. NORMALIZATION VALIDATION")
        report.append("-"*40)
        report.append(f"Rows normalized:       {normalization_report['n_rows']:>10,}")
        report.append(f"Columns normalized:    {normalization_report['n_columns']:>10}")
        if normalization_report['issues']:
            report.append(f"\nIssues found: {len(normalization_report['issues'])}")
            for issue in normalization_report['issues']:
                report.append(f"  - {issue}")
        else:
            report.append("\n✓ No issues found")

        report.append("\n" + "="*80)

        return "\n".join(report)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Example usage
    from utils.config_loader import ConfigLoader

    config = ConfigLoader().config
    validator = DataValidator(config)

    # Create sample cohort
    sample_cohort = pd.DataFrame({
        'stay_id': range(1, 501),
        'subject_id': range(1, 501),
        'hadm_id': range(1, 501),
        'intime': pd.date_range('2020-01-01', periods=500, freq='H'),
        'outtime': pd.date_range('2020-01-02', periods=500, freq='H'),
        'anchor_age': np.random.normal(60, 15, 500)
    })

    # Validate cohort
    is_valid, report = validator.validate_cohort(sample_cohort, min_stays=100)
    print(f"\nCohort valid: {is_valid}")
    print(f"Report: {report}")

    # Create sample features
    sample_features = pd.DataFrame({
        'stay_id': np.repeat(range(1, 101), 10),
        'time_window': np.tile(range(10), 100),
        'HR': np.random.normal(80, 15, 1000),
        'SysBP': np.random.normal(120, 20, 1000),
    })

    # Validate features
    is_valid, report = validator.validate_features(sample_features, stage="extraction")
    print(f"\nFeatures valid: {is_valid}")
    print(f"Report: {report}")
