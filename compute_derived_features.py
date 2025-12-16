#!/usr/bin/env python3
"""
Compute Derived Features Script

Adds SOFA score and other derived features to the preprocessed data.
This script should be run after preprocessing but before MDP construction.

Features computed:
1. SOFA score (5-component: missing GCS assumed normal = 0)
2. PaO2_FiO2 ratio (respiratory function)
3. Shock_Index (HR / SysBP)

Usage:
    python compute_derived_features.py

Author: AI Clinician Project
Date: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/derived_features.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DerivedFeatureComputer:
    """
    Computes derived clinical features from raw preprocessed data.

    Features:
    - SOFA: Sequential Organ Failure Assessment (0-24, computed from 5/6 components)
    - PaO2_FiO2: Oxygenation ratio (respiratory function indicator)
    - Shock_Index: HR/SysBP (hemodynamic instability indicator)
    """

    def __init__(self):
        """Initialize the derived feature computer."""
        logger.info("DerivedFeatureComputer initialized")

    def compute_all_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all derived features for a dataframe.

        Args:
            df: DataFrame with preprocessed clinical features

        Returns:
            DataFrame with added derived features
        """
        logger.info(f"Computing derived features for {len(df):,} observations...")

        result = df.copy()

        # 1. Compute PaO2/FiO2 ratio (needed for SOFA respiration)
        result = self._compute_pao2_fio2(result)

        # 2. Compute Shock Index
        result = self._compute_shock_index(result)

        # 3. Compute SOFA score (using 5 components, GCS assumed normal)
        result = self._compute_sofa(result)

        logger.info(f"Derived features computed successfully")
        return result

    def _compute_pao2_fio2(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute PaO2/FiO2 ratio (P/F ratio).

        Normal: > 400
        Mild ARDS: 200-300
        Moderate ARDS: 100-200
        Severe ARDS: < 100

        Args:
            df: DataFrame with paO2 and FiO2_1 columns

        Returns:
            DataFrame with PaO2_FiO2 column added
        """
        if 'paO2' not in df.columns or 'FiO2_1' not in df.columns:
            logger.warning("Missing paO2 or FiO2_1 - cannot compute PaO2_FiO2")
            df['PaO2_FiO2'] = np.nan
            return df

        # FiO2 is stored as percentage (21-100), need to convert to fraction
        # PaO2/FiO2 = paO2 / (FiO2/100) for FiO2 in percentage
        # Or simply PaO2 * 100 / FiO2

        # Handle edge cases
        fio2 = df['FiO2_1'].copy()
        fio2 = fio2.replace(0, np.nan)  # Avoid division by zero

        # If FiO2 > 1, it's likely stored as percentage (21-100)
        # If FiO2 <= 1, it's stored as fraction (0.21-1.0)
        # Based on our data, FiO2 is likely 21-100 range

        # Compute ratio
        df['PaO2_FiO2'] = df['paO2'] / (fio2 / 100)

        # Clip to physiological range (0-700)
        df['PaO2_FiO2'] = df['PaO2_FiO2'].clip(0, 700)

        # Fill any NaN with median (for cases where FiO2 was 0 or missing)
        median_pf = df['PaO2_FiO2'].median()
        df['PaO2_FiO2'] = df['PaO2_FiO2'].fillna(median_pf)

        logger.info(f"  PaO2_FiO2 computed: mean={df['PaO2_FiO2'].mean():.1f}, "
                   f"median={df['PaO2_FiO2'].median():.1f}, "
                   f"range=[{df['PaO2_FiO2'].min():.1f}, {df['PaO2_FiO2'].max():.1f}]")

        return df

    def _compute_shock_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Shock Index (SI = HR / SysBP).

        Normal: 0.5-0.7
        Elevated: > 0.9 (suggests hemodynamic instability)
        Severe: > 1.0 (high risk)

        Args:
            df: DataFrame with HR and SysBP columns

        Returns:
            DataFrame with Shock_Index column added
        """
        if 'HR' not in df.columns or 'SysBP' not in df.columns:
            logger.warning("Missing HR or SysBP - cannot compute Shock_Index")
            df['Shock_Index'] = np.nan
            return df

        # Avoid division by zero
        sysbp = df['SysBP'].copy()
        sysbp = sysbp.replace(0, np.nan)

        df['Shock_Index'] = df['HR'] / sysbp

        # Clip to physiological range (0-5)
        df['Shock_Index'] = df['Shock_Index'].clip(0, 5)

        # Fill any NaN with median
        median_si = df['Shock_Index'].median()
        df['Shock_Index'] = df['Shock_Index'].fillna(median_si)

        logger.info(f"  Shock_Index computed: mean={df['Shock_Index'].mean():.2f}, "
                   f"median={df['Shock_Index'].median():.2f}, "
                   f"range=[{df['Shock_Index'].min():.2f}, {df['Shock_Index'].max():.2f}]")

        return df

    def _compute_sofa(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SOFA score from available components.

        SOFA Components (0-4 points each):
        1. Respiration: PaO2/FiO2 ratio [AVAILABLE]
        2. Coagulation: Platelets [AVAILABLE]
        3. Liver: Bilirubin [AVAILABLE]
        4. Cardiovascular: MAP + vasopressors [AVAILABLE]
        5. CNS: GCS [MISSING - assume normal = 0 points]
        6. Renal: Creatinine [AVAILABLE]

        Max score with 5 components: 20 (instead of 24)

        Args:
            df: DataFrame with clinical features

        Returns:
            DataFrame with SOFA and component scores added
        """
        logger.info("  Computing SOFA score (5 components, GCS assumed normal)...")

        # Initialize component scores
        df['sofa_respiration'] = 0
        df['sofa_coagulation'] = 0
        df['sofa_liver'] = 0
        df['sofa_cardiovascular'] = 0
        df['sofa_cns'] = 0  # GCS missing - assume normal (score = 0)
        df['sofa_renal'] = 0

        # 1. Respiration (PaO2/FiO2)
        if 'PaO2_FiO2' in df.columns:
            df['sofa_respiration'] = df['PaO2_FiO2'].apply(self._sofa_respiration)
            logger.info(f"    Respiration: computed from PaO2_FiO2")
        else:
            logger.warning("    Respiration: PaO2_FiO2 not available, score = 0")

        # 2. Coagulation (Platelets)
        if 'Platelets_count' in df.columns:
            df['sofa_coagulation'] = df['Platelets_count'].apply(self._sofa_coagulation)
            logger.info(f"    Coagulation: computed from Platelets_count")
        else:
            logger.warning("    Coagulation: Platelets_count not available, score = 0")

        # 3. Liver (Bilirubin)
        if 'Total_bili' in df.columns:
            df['sofa_liver'] = df['Total_bili'].apply(self._sofa_liver)
            logger.info(f"    Liver: computed from Total_bili")
        else:
            logger.warning("    Liver: Total_bili not available, score = 0")

        # 4. Cardiovascular (MAP + vasopressors)
        if 'MeanBP' in df.columns or 'max_dose_vaso' in df.columns:
            df['sofa_cardiovascular'] = df.apply(self._sofa_cardiovascular, axis=1)
            logger.info(f"    Cardiovascular: computed from MeanBP + max_dose_vaso")
        else:
            logger.warning("    Cardiovascular: MeanBP/max_dose_vaso not available, score = 0")

        # 5. CNS (GCS) - MISSING, assume normal
        logger.info(f"    CNS: GCS not available, assuming normal (score = 0)")

        # 6. Renal (Creatinine)
        if 'Creatinine' in df.columns:
            df['sofa_renal'] = df['Creatinine'].apply(self._sofa_renal)
            logger.info(f"    Renal: computed from Creatinine")
        else:
            logger.warning("    Renal: Creatinine not available, score = 0")

        # Total SOFA score
        df['SOFA'] = (
            df['sofa_respiration'] +
            df['sofa_coagulation'] +
            df['sofa_liver'] +
            df['sofa_cardiovascular'] +
            df['sofa_cns'] +
            df['sofa_renal']
        )

        logger.info(f"  SOFA computed: mean={df['SOFA'].mean():.2f}, "
                   f"median={df['SOFA'].median():.1f}, "
                   f"range=[{df['SOFA'].min():.0f}, {df['SOFA'].max():.0f}]")

        # Component statistics
        logger.info(f"    Component means: resp={df['sofa_respiration'].mean():.2f}, "
                   f"coag={df['sofa_coagulation'].mean():.2f}, "
                   f"liver={df['sofa_liver'].mean():.2f}, "
                   f"cardio={df['sofa_cardiovascular'].mean():.2f}, "
                   f"renal={df['sofa_renal'].mean():.2f}")

        return df

    def _sofa_respiration(self, pao2_fio2: float) -> int:
        """SOFA respiration component (PaO2/FiO2 ratio)."""
        if pd.isna(pao2_fio2):
            return 0
        if pao2_fio2 >= 400:
            return 0
        elif pao2_fio2 >= 300:
            return 1
        elif pao2_fio2 >= 200:
            return 2
        elif pao2_fio2 >= 100:
            return 3
        else:
            return 4

    def _sofa_coagulation(self, platelets: float) -> int:
        """SOFA coagulation component (platelets x 10^3/uL)."""
        if pd.isna(platelets):
            return 0
        if platelets >= 150:
            return 0
        elif platelets >= 100:
            return 1
        elif platelets >= 50:
            return 2
        elif platelets >= 20:
            return 3
        else:
            return 4

    def _sofa_liver(self, bilirubin: float) -> int:
        """SOFA liver component (bilirubin mg/dL)."""
        if pd.isna(bilirubin):
            return 0
        if bilirubin < 1.2:
            return 0
        elif bilirubin < 2.0:
            return 1
        elif bilirubin < 6.0:
            return 2
        elif bilirubin < 12.0:
            return 3
        else:
            return 4

    def _sofa_cardiovascular(self, row: pd.Series) -> int:
        """
        SOFA cardiovascular component (MAP + vasopressors).

        Scoring:
        0: MAP >= 70 mmHg, no vasopressors
        1: MAP < 70 mmHg
        2: Low dose vasopressors
        3: Norepinephrine <= 0.1 mcg/kg/min
        4: Norepinephrine > 0.1 mcg/kg/min
        """
        mean_bp = row.get('MeanBP', np.nan)
        vaso_dose = row.get('max_dose_vaso', 0)

        # Vasopressor dose takes precedence
        if pd.notna(vaso_dose) and vaso_dose > 0:
            if vaso_dose > 0.1:
                return 4
            elif vaso_dose > 0:
                return 3

        # No vasopressors - check MAP
        if pd.notna(mean_bp):
            if mean_bp < 70:
                return 1
            else:
                return 0

        return 0

    def _sofa_renal(self, creatinine: float) -> int:
        """SOFA renal component (creatinine mg/dL)."""
        if pd.isna(creatinine):
            return 0
        if creatinine < 1.2:
            return 0
        elif creatinine < 2.0:
            return 1
        elif creatinine < 3.5:
            return 2
        elif creatinine < 5.0:
            return 3
        else:
            return 4


def normalize_new_features(df: pd.DataFrame, features_to_normalize: list) -> pd.DataFrame:
    """
    Apply z-score normalization to new derived features.

    Args:
        df: DataFrame with derived features
        features_to_normalize: List of feature names to normalize

    Returns:
        DataFrame with normalized features
    """
    result = df.copy()

    for feature in features_to_normalize:
        if feature in result.columns:
            mean = result[feature].mean()
            std = result[feature].std()
            if std > 0:
                result[feature] = (result[feature] - mean) / std
            logger.info(f"  Normalized {feature}: mean={mean:.2f}, std={std:.2f}")

    return result


def main():
    """Main function to compute derived features for all data splits."""

    logger.info("=" * 80)
    logger.info("COMPUTING DERIVED FEATURES")
    logger.info("=" * 80)

    # Create logs directory if needed
    Path("logs").mkdir(exist_ok=True)

    # Initialize computer
    computer = DerivedFeatureComputer()

    # Define file paths
    data_dir = Path("data/processed")

    splits = ['train', 'val', 'test']

    # Features to add (for column ordering)
    derived_features = ['PaO2_FiO2', 'Shock_Index', 'SOFA',
                       'sofa_respiration', 'sofa_coagulation', 'sofa_liver',
                       'sofa_cardiovascular', 'sofa_cns', 'sofa_renal']

    # Process each split
    for split in splits:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {split} set")
        logger.info(f"{'='*80}")

        # Load features
        features_path = data_dir / f"{split}_features.csv"
        normalized_path = data_dir / f"{split}_features_normalized.csv"

        if not features_path.exists():
            logger.error(f"File not found: {features_path}")
            continue

        # Load data
        logger.info(f"Loading {features_path}...")
        df = pd.read_csv(features_path)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

        # Compute derived features
        df_with_derived = computer.compute_all_derived_features(df)

        # Save non-normalized version
        output_path = data_dir / f"{split}_features.csv"
        df_with_derived.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

        # Load and process normalized version
        if normalized_path.exists():
            logger.info(f"\nProcessing normalized version...")
            df_norm = pd.read_csv(normalized_path)

            # Compute derived features on normalized data
            # But we need to use original (non-normalized) values for SOFA calculation!
            # So we compute on original and then normalize the derived features

            # Add derived features (computed from non-normalized values)
            for feat in derived_features:
                if feat in df_with_derived.columns:
                    df_norm[feat] = df_with_derived[feat].values

            # Normalize the new derived features
            features_to_normalize = ['PaO2_FiO2', 'Shock_Index', 'SOFA']
            df_norm = normalize_new_features(df_norm, features_to_normalize)

            # Save normalized version
            output_norm_path = data_dir / f"{split}_features_normalized.csv"
            df_norm.to_csv(output_norm_path, index=False)
            logger.info(f"Saved normalized to {output_norm_path}")

        # Print summary
        logger.info(f"\n{split.upper()} SET SUMMARY:")
        logger.info(f"  Total features: {len(df_with_derived.columns)}")
        logger.info(f"  New features added: {len(derived_features)}")
        logger.info(f"  SOFA range: {df_with_derived['SOFA'].min():.0f} - {df_with_derived['SOFA'].max():.0f}")
        logger.info(f"  SOFA mean: {df_with_derived['SOFA'].mean():.2f}")

if __name__ == "__main__":
    main()