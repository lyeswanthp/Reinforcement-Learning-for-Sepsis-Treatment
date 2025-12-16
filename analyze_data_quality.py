#!/usr/bin/env python3
"""
Data Quality Analysis Script

Analyzes the preprocessing results to understand:
1. Feature availability and missingness patterns
2. What features remain after cleaning
3. Recommendations for improving data quality

Usage:
    python analyze_data_quality.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_quality_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def analyze_intermediate_features(intermediate_path: str = "data/intermediate/features_raw.csv"):
    """Analyze the raw features before cleaning to understand missingness."""

    logger.info("="*80)
    logger.info("ANALYZING RAW FEATURES (Before Cleaning)")
    logger.info("="*80)

    if not Path(intermediate_path).exists():
        logger.warning(f"⚠️  Intermediate file not found: {intermediate_path}")
        logger.warning("   Run preprocessing with --save-intermediate flag")
        return None

    # Load raw features
    logger.info(f"Loading {intermediate_path}...")
    df = pd.read_csv(intermediate_path)

    logger.info(f"\n✓ Loaded {len(df):,} observations")
    logger.info(f"✓ Total columns: {len(df.columns)}")

    # Analyze missingness
    logger.info("\n" + "="*80)
    logger.info("FEATURE MISSINGNESS ANALYSIS")
    logger.info("="*80)

    missing_stats = []
    for col in df.columns:
        if col not in ['stay_id', 'time_window']:
            total = len(df)
            missing = df[col].isna().sum()
            missing_pct = (missing / total) * 100
            available = total - missing

            missing_stats.append({
                'feature': col,
                'total_obs': total,
                'missing_count': missing,
                'missing_pct': missing_pct,
                'available_count': available,
                'available_pct': 100 - missing_pct
            })

    missing_df = pd.DataFrame(missing_stats).sort_values('missing_pct', ascending=False)

    # Group by missingness level
    high_missing = missing_df[missing_df['missing_pct'] > 50]
    medium_missing = missing_df[(missing_df['missing_pct'] > 20) & (missing_df['missing_pct'] <= 50)]
    low_missing = missing_df[missing_df['missing_pct'] <= 20]

    logger.info(f"\n📊 Missingness Summary:")
    logger.info(f"   High missing (>50%):  {len(high_missing)} features - WILL BE DROPPED")
    logger.info(f"   Medium missing (20-50%): {len(medium_missing)} features")
    logger.info(f"   Low missing (<20%):   {len(low_missing)} features")

    # Print detailed breakdown
    if len(high_missing) > 0:
        logger.info("\n⚠️  HIGH MISSING FEATURES (>50% - will be dropped):")
        for _, row in high_missing.iterrows():
            logger.info(f"   {row['feature']:25s} {row['missing_pct']:6.2f}% missing ({row['available_count']:,} available)")

    if len(medium_missing) > 0:
        logger.info("\n⚠️  MEDIUM MISSING FEATURES (20-50%):")
        for _, row in medium_missing.iterrows():
            logger.info(f"   {row['feature']:25s} {row['missing_pct']:6.2f}% missing ({row['available_count']:,} available)")

    if len(low_missing) > 0:
        logger.info("\n✓ LOW MISSING FEATURES (<20%):")
        for _, row in low_missing.head(20).iterrows():
            logger.info(f"   {row['feature']:25s} {row['missing_pct']:6.2f}% missing ({row['available_count']:,} available)")

    return missing_df


def analyze_processed_features(processed_path: str = "data/processed/train_features.csv"):
    """Analyze the processed features after cleaning."""

    logger.info("\n" + "="*80)
    logger.info("ANALYZING PROCESSED FEATURES (After Cleaning)")
    logger.info("="*80)

    if not Path(processed_path).exists():
        logger.error(f"❌ Processed file not found: {processed_path}")
        logger.error("   Please run preprocessing first!")
        return None

    # Load processed features
    logger.info(f"Loading {processed_path}...")
    df = pd.read_csv(processed_path)

    logger.info(f"\n✓ Loaded {len(df):,} observations")
    logger.info(f"✓ Total columns: {len(df.columns)}")

    # List all features
    feature_cols = [col for col in df.columns if col not in ['stay_id', 'time_window']]

    logger.info(f"\n📋 REMAINING FEATURES ({len(feature_cols)}):")
    logger.info("   " + "-"*60)

    # Categorize features
    vitals = [f for f in feature_cols if f in ['HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'SpO2', 'FiO2_1', 'GCS']]
    labs = [f for f in feature_cols if f in ['Potassium', 'Sodium', 'Chloride', 'Glucose', 'BUN', 'Creatinine',
                                               'Magnesium', 'Calcium', 'SGOT', 'SGPT', 'Total_bili', 'Hb',
                                               'WBC_count', 'Platelets_count', 'PTT', 'PT', 'INR']]
    blood_gas = [f for f in feature_cols if f in ['Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'HCO3', 'Arterial_lactate']]
    fluid = [f for f in feature_cols if 'input' in f or 'output' in f or 'balance' in f]
    demo = [f for f in feature_cols if f in ['gender', 'age', 'Weight_kg', 're_admission']]
    derived = [f for f in feature_cols if f in ['SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2', 'mechvent', 'max_dose_vaso']]
    other = [f for f in feature_cols if f not in vitals + labs + blood_gas + fluid + demo + derived]

    if vitals:
        logger.info(f"\n   ✓ Vitals ({len(vitals)}):")
        for f in vitals:
            logger.info(f"      - {f}")
    else:
        logger.warning("   ⚠️  NO VITAL SIGNS!")

    if labs:
        logger.info(f"\n   ✓ Lab Values ({len(labs)}):")
        for f in labs:
            logger.info(f"      - {f}")
    else:
        logger.warning("   ⚠️  NO LAB VALUES!")

    if blood_gas:
        logger.info(f"\n   ✓ Blood Gas ({len(blood_gas)}):")
        for f in blood_gas:
            logger.info(f"      - {f}")
    else:
        logger.warning("   ⚠️  NO BLOOD GAS VALUES!")

    if fluid:
        logger.info(f"\n   ✓ Fluid Balance ({len(fluid)}):")
        for f in fluid:
            logger.info(f"      - {f}")

    if demo:
        logger.info(f"\n   ✓ Demographics ({len(demo)}):")
        for f in demo:
            logger.info(f"      - {f}")

    if derived:
        logger.info(f"\n   ✓ Derived Features ({len(derived)}):")
        for f in derived:
            logger.info(f"      - {f}")

    if other:
        logger.info(f"\n   ✓ Other ({len(other)}):")
        for f in other:
            logger.info(f"      - {f}")

    # Check for missing values
    missing_count = df[feature_cols].isna().sum().sum()
    if missing_count > 0:
        logger.warning(f"\n⚠️  WARNING: {missing_count:,} missing values found in processed data!")
    else:
        logger.info(f"\n✓ No missing values in processed data")

    # Basic statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    logger.info(f"Total observations:      {len(df):,}")
    logger.info(f"Unique ICU stays:        {df['stay_id'].nunique():,}")
    logger.info(f"Features extracted:      {len(feature_cols)}")
    logger.info(f"Expected features:       48+")
    logger.info(f"Feature coverage:        {len(feature_cols)/48*100:.1f}%")

    return df


def recommendations():
    """Provide recommendations based on analysis."""

    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS")
    logger.info("="*80)

    logger.info("""
📋 IMMEDIATE ACTIONS:

1. **Adjust Missing Data Threshold:**
   - Current: 50% (too strict for clinical data)
   - Recommended: 70-80% for initial exploration
   - Edit configs/config.yaml:
     preprocessing:
       missing_data:
         max_missing_ratio: 0.70  # Change from 0.5 to 0.7

2. **Imputation Strategy:**
   - Consider using more sophisticated imputation (KNN, MICE)
   - Forward-fill within ICU stays (current approach is good)
   - Use clinical domain knowledge for defaults

3. **Feature Engineering:**
   - Some features may need to be computed (SOFA, SIRS, GCS)
   - Check if vital signs exist in chartevents with different item IDs
   - Verify blood pressure extraction from chartevents

4. **Re-run Preprocessing:**
   After adjusting config:
   ```bash
   python run_preprocessing.py --data-path <your-path>
   ```

5. **Proceed with Current Data (Alternative):**
   If you want to proceed with the 13 features:
   - You can still build a simplified RL model
   - Focus on fluid management decisions
   - Use available vitals (HR, RR, SpO2) + fluid balance

📊 EXPECTED FEATURE GROUPS:

For a complete RL model, you should have:
✓ Vitals: HR, BP (sys/mean/dia), RR, Temp, SpO2, FiO2 (8-10 features)
✓ Labs: Basic chemistry, CBC, coagulation (15-20 features)
✓ Blood gas: pH, lactate, pO2, pCO2 (4-6 features)
✓ Fluid: Input/output/balance (4-5 features)
✓ Demographics: age, gender, weight (3-4 features)
✓ Derived: SOFA, SIRS, shock index (3-4 features)

Total: ~40-50 features

🎯 NEXT STEPS:

Once you have adequate features:
1. Extract actions (IV fluids + vasopressor bins)
2. Compute rewards (SOFA-based)
3. Build MDP trajectories
4. Train Q-learning model
5. Evaluate with WDR-OPE
""")


def main():
    """Main analysis function."""

    logger.info("="*80)
    logger.info("MIMIC-IV DATA QUALITY ANALYSIS")
    logger.info("="*80)

    # Create logs directory if needed
    Path("logs").mkdir(exist_ok=True)

    # Analyze intermediate features (if available)
    missing_df = analyze_intermediate_features()

    # Analyze processed features
    processed_df = analyze_processed_features()

    # Provide recommendations
    recommendations()

    # Save detailed report
    if missing_df is not None:
        report_path = "logs/feature_missingness_report.csv"
        missing_df.to_csv(report_path, index=False)
        logger.info(f"\n✓ Detailed missingness report saved to: {report_path}")

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()