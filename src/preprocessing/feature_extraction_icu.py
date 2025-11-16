"""
Feature Extraction from ICU Data
Extracts vital signs, interventions, fluids, and other ICU-level features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class ICUFeatureExtractor:
    """
    Extract features from ICU data files:
    - Vital signs from chartevents (HR, BP, RR, Temp, SpO2, GCS)
    - Mechanical ventilation status
    - Fluid balance from inputevents and outputevents
    - Vasopressor doses
    """

    # Chart item IDs for MIMIC-IV vitals (need to be mapped from d_items)
    VITAL_SIGN_ITEMS = {
        'HR': ['Heart Rate', 'HR'],
        'SysBP': ['Systolic Blood Pressure', 'NBP [Systolic]', 'Arterial Blood Pressure systolic'],
        'MeanBP': ['Mean Blood Pressure', 'Arterial Blood Pressure mean'],
        'DiaBP': ['Diastolic Blood Pressure', 'NBP [Diastolic]', 'Arterial Blood Pressure diastolic'],
        'RR': ['Respiratory Rate', 'RR'],
        'Temp_C': ['Temperature Celsius', 'Temperature C'],
        'SpO2': ['SpO2', 'O2 saturation pulseoxymetry'],
        'FiO2_1': ['FiO2', 'Inspired O2 Fraction'],
        'GCS': ['GCS Total', 'Glasgow Coma Scale'],
    }

    # Mechanical ventilation item keywords
    MECHVENT_ITEMS = ['Mechanical Ventilation', 'Ventilator', 'Vent Mode']

    # Vasopressor medications
    VASOPRESSOR_ITEMS = {
        'norepinephrine': ['Norepinephrine', 'Levophed'],
        'epinephrine': ['Epinephrine', 'Adrenaline'],
        'vasopressin': ['Vasopressin'],
        'dopamine': ['Dopamine'],
        'phenylephrine': ['Phenylephrine', 'Neo-Synephrine'],
        'dobutamine': ['Dobutamine']
    }

    # IV fluid keywords
    IV_FLUID_KEYWORDS = [
        'Sodium Chloride', 'NaCl', 'Normal Saline', 'NS',
        'Lactated Ringers', 'LR',
        'Albumin',
        'Dextrose',
        'Crystalloid', 'Colloid'
    ]

    def __init__(self, config: Dict):
        """
        Initialize ICU feature extractor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.time_window_hours = config.get('mdp', {}).get('time_window_hours', 4)

        logger.info(f"ICUFeatureExtractor initialized (time window: {self.time_window_hours}h)")

    def extract_vital_signs(
        self,
        cohort: pd.DataFrame,
        chartevents: pd.DataFrame,
        d_items: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract vital signs for each ICU stay and time window

        Args:
            cohort: Cohort ICU stays
            chartevents: Chart events
            d_items: Chart items dictionary

        Returns:
            DataFrame with vital sign features per stay and time window
        """
        logger.info("Extracting vital signs from chartevents...")

        # Map vital sign names to itemids
        vital_itemid_map = self._create_vital_itemid_map(d_items)

        # Get relevant itemids
        all_itemids = []
        for itemids in vital_itemid_map.values():
            all_itemids.extend(itemids)

        logger.info(f"  Filtering chartevents for {len(all_itemids)} vital sign items...")

        # Filter chartevents for cohort and relevant items
        cohort_stays = cohort['stay_id'].unique()
        chartevents_filtered = chartevents[
            (chartevents['stay_id'].isin(cohort_stays)) &
            (chartevents['itemid'].isin(all_itemids))
        ].copy()

        logger.info(f"  Filtered to {len(chartevents_filtered):,} chart events")

        # Merge with cohort to get ICU stay times
        chartevents_filtered = chartevents_filtered.merge(
            cohort[['stay_id', 'intime', 'outtime']],
            on='stay_id',
            how='left'
        )

        # Create time windows
        chartevents_filtered['hours_since_admission'] = (
            (chartevents_filtered['charttime'] - chartevents_filtered['intime']).dt.total_seconds() / 3600
        )
        chartevents_filtered['time_window'] = (
            chartevents_filtered['hours_since_admission'] // self.time_window_hours
        ).astype(int)

        # Map itemids to feature names
        itemid_to_feature = {}
        for feature, itemids in vital_itemid_map.items():
            for itemid in itemids:
                itemid_to_feature[itemid] = feature

        chartevents_filtered['feature_name'] = chartevents_filtered['itemid'].map(itemid_to_feature)

        # Use valuenum for numeric vitals
        chartevents_filtered['value'] = pd.to_numeric(chartevents_filtered['valuenum'], errors='coerce')

        # Aggregate: mean value per stay, time window, and feature
        vital_features = chartevents_filtered.groupby(
            ['stay_id', 'time_window', 'feature_name']
        )['value'].agg(['mean', 'min', 'max', 'count']).reset_index()

        # Use mean as primary value
        vital_features = vital_features.rename(columns={'mean': 'value'})

        # Pivot to wide format
        vital_features_wide = vital_features.pivot_table(
            index=['stay_id', 'time_window'],
            columns='feature_name',
            values='value'
        ).reset_index()

        logger.info(f"✓ Extracted vital signs: {len(vital_features_wide):,} stay-time windows")
        return vital_features_wide

    def _create_vital_itemid_map(self, d_items: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create mapping from vital sign names to itemids

        Args:
            d_items: Chart items dictionary

        Returns:
            Dictionary mapping feature names to lists of itemids
        """
        vital_map = {}

        for feature, keywords in self.VITAL_SIGN_ITEMS.items():
            itemids = []
            for keyword in keywords:
                matches = d_items[
                    d_items['label'].str.contains(keyword, case=False, na=False)
                ]['itemid'].tolist()
                itemids.extend(matches)

            # Remove duplicates
            itemids = list(set(itemids))
            vital_map[feature] = itemids

            logger.debug(f"  {feature}: {len(itemids)} itemids")

        return vital_map

    def extract_mechanical_ventilation(
        self,
        cohort: pd.DataFrame,
        chartevents: pd.DataFrame,
        d_items: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract mechanical ventilation status

        Args:
            cohort: Cohort ICU stays
            chartevents: Chart events
            d_items: Chart items dictionary

        Returns:
            DataFrame with mechanical ventilation binary feature
        """
        logger.info("Extracting mechanical ventilation status...")

        # Find mechvent itemids
        mechvent_itemids = []
        for keyword in self.MECHVENT_ITEMS:
            matches = d_items[
                d_items['label'].str.contains(keyword, case=False, na=False)
            ]['itemid'].tolist()
            mechvent_itemids.extend(matches)

        mechvent_itemids = list(set(mechvent_itemids))

        if len(mechvent_itemids) == 0:
            logger.warning("  No mechanical ventilation items found")
            return pd.DataFrame()

        # Filter chartevents
        cohort_stays = cohort['stay_id'].unique()
        mechvent_events = chartevents[
            (chartevents['stay_id'].isin(cohort_stays)) &
            (chartevents['itemid'].isin(mechvent_itemids))
        ].copy()

        # Merge with cohort
        mechvent_events = mechvent_events.merge(
            cohort[['stay_id', 'intime', 'outtime']],
            on='stay_id',
            how='left'
        )

        # Create time windows
        mechvent_events['hours_since_admission'] = (
            (mechvent_events['charttime'] - mechvent_events['intime']).dt.total_seconds() / 3600
        )
        mechvent_events['time_window'] = (
            mechvent_events['hours_since_admission'] // self.time_window_hours
        ).astype(int)

        # Mechanical ventilation status (binary)
        mechvent_events['mechvent'] = 1

        # Aggregate: any mechvent event in time window
        mechvent_features = mechvent_events.groupby(
            ['stay_id', 'time_window']
        )['mechvent'].max().reset_index()

        logger.info(f"✓ Extracted mechvent status: {len(mechvent_features):,} observations")
        return mechvent_features

    def extract_fluid_balance(
        self,
        cohort: pd.DataFrame,
        inputevents: pd.DataFrame,
        outputevents: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract fluid balance features (input and output)

        Features:
        - input_total: Cumulative input volume
        - input_4hourly: Input in current time window
        - output_total: Cumulative output volume
        - output_4hourly: Output in current time window

        Args:
            cohort: Cohort ICU stays
            inputevents: Input events
            outputevents: Output events

        Returns:
            DataFrame with fluid balance features
        """
        logger.info("Extracting fluid balance features...")

        # Process input events
        logger.info("  Processing input events...")
        cohort_stays = cohort['stay_id'].unique()
        inputs = inputevents[
            inputevents['stay_id'].isin(cohort_stays)
        ].copy()

        # Merge with cohort
        inputs = inputs.merge(
            cohort[['stay_id', 'intime', 'outtime']],
            on='stay_id',
            how='left'
        )

        # Create time windows
        inputs['hours_since_admission'] = (
            (inputs['starttime'] - inputs['intime']).dt.total_seconds() / 3600
        )
        inputs['time_window'] = (
            inputs['hours_since_admission'] // self.time_window_hours
        ).astype(int)

        # Sum input volume per time window
        input_features = inputs.groupby(
            ['stay_id', 'time_window']
        )['amount'].sum().reset_index()
        input_features.columns = ['stay_id', 'time_window', 'input_4hourly']

        # Calculate cumulative input
        input_features = input_features.sort_values(['stay_id', 'time_window'])
        input_features['input_total'] = input_features.groupby('stay_id')['input_4hourly'].cumsum()

        # Process output events
        logger.info("  Processing output events...")
        outputs = outputevents[
            outputevents['stay_id'].isin(cohort_stays)
        ].copy()

        # Merge with cohort
        outputs = outputs.merge(
            cohort[['stay_id', 'intime', 'outtime']],
            on='stay_id',
            how='left'
        )

        # Create time windows
        outputs['hours_since_admission'] = (
            (outputs['charttime'] - outputs['intime']).dt.total_seconds() / 3600
        )
        outputs['time_window'] = (
            outputs['hours_since_admission'] // self.time_window_hours
        ).astype(int)

        # Sum output volume per time window
        output_features = outputs.groupby(
            ['stay_id', 'time_window']
        )['value'].sum().reset_index()
        output_features.columns = ['stay_id', 'time_window', 'output_4hourly']

        # Calculate cumulative output
        output_features = output_features.sort_values(['stay_id', 'time_window'])
        output_features['output_total'] = output_features.groupby('stay_id')['output_4hourly'].cumsum()

        # Merge input and output
        fluid_features = input_features.merge(
            output_features,
            on=['stay_id', 'time_window'],
            how='outer'
        )

        # Fill NaN with 0 for missing time windows
        fluid_columns = ['input_4hourly', 'input_total', 'output_4hourly', 'output_total']
        fluid_features[fluid_columns] = fluid_features[fluid_columns].fillna(0)

        # Calculate cumulative balance
        fluid_features['cumulated_balance'] = fluid_features['input_total'] - fluid_features['output_total']

        logger.info(f"✓ Extracted fluid balance: {len(fluid_features):,} observations")
        return fluid_features

    def extract_vasopressor_dose(
        self,
        cohort: pd.DataFrame,
        inputevents: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract vasopressor doses

        Args:
            cohort: Cohort ICU stays
            inputevents: Input events

        Returns:
            DataFrame with vasopressor dose features
        """
        logger.info("Extracting vasopressor doses...")

        cohort_stays = cohort['stay_id'].unique()

        # Filter for vasopressors
        vasopressor_events = []

        for vaso_name, keywords in self.VASOPRESSOR_ITEMS.items():
            for keyword in keywords:
                vaso_data = inputevents[
                    (inputevents['stay_id'].isin(cohort_stays)) &
                    (inputevents['ordercategoryname'].str.contains('Vasoactive', case=False, na=False) |
                     inputevents['ordercategorydescription'].str.contains(keyword, case=False, na=False))
                ].copy()

                if len(vaso_data) > 0:
                    vaso_data['vasopressor_type'] = vaso_name
                    vasopressor_events.append(vaso_data)

        if len(vasopressor_events) == 0:
            logger.warning("  No vasopressor data found")
            return pd.DataFrame()

        vasopressors = pd.concat(vasopressor_events, ignore_index=True)

        # Merge with cohort
        vasopressors = vasopressors.merge(
            cohort[['stay_id', 'intime', 'outtime']],
            on='stay_id',
            how='left'
        )

        # Create time windows
        vasopressors['hours_since_admission'] = (
            (vasopressors['starttime'] - vasopressors['intime']).dt.total_seconds() / 3600
        )
        vasopressors['time_window'] = (
            vasopressors['hours_since_admission'] // self.time_window_hours
        ).astype(int)

        # Calculate total vasopressor dose (standardized units)
        vasopressors['dose'] = pd.to_numeric(vasopressors['rate'], errors='coerce').fillna(0)

        # Aggregate: max dose per time window
        vaso_features = vasopressors.groupby(
            ['stay_id', 'time_window']
        )['dose'].max().reset_index()
        vaso_features.columns = ['stay_id', 'time_window', 'max_dose_vaso']

        logger.info(f"✓ Extracted vasopressor doses: {len(vaso_features):,} observations")
        return vaso_features

    def calculate_derived_scores(
        self,
        vital_signs: pd.DataFrame,
        lab_values: pd.DataFrame,
        fluid_balance: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate derived clinical scores

        Scores:
        - SIRS (Systemic Inflammatory Response Syndrome)
        - Shock Index (HR / SysBP)
        - PaO2/FiO2 ratio

        Args:
            vital_signs: Vital sign features
            lab_values: Lab value features
            fluid_balance: Fluid balance features

        Returns:
            DataFrame with derived scores
        """
        logger.info("Calculating derived clinical scores...")

        # Merge all features
        features = vital_signs.copy()
        if not lab_values.empty:
            features = features.merge(lab_values, on=['stay_id', 'time_window'], how='outer')
        if not fluid_balance.empty:
            features = features.merge(fluid_balance, on=['stay_id', 'time_window'], how='outer')

        # Shock Index = HR / SysBP
        if 'HR' in features.columns and 'SysBP' in features.columns:
            features['Shock_Index'] = features['HR'] / features['SysBP']
        else:
            features['Shock_Index'] = np.nan

        # PaO2/FiO2 ratio
        if 'paO2' in features.columns and 'FiO2_1' in features.columns:
            features['PaO2_FiO2'] = features['paO2'] / (features['FiO2_1'] / 100)
        else:
            features['PaO2_FiO2'] = np.nan

        # SIRS criteria (need: Temp, HR, RR, WBC)
        sirs_score = 0
        if 'Temp_C' in features.columns:
            sirs_score += ((features['Temp_C'] > 38) | (features['Temp_C'] < 36)).astype(int)
        if 'HR' in features.columns:
            sirs_score += (features['HR'] > 90).astype(int)
        if 'RR' in features.columns:
            sirs_score += (features['RR'] > 20).astype(int)
        if 'WBC_count' in features.columns:
            sirs_score += ((features['WBC_count'] > 12) | (features['WBC_count'] < 4)).astype(int)

        features['SIRS'] = sirs_score

        logger.info(f"✓ Calculated derived scores")
        return features


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Example usage
    from data_loader import MIMICDataLoader
    from cohort_selection import SepsisCohortSelector
    from utils.config_loader import ConfigLoader

    config = ConfigLoader().config
    loader = MIMICDataLoader("data/raw/mimic-iv-3.1", config)

    try:
        # Load data
        patients = loader.load_patients()
        admissions = loader.load_admissions()
        icustays = loader.load_icustays()
        diagnoses_icd = loader.load_table('diagnoses_icd')
        d_items = loader.load_d_items()

        # Select cohort
        selector = SepsisCohortSelector(config)
        cohort, _ = selector.select_cohort(patients, admissions, icustays, diagnoses_icd)

        print(f"\nCohort size: {len(cohort):,} ICU stays")
        print("\nReady to extract ICU features")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
