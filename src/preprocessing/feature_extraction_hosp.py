"""
Feature Extraction from Hospital (hosp) Data
Extracts demographics, lab values, and other hospital-level features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class HospitalFeatureExtractor:
    """
    Extract features from hospital data files:
    - Demographics from patients
    - Lab values from labevents
    - Diagnoses from diagnoses_icd
    - Procedures from procedures_icd
    - Medications from prescriptions
    """

    # Lab item IDs for MIMIC-IV (these need to be mapped from d_labitems)
    # Key lab values we need for the 48 features
    LAB_FEATURES = {
        'Potassium': ['Potassium'],
        'Sodium': ['Sodium'],
        'Chloride': ['Chloride'],
        'Glucose': ['Glucose'],
        'BUN': ['Blood Urea Nitrogen', 'BUN'],
        'Creatinine': ['Creatinine'],
        'Magnesium': ['Magnesium'],
        'Calcium': ['Calcium', 'Calcium, Total'],
        'SGOT': ['Aspartate Aminotransferase (AST)', 'AST'],
        'SGPT': ['Alanine Aminotransferase (ALT)', 'ALT'],
        'Total_bili': ['Bilirubin, Total'],
        'Hb': ['Hemoglobin', 'Hgb'],
        'WBC_count': ['White Blood Cells', 'WBC'],
        'Platelets_count': ['Platelet Count', 'Platelets'],
        'PTT': ['PTT'],
        'PT': ['PT'],
        'INR': ['INR'],
        'Arterial_pH': ['pH', 'Arterial pH'],
        'paO2': ['pO2', 'Arterial O2 Pressure'],
        'paCO2': ['pCO2', 'Arterial CO2 Pressure'],
        'Arterial_BE': ['Base Excess'],
        'HCO3': ['Bicarbonate', 'HCO3'],
        'Arterial_lactate': ['Lactate', 'Lactic Acid'],
    }

    def __init__(self, config: Dict):
        """
        Initialize hospital feature extractor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.time_window_hours = config.get('mdp', {}).get('time_window_hours', 4)

        logger.info(f"HospitalFeatureExtractor initialized (time window: {self.time_window_hours}h)")

    def extract_demographics(
        self,
        cohort: pd.DataFrame,
        patients: pd.DataFrame,
        admissions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract demographic features for each ICU stay

        Features: gender, age, re_admission

        Args:
            cohort: Cohort ICU stays
            patients: Patient demographics
            admissions: Admission records

        Returns:
            DataFrame with demographic features
        """
        logger.info("Extracting demographic features...")

        demographics = cohort[['stay_id', 'subject_id', 'hadm_id']].copy()

        # Merge with patients
        demographics = demographics.merge(
            patients[['subject_id', 'gender', 'anchor_age']],
            on='subject_id',
            how='left'
        )

        # Gender (binary: M=1, F=0)
        demographics['gender'] = (demographics['gender'] == 'M').astype(int)

        # Age
        demographics['age'] = demographics['anchor_age']

        # Re-admission: check if patient has prior admissions
        admission_counts = admissions.groupby('subject_id').size().reset_index(name='admission_count')
        demographics = demographics.merge(admission_counts, on='subject_id', how='left')
        demographics['re_admission'] = (demographics['admission_count'] > 1).astype(int)

        # Select final columns
        demographics = demographics[['stay_id', 'gender', 'age', 're_admission']]

        logger.info(f"✓ Extracted demographics for {len(demographics):,} stays")
        return demographics

    def extract_lab_features(
        self,
        cohort: pd.DataFrame,
        labevents: pd.DataFrame,
        d_labitems: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract lab values for each ICU stay and time window

        Args:
            cohort: Cohort ICU stays
            labevents: Lab events
            d_labitems: Lab item dictionary

        Returns:
            DataFrame with lab features per stay and time window
        """
        logger.info("Extracting lab features...")

        # Map lab item names to itemids
        lab_itemid_map = self._create_lab_itemid_map(d_labitems)

        # Get relevant itemids
        all_itemids = []
        for itemids in lab_itemid_map.values():
            all_itemids.extend(itemids)

        # Filter labevents for cohort and relevant items
        logger.info(f"  Filtering {len(labevents):,} lab events...")
        cohort_subjects = cohort['subject_id'].unique()
        labevents_filtered = labevents[
            (labevents['subject_id'].isin(cohort_subjects)) &
            (labevents['itemid'].isin(all_itemids))
        ].copy()

        logger.info(f"  Filtered to {len(labevents_filtered):,} relevant lab events")

        # Merge with cohort to get ICU stay times
        labevents_filtered = labevents_filtered.merge(
            cohort[['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime']],
            on=['subject_id', 'hadm_id'],
            how='inner'
        )

        # Filter for events during ICU stay
        labevents_filtered = labevents_filtered[
            (labevents_filtered['charttime'] >= labevents_filtered['intime']) &
            (labevents_filtered['charttime'] <= labevents_filtered['outtime'])
        ]

        logger.info(f"  {len(labevents_filtered):,} lab events during ICU stays")

        # Create time windows
        labevents_filtered['hours_since_admission'] = (
            (labevents_filtered['charttime'] - labevents_filtered['intime']).dt.total_seconds() / 3600
        )
        labevents_filtered['time_window'] = (
            labevents_filtered['hours_since_admission'] // self.time_window_hours
        ).astype(int)

        # Map itemids to feature names
        itemid_to_feature = {}
        for feature, itemids in lab_itemid_map.items():
            for itemid in itemids:
                itemid_to_feature[itemid] = feature

        labevents_filtered['feature_name'] = labevents_filtered['itemid'].map(itemid_to_feature)

        # Aggregate: mean value per stay, time window, and feature
        lab_features = labevents_filtered.groupby(
            ['stay_id', 'time_window', 'feature_name']
        )['valuenum'].mean().reset_index()

        # Pivot to wide format
        lab_features_wide = lab_features.pivot_table(
            index=['stay_id', 'time_window'],
            columns='feature_name',
            values='valuenum'
        ).reset_index()

        logger.info(f"✓ Extracted lab features: {len(lab_features_wide):,} stay-time windows")
        return lab_features_wide

    def _create_lab_itemid_map(self, d_labitems: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create mapping from feature names to lab itemids

        Args:
            d_labitems: Lab items dictionary

        Returns:
            Dictionary mapping feature names to lists of itemids
        """
        lab_map = {}

        for feature, keywords in self.LAB_FEATURES.items():
            itemids = []
            for keyword in keywords:
                matches = d_labitems[
                    d_labitems['label'].str.contains(keyword, case=False, na=False)
                ]['itemid'].tolist()
                itemids.extend(matches)

            # Remove duplicates
            itemids = list(set(itemids))
            lab_map[feature] = itemids

            logger.debug(f"  {feature}: {len(itemids)} itemids")

        return lab_map

    def extract_weight(
        self,
        cohort: pd.DataFrame,
        omr: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract patient weight from OMR (Observation Medical Record)

        Args:
            cohort: Cohort ICU stays
            omr: OMR data

        Returns:
            DataFrame with weight feature
        """
        logger.info("Extracting weight from OMR...")

        # Filter for weight measurements
        weight_data = omr[
            (omr['subject_id'].isin(cohort['subject_id'].unique())) &
            (omr['result_name'].str.contains('Weight', case=False, na=False))
        ].copy()

        if len(weight_data) == 0:
            logger.warning("  No weight data found in OMR")
            return pd.DataFrame({'stay_id': cohort['stay_id'], 'Weight_kg': np.nan})

        # Convert to numeric (handle units)
        weight_data['weight_value'] = pd.to_numeric(weight_data['result_value'], errors='coerce')

        # Merge with cohort
        weight_data = weight_data.merge(
            cohort[['subject_id', 'stay_id', 'intime']],
            on='subject_id',
            how='inner'
        )

        # Get weight closest to ICU admission
        weight_data['time_diff'] = abs(
            (weight_data['chartdate'] - weight_data['intime'].dt.date).dt.days
        )

        # Get weight with minimum time difference per stay
        weight_features = weight_data.loc[
            weight_data.groupby('stay_id')['time_diff'].idxmin()
        ][['stay_id', 'weight_value']]

        weight_features.columns = ['stay_id', 'Weight_kg']

        logger.info(f"✓ Extracted weight for {len(weight_features):,} stays")
        return weight_features

    def calculate_sofa_components(
        self,
        cohort: pd.DataFrame,
        labevents: pd.DataFrame,
        d_labitems: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate SOFA score components from lab values

        SOFA components:
        - Respiratory (PaO2/FiO2)
        - Coagulation (Platelets)
        - Liver (Bilirubin)
        - Cardiovascular (MAP or vasopressor use)
        - CNS (GCS - from ICU data)
        - Renal (Creatinine or urine output)

        Args:
            cohort: Cohort ICU stays
            labevents: Lab events
            d_labitems: Lab items dictionary

        Returns:
            DataFrame with SOFA component scores
        """
        logger.info("Calculating SOFA score components from labs...")

        # Extract relevant lab values
        lab_features = self.extract_lab_features(cohort, labevents, d_labitems)

        sofa_scores = lab_features[['stay_id', 'time_window']].copy()

        # Coagulation: Platelets
        if 'Platelets_count' in lab_features.columns:
            sofa_scores['sofa_coag'] = pd.cut(
                lab_features['Platelets_count'],
                bins=[0, 20, 50, 100, 150, np.inf],
                labels=[4, 3, 2, 1, 0],
                right=False
            ).astype(float)
        else:
            sofa_scores['sofa_coag'] = 0

        # Liver: Total Bilirubin
        if 'Total_bili' in lab_features.columns:
            sofa_scores['sofa_liver'] = pd.cut(
                lab_features['Total_bili'],
                bins=[0, 1.2, 2.0, 6.0, 12.0, np.inf],
                labels=[0, 1, 2, 3, 4],
                right=False
            ).astype(float)
        else:
            sofa_scores['sofa_liver'] = 0

        # Renal: Creatinine
        if 'Creatinine' in lab_features.columns:
            sofa_scores['sofa_renal'] = pd.cut(
                lab_features['Creatinine'],
                bins=[0, 1.2, 2.0, 3.5, 5.0, np.inf],
                labels=[0, 1, 2, 3, 4],
                right=False
            ).astype(float)
        else:
            sofa_scores['sofa_renal'] = 0

        logger.info(f"✓ Calculated SOFA components for {len(sofa_scores):,} observations")
        return sofa_scores

    def extract_diagnosis_features(
        self,
        cohort: pd.DataFrame,
        diagnoses_icd: pd.DataFrame,
        d_icd_diagnoses: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract diagnosis-based features

        Args:
            cohort: Cohort ICU stays
            diagnoses_icd: Diagnosis codes
            d_icd_diagnoses: ICD diagnosis dictionary

        Returns:
            DataFrame with diagnosis features
        """
        logger.info("Extracting diagnosis features...")

        # Get diagnoses for cohort admissions
        cohort_diagnoses = diagnoses_icd[
            diagnoses_icd['hadm_id'].isin(cohort['hadm_id'].unique())
        ].copy()

        # Count diagnoses per admission
        diagnosis_counts = cohort_diagnoses.groupby('hadm_id').size().reset_index(name='num_diagnoses')

        # Merge with cohort
        diagnosis_features = cohort[['stay_id', 'hadm_id']].merge(
            diagnosis_counts,
            on='hadm_id',
            how='left'
        )

        diagnosis_features['num_diagnoses'] = diagnosis_features['num_diagnoses'].fillna(0)

        logger.info(f"✓ Extracted diagnosis features for {len(diagnosis_features):,} stays")
        return diagnosis_features[['stay_id', 'num_diagnoses']]


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
        d_labitems = loader.load_d_labitems()

        # Select cohort
        selector = SepsisCohortSelector(config)
        cohort, _ = selector.select_cohort(patients, admissions, icustays, diagnoses_icd)

        # Extract features
        extractor = HospitalFeatureExtractor(config)

        demographics = extractor.extract_demographics(cohort, patients, admissions)
        print("\nDemographics:")
        print(demographics.head())

        print("\nDone!")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
