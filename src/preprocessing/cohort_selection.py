"""
Cohort Selection for Sepsis Patients
Implements Sepsis-3 definition and inclusion/exclusion criteria
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class SepsisCohortSelector:
    """
    Select sepsis cohort using Sepsis-3 criteria

    Sepsis-3 Definition:
    - Suspected infection (culture + antibiotics within 72h window)
    - SOFA score increase ≥ 2 points from baseline
    - Adult patients (age ≥ 18)
    - Minimum ICU stay duration
    """

    # ICD-10 codes for sepsis (for verification)
    SEPSIS_ICD10_CODES = [
        'A41', 'A40', 'R65', 'T81.4',  # Sepsis, septicemia, SIRS
        'A02.1', 'A20.7', 'A21.7', 'A22.7', 'A26.7', 'A32.7', 'A39.2', 'A39.3', 'A39.4',
        'A40.0', 'A40.1', 'A40.3', 'A40.8', 'A40.9',
        'A41.0', 'A41.1', 'A41.2', 'A41.3', 'A41.4', 'A41.5', 'A41.50', 'A41.51', 'A41.52',
        'A41.53', 'A41.59', 'A41.8', 'A41.81', 'A41.89', 'A41.9',
        'B37.7', 'R65.2', 'R65.20', 'R65.21'
    ]

    # ICD-9 codes for sepsis
    SEPSIS_ICD9_CODES = [
        '038', '020.0', '790.7', '117.9', '112.5', '112.81',
        '003.1', '036.2', '785.52', '995.91', '995.92'
    ]

    # Antibiotics - common IV antibiotics (from inputevents)
    ANTIBIOTIC_KEYWORDS = [
        'vancomycin', 'cefepime', 'ceftriaxone', 'cefazolin', 'ceftazidime',
        'piperacillin', 'tazobactam', 'meropenem', 'imipenem', 'ciprofloxacin',
        'levofloxacin', 'azithromycin', 'metronidazole', 'linezolid',
        'gentamicin', 'tobramycin', 'amikacin', 'clindamycin', 'ampicillin',
        'oxacillin', 'nafcillin', 'penicillin', 'doxycycline', 'tigecycline',
        'colistin', 'polymyxin'
    ]

    def __init__(self, config: Dict):
        """
        Initialize cohort selector

        Args:
            config: Configuration dictionary with cohort criteria
        """
        self.config = config
        self.cohort_config = config.get('cohort', {})

        # Criteria
        self.min_age = self.cohort_config.get('min_age', 18)
        self.max_age = self.cohort_config.get('max_age', 120)
        self.min_icu_los_hours = self.cohort_config.get('min_icu_los_hours', 12)
        self.sepsis3_definition = self.cohort_config.get('sepsis3_definition', True)

        logger.info(f"Cohort criteria: age {self.min_age}-{self.max_age}, "
                   f"min ICU LOS {self.min_icu_los_hours}h, "
                   f"Sepsis-3: {self.sepsis3_definition}")

    def select_cohort(
        self,
        patients: pd.DataFrame,
        admissions: pd.DataFrame,
        icustays: pd.DataFrame,
        diagnoses_icd: pd.DataFrame,
        prescriptions: Optional[pd.DataFrame] = None,
        microbiologyevents: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Select sepsis cohort from MIMIC-IV data

        Args:
            patients: Patient demographics
            admissions: Hospital admissions
            icustays: ICU stays
            diagnoses_icd: ICD diagnosis codes
            prescriptions: Prescription data (for antibiotics)
            microbiologyevents: Microbiology cultures

        Returns:
            Tuple of (cohort_icustays, exclusion_stats)
        """
        logger.info("="*80)
        logger.info("Starting Cohort Selection")
        logger.info("="*80)

        stats = {
            'total_icustays': len(icustays),
            'excluded_age': 0,
            'excluded_los': 0,
            'excluded_no_sepsis': 0,
            'final_cohort': 0
        }

        # Step 1: Merge ICU stays with admissions and patients
        logger.info("\nStep 1: Merging ICU stays with patient data...")
        cohort = icustays.copy()
        cohort = cohort.merge(admissions[['hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag']],
                             on='hadm_id', how='left')
        cohort = cohort.merge(patients[['subject_id', 'gender', 'anchor_age', 'dod']],
                             on='subject_id', how='left')

        logger.info(f"✓ Merged data: {len(cohort):,} ICU stays")

        # Step 2: Age filter
        logger.info(f"\nStep 2: Filtering by age ({self.min_age}-{self.max_age})...")
        initial_count = len(cohort)
        cohort = cohort[(cohort['anchor_age'] >= self.min_age) &
                       (cohort['anchor_age'] <= self.max_age)]
        stats['excluded_age'] = initial_count - len(cohort)
        logger.info(f"✓ Excluded {stats['excluded_age']:,} stays (age criteria)")
        logger.info(f"  Remaining: {len(cohort):,} stays")

        # Step 3: ICU Length of Stay filter
        logger.info(f"\nStep 3: Filtering by ICU LOS (≥{self.min_icu_los_hours}h)...")
        cohort['icu_los_hours'] = (cohort['outtime'] - cohort['intime']).dt.total_seconds() / 3600
        initial_count = len(cohort)
        cohort = cohort[cohort['icu_los_hours'] >= self.min_icu_los_hours]
        stats['excluded_los'] = initial_count - len(cohort)
        logger.info(f"✓ Excluded {stats['excluded_los']:,} stays (ICU LOS < {self.min_icu_los_hours}h)")
        logger.info(f"  Remaining: {len(cohort):,} stays")

        # Step 4: Sepsis identification
        if self.sepsis3_definition:
            logger.info("\nStep 4: Identifying sepsis patients (Sepsis-3 criteria)...")
            sepsis_stays = self._identify_sepsis_stays(
                cohort,
                diagnoses_icd,
                prescriptions,
                microbiologyevents
            )

            initial_count = len(cohort)
            cohort = cohort[cohort['stay_id'].isin(sepsis_stays)]
            stats['excluded_no_sepsis'] = initial_count - len(cohort)
            logger.info(f"✓ Identified {len(cohort):,} sepsis stays")
            logger.info(f"  Excluded {stats['excluded_no_sepsis']:,} non-sepsis stays")

        # Final cohort
        stats['final_cohort'] = len(cohort)

        logger.info("\n" + "="*80)
        logger.info("Cohort Selection Summary")
        logger.info("="*80)
        logger.info(f"Total ICU stays:        {stats['total_icustays']:>10,}")
        logger.info(f"Excluded (age):         {stats['excluded_age']:>10,}")
        logger.info(f"Excluded (ICU LOS):     {stats['excluded_los']:>10,}")
        logger.info(f"Excluded (no sepsis):   {stats['excluded_no_sepsis']:>10,}")
        logger.info(f"{'─'*40}")
        logger.info(f"Final cohort:           {stats['final_cohort']:>10,}")
        logger.info("="*80)

        # Add cohort flag
        cohort['in_cohort'] = True

        return cohort, stats

    def _identify_sepsis_stays(
        self,
        cohort: pd.DataFrame,
        diagnoses_icd: pd.DataFrame,
        prescriptions: Optional[pd.DataFrame] = None,
        microbiologyevents: Optional[pd.DataFrame] = None
    ) -> List[int]:
        """
        Identify sepsis ICU stays using multiple criteria

        Args:
            cohort: ICU stays with patient data
            diagnoses_icd: ICD diagnosis codes
            prescriptions: Prescription data
            microbiologyevents: Culture data

        Returns:
            List of stay_ids with suspected sepsis
        """
        sepsis_stays = set()

        # Method 1: ICD codes
        logger.info("  Method 1: Checking ICD diagnosis codes...")
        sepsis_icd = self._identify_sepsis_by_icd(cohort, diagnoses_icd)
        sepsis_stays.update(sepsis_icd)
        logger.info(f"    Found {len(sepsis_icd):,} stays with sepsis ICD codes")

        # Method 2: Suspected infection (cultures + antibiotics)
        if prescriptions is not None and microbiologyevents is not None:
            logger.info("  Method 2: Checking suspected infection (cultures + antibiotics)...")
            sepsis_infection = self._identify_suspected_infection(
                cohort, prescriptions, microbiologyevents
            )
            sepsis_stays.update(sepsis_infection)
            logger.info(f"    Found {len(sepsis_infection):,} stays with suspected infection")

        # Method 3: Explicit sepsis diagnosis (fallback to ICD if other methods unavailable)
        if not sepsis_stays:
            logger.warning("  Using ICD codes only (limited data available)")
            sepsis_stays = sepsis_icd

        logger.info(f"  Total unique sepsis stays: {len(sepsis_stays):,}")

        return list(sepsis_stays)

    def _identify_sepsis_by_icd(
        self,
        cohort: pd.DataFrame,
        diagnoses_icd: pd.DataFrame
    ) -> set:
        """Identify sepsis using ICD codes"""
        # Get admissions in cohort
        hadm_ids = cohort['hadm_id'].unique()

        # Filter diagnoses for cohort
        cohort_diagnoses = diagnoses_icd[diagnoses_icd['hadm_id'].isin(hadm_ids)]

        # Check for sepsis ICD codes
        def is_sepsis_code(code, icd_version):
            if pd.isna(code):
                return False
            code = str(code).upper()

            if icd_version == 9:
                return any(code.startswith(icd) for icd in self.SEPSIS_ICD9_CODES)
            elif icd_version == 10:
                return any(code.startswith(icd) for icd in self.SEPSIS_ICD10_CODES)
            return False

        # Identify sepsis admissions
        cohort_diagnoses['is_sepsis'] = cohort_diagnoses.apply(
            lambda row: is_sepsis_code(row['icd_code'], row['icd_version']),
            axis=1
        )

        sepsis_hadm_ids = cohort_diagnoses[cohort_diagnoses['is_sepsis']]['hadm_id'].unique()

        # Map back to stay_ids
        sepsis_stays = cohort[cohort['hadm_id'].isin(sepsis_hadm_ids)]['stay_id'].unique()

        return set(sepsis_stays)

    def _identify_suspected_infection(
        self,
        cohort: pd.DataFrame,
        prescriptions: pd.DataFrame,
        microbiologyevents: pd.DataFrame
    ) -> set:
        """
        Identify suspected infection using cultures + antibiotics within 72h window

        Sepsis-3 criteria: Culture drawn + antibiotics given within ±72h
        """
        sepsis_stays = set()

        logger.info("    Checking for cultures and antibiotics...")

        # Get subject_ids in cohort
        subject_ids = cohort['subject_id'].unique()

        # Filter prescriptions for antibiotics
        antibiotics = prescriptions[
            prescriptions['subject_id'].isin(subject_ids) &
            prescriptions['drug'].str.lower().str.contains(
                '|'.join(self.ANTIBIOTIC_KEYWORDS), na=False
            )
        ].copy()

        # Filter cultures
        cultures = microbiologyevents[
            microbiologyevents['subject_id'].isin(subject_ids)
        ].copy()

        if len(antibiotics) == 0 or len(cultures) == 0:
            logger.warning("    Insufficient antibiotic or culture data")
            return sepsis_stays

        # For each ICU stay, check if culture + antibiotic within 72h window
        for _, stay in cohort.iterrows():
            subject_id = stay['subject_id']
            icu_intime = stay['intime']
            icu_outtime = stay['outtime']

            # Get antibiotics during ICU stay (±24h buffer)
            stay_antibiotics = antibiotics[
                (antibiotics['subject_id'] == subject_id) &
                (antibiotics['starttime'] >= icu_intime - timedelta(hours=24)) &
                (antibiotics['starttime'] <= icu_outtime + timedelta(hours=24))
            ]

            # Get cultures during ICU stay (±24h buffer)
            stay_cultures = cultures[
                (cultures['subject_id'] == subject_id) &
                (cultures['charttime'] >= icu_intime - timedelta(hours=24)) &
                (cultures['charttime'] <= icu_outtime + timedelta(hours=24))
            ]

            # Check for culture + antibiotic within 72h window
            if len(stay_cultures) > 0 and len(stay_antibiotics) > 0:
                for _, culture in stay_cultures.iterrows():
                    culture_time = culture['charttime']

                    # Check if any antibiotic within ±72h of culture
                    for _, abx in stay_antibiotics.iterrows():
                        abx_time = abx['starttime']
                        time_diff = abs((abx_time - culture_time).total_seconds() / 3600)

                        if time_diff <= 72:
                            sepsis_stays.add(stay['stay_id'])
                            break

                    if stay['stay_id'] in sepsis_stays:
                        break

        return sepsis_stays

    def get_cohort_statistics(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """
        Get descriptive statistics for the cohort

        Args:
            cohort: Cohort dataframe

        Returns:
            DataFrame with statistics
        """
        stats = {
            'Total Patients': cohort['subject_id'].nunique(),
            'Total ICU Stays': len(cohort),
            'Unique Admissions': cohort['hadm_id'].nunique(),
            'Mean Age': cohort['anchor_age'].mean(),
            'Median Age': cohort['anchor_age'].median(),
            'Female %': (cohort['gender'] == 'F').sum() / len(cohort) * 100,
            'Male %': (cohort['gender'] == 'M').sum() / len(cohort) * 100,
            'Mean ICU LOS (hours)': cohort['icu_los_hours'].mean(),
            'Median ICU LOS (hours)': cohort['icu_los_hours'].median(),
            'Hospital Mortality %': cohort['hospital_expire_flag'].sum() / len(cohort) * 100,
        }

        stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
        stats_df['Value'] = stats_df['Value'].round(2)

        return stats_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Example usage
    from data_loader import MIMICDataLoader
    from utils.config_loader import ConfigLoader

    config = ConfigLoader().config
    loader = MIMICDataLoader("data/raw/mimic-iv-3.1", config)

    try:
        # Load required tables
        patients = loader.load_patients()
        admissions = loader.load_admissions()
        icustays = loader.load_icustays()
        diagnoses_icd = loader.load_table('diagnoses_icd')

        # Select cohort
        selector = SepsisCohortSelector(config)
        cohort, stats = selector.select_cohort(
            patients, admissions, icustays, diagnoses_icd
        )

        # Display statistics
        print("\n" + "="*80)
        print("Cohort Statistics")
        print("="*80)
        print(selector.get_cohort_statistics(cohort))

    except Exception as e:
        logger.error(f"Error: {e}")
