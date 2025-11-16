"""
Main Preprocessing Pipeline
Orchestrates the complete preprocessing workflow for MIMIC-IV data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime
import pickle

# Import preprocessing modules
from .data_loader import MIMICDataLoader
from .cohort_selection import SepsisCohortSelector
from .feature_extraction_hosp import HospitalFeatureExtractor
from .feature_extraction_icu import ICUFeatureExtractor
from .data_cleaning import DataCleaner
from .normalization import FeatureNormalizer
from .data_validation import DataValidator

logger = logging.getLogger(__name__)


class MIMICPreprocessingPipeline:
    """
    Complete preprocessing pipeline for MIMIC-IV v3.1 data

    Pipeline stages:
    1. Data Loading (all 31 CSV files)
    2. Cohort Selection (sepsis patients)
    3. Feature Extraction (hospital + ICU data)
    4. Data Cleaning (outliers, missing values)
    5. Feature Normalization (z-score)
    6. Data Validation
    7. Train/Val/Test Split
    8. Save processed data
    """

    def __init__(self, config: Dict, data_path: str):
        """
        Initialize preprocessing pipeline

        Args:
            config: Configuration dictionary
            data_path: Path to MIMIC-IV data directory
        """
        self.config = config
        self.data_path = Path(data_path)

        # Output directories
        self.output_config = config.get('output', {})
        self.processed_dir = Path(self.output_config.get('processed_dir', 'data/processed'))
        self.intermediate_dir = Path(self.output_config.get('intermediate_dir', 'data/intermediate'))
        self.log_dir = Path(self.output_config.get('log_dir', 'logs'))

        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize modules
        self.loader = MIMICDataLoader(self.data_path, config)
        self.cohort_selector = SepsisCohortSelector(config)
        self.hosp_extractor = HospitalFeatureExtractor(config)
        self.icu_extractor = ICUFeatureExtractor(config)
        self.cleaner = DataCleaner(config)
        self.normalizer = FeatureNormalizer(config)
        self.validator = DataValidator(config)

        # Pipeline state
        self.cohort = None
        self.features = None
        self.features_clean = None
        self.features_normalized = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Validation reports
        self.validation_reports = {}

        logger.info("="*80)
        logger.info("MIMIC-IV Preprocessing Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Output directory: {self.processed_dir}")
        logger.info(f"Intermediate directory: {self.intermediate_dir}")

    def run_full_pipeline(
        self,
        save_intermediate: bool = True,
        validate_each_stage: bool = True
    ) -> Dict:
        """
        Run the complete preprocessing pipeline

        Args:
            save_intermediate: Save intermediate results
            validate_each_stage: Validate data at each stage

        Returns:
            Dictionary with pipeline statistics
        """
        start_time = datetime.now()
        logger.info("\n" + "="*80)
        logger.info("STARTING FULL PREPROCESSING PIPELINE")
        logger.info("="*80)
        logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        stats = {}

        try:
            # Stage 1: Load Data
            logger.info("\n" + "▶"*40)
            logger.info("STAGE 1: DATA LOADING")
            logger.info("▶"*40)
            stage_start = datetime.now()
            self._load_data()
            stats['stage_1_duration'] = (datetime.now() - stage_start).total_seconds()

            # Stage 2: Cohort Selection
            logger.info("\n" + "▶"*40)
            logger.info("STAGE 2: COHORT SELECTION")
            logger.info("▶"*40)
            stage_start = datetime.now()
            cohort_stats = self._select_cohort()
            stats['cohort_stats'] = cohort_stats
            stats['stage_2_duration'] = (datetime.now() - stage_start).total_seconds()

            if validate_each_stage:
                is_valid, report = self.validator.validate_cohort(self.cohort)
                self.validation_reports['cohort'] = report
                if not is_valid:
                    raise ValueError("Cohort validation failed")

            if save_intermediate:
                self._save_intermediate('cohort', self.cohort)

            # Stage 3: Feature Extraction
            logger.info("\n" + "▶"*40)
            logger.info("STAGE 3: FEATURE EXTRACTION")
            logger.info("▶"*40)
            stage_start = datetime.now()
            self._extract_features()
            stats['stage_3_duration'] = (datetime.now() - stage_start).total_seconds()

            if validate_each_stage:
                is_valid, report = self.validator.validate_features(self.features, stage='extraction')
                self.validation_reports['extraction'] = report

            if save_intermediate:
                self._save_intermediate('features_raw', self.features)

            # Stage 4: Data Cleaning
            logger.info("\n" + "▶"*40)
            logger.info("STAGE 4: DATA CLEANING")
            logger.info("▶"*40)
            stage_start = datetime.now()
            cleaning_stats = self._clean_data()
            stats['cleaning_stats'] = cleaning_stats
            stats['stage_4_duration'] = (datetime.now() - stage_start).total_seconds()

            if validate_each_stage:
                is_valid, report = self.validator.validate_features(self.features_clean, stage='cleaning')
                self.validation_reports['cleaning'] = report

            if save_intermediate:
                self._save_intermediate('features_clean', self.features_clean)

            # Stage 5: Train/Val/Test Split
            logger.info("\n" + "▶"*40)
            logger.info("STAGE 5: TRAIN/VAL/TEST SPLIT")
            logger.info("▶"*40)
            stage_start = datetime.now()
            split_stats = self._split_data()
            stats['split_stats'] = split_stats
            stats['stage_5_duration'] = (datetime.now() - stage_start).total_seconds()

            # Stage 6: Normalization (fit on train, transform all)
            logger.info("\n" + "▶"*40)
            logger.info("STAGE 6: FEATURE NORMALIZATION")
            logger.info("▶"*40)
            stage_start = datetime.now()
            self._normalize_features()
            stats['stage_6_duration'] = (datetime.now() - stage_start).total_seconds()

            if validate_each_stage:
                is_valid, report = self.validator.validate_features(
                    self.train_data['features_normalized'], stage='normalization'
                )
                self.validation_reports['normalization'] = report

            # Stage 7: Save Processed Data
            logger.info("\n" + "▶"*40)
            logger.info("STAGE 7: SAVING PROCESSED DATA")
            logger.info("▶"*40)
            stage_start = datetime.now()
            self._save_processed_data()
            stats['stage_7_duration'] = (datetime.now() - stage_start).total_seconds()

            # Generate final validation report
            if validate_each_stage:
                validation_report = self.validator.generate_validation_report(
                    self.validation_reports.get('cohort', {}),
                    self.validation_reports.get('extraction', {}),
                    self.validation_reports.get('cleaning', {}),
                    self.validation_reports.get('normalization', {})
                )
                logger.info("\n" + validation_report)

                # Save validation report
                report_path = self.log_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(report_path, 'w') as f:
                    f.write(validation_report)
                logger.info(f"\n✓ Validation report saved to {report_path}")

            # Pipeline completion
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            stats['total_duration'] = total_duration

            logger.info("\n" + "="*80)
            logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"Total duration: {total_duration/60:.2f} minutes")
            logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*80)

            return stats

        except Exception as e:
            logger.error(f"\n{'='*80}")
            logger.error("PREPROCESSING PIPELINE FAILED")
            logger.error(f"{'='*80}")
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _load_data(self):
        """Load all required MIMIC-IV CSV files"""
        logger.info("Loading MIMIC-IV data files...")

        # Get file information
        file_info = self.loader.get_file_info()
        logger.info(f"\nFound {len(file_info)} CSV files:")
        logger.info(f"Total size: {file_info['size_mb'].sum():.2f} MB\n")

        # Load core tables
        self.patients = self.loader.load_patients()
        logger.info(f"✓ Loaded patients: {len(self.patients):,} rows")

        self.admissions = self.loader.load_admissions()
        logger.info(f"✓ Loaded admissions: {len(self.admissions):,} rows")

        self.icustays = self.loader.load_icustays()
        logger.info(f"✓ Loaded icustays: {len(self.icustays):,} rows")

        self.diagnoses_icd = self.loader.load_table('diagnoses_icd')
        logger.info(f"✓ Loaded diagnoses_icd: {len(self.diagnoses_icd):,} rows")

        self.d_icd_diagnoses = self.loader.load_d_icd_diagnoses()
        logger.info(f"✓ Loaded d_icd_diagnoses: {len(self.d_icd_diagnoses):,} rows")

        self.d_labitems = self.loader.load_d_labitems()
        logger.info(f"✓ Loaded d_labitems: {len(self.d_labitems):,} rows")

        self.d_items = self.loader.load_d_items()
        logger.info(f"✓ Loaded d_items: {len(self.d_items):,} rows")

        logger.info("\n✓ Core data files loaded successfully")

    def _select_cohort(self) -> Dict:
        """Select sepsis cohort"""
        # Try to load additional tables for sepsis identification
        try:
            prescriptions = self.loader.load_table('prescriptions')
            microbiologyevents = self.loader.load_table('microbiologyevents')
        except:
            prescriptions = None
            microbiologyevents = None
            logger.warning("Could not load prescriptions or microbiologyevents - using ICD codes only")

        self.cohort, stats = self.cohort_selector.select_cohort(
            self.patients,
            self.admissions,
            self.icustays,
            self.diagnoses_icd,
            prescriptions,
            microbiologyevents
        )

        # Get cohort statistics
        cohort_stats_df = self.cohort_selector.get_cohort_statistics(self.cohort)
        logger.info("\n" + str(cohort_stats_df))

        return stats

    def _extract_features(self):
        """Extract features from hospital and ICU data"""
        logger.info("Extracting features from all data sources...\n")

        # Extract demographics
        demographics = self.hosp_extractor.extract_demographics(
            self.cohort, self.patients, self.admissions
        )

        # Load large files with filtering for cohort
        logger.info("\nLoading large event tables (filtered for cohort)...")
        cohort_subjects = self.cohort['subject_id'].unique()

        # Get lab itemids for filtering
        lab_itemid_map = self.hosp_extractor._create_lab_itemid_map(self.d_labitems)
        lab_itemids = []
        for itemids in lab_itemid_map.values():
            lab_itemids.extend(itemids)

        # Load labevents (large file)
        try:
            labevents = self.loader.load_labevents(
                itemids=lab_itemids,
                subject_ids=cohort_subjects.tolist()
            )
            logger.info("Extracting lab features...")
            lab_features = self.hosp_extractor.extract_lab_features(
                self.cohort, labevents, self.d_labitems
            )
        except Exception as e:
            logger.warning(f"Could not extract lab features: {e}")
            lab_features = pd.DataFrame()

        # Get vital sign itemids for filtering
        vital_itemid_map = self.icu_extractor._create_vital_itemid_map(self.d_items)
        vital_itemids = []
        for itemids in vital_itemid_map.values():
            vital_itemids.extend(itemids)

        # Load chartevents (large file)
        try:
            chartevents = self.loader.load_chartevents(
                itemids=vital_itemids,
                subject_ids=cohort_subjects.tolist()
            )
            logger.info("Extracting vital signs...")
            vital_signs = self.icu_extractor.extract_vital_signs(
                self.cohort, chartevents, self.d_items
            )

            logger.info("Extracting mechanical ventilation status...")
            mechvent = self.icu_extractor.extract_mechanical_ventilation(
                self.cohort, chartevents, self.d_items
            )
        except Exception as e:
            logger.warning(f"Could not extract vital signs: {e}")
            vital_signs = pd.DataFrame()
            mechvent = pd.DataFrame()

        # Load fluid data
        try:
            inputevents = self.loader.load_inputevents(subject_ids=cohort_subjects.tolist())
            outputevents = self.loader.load_outputevents(subject_ids=cohort_subjects.tolist())

            logger.info("Extracting fluid balance...")
            fluid_balance = self.icu_extractor.extract_fluid_balance(
                self.cohort, inputevents, outputevents
            )

            logger.info("Extracting vasopressor doses...")
            vasopressors = self.icu_extractor.extract_vasopressor_dose(
                self.cohort, inputevents
            )
        except Exception as e:
            logger.warning(f"Could not extract fluid/vasopressor data: {e}")
            fluid_balance = pd.DataFrame()
            vasopressors = pd.DataFrame()

        # Merge all features
        logger.info("\nMerging all features...")
        features_list = []

        # Start with vital signs (has stay_id + time_window)
        if not vital_signs.empty:
            self.features = vital_signs
        else:
            # Create empty dataframe with time windows
            self.features = pd.DataFrame({
                'stay_id': self.cohort['stay_id'].values,
                'time_window': 0
            })

        # Merge temporal features
        for df in [lab_features, mechvent, fluid_balance, vasopressors]:
            if not df.empty:
                self.features = self.features.merge(df, on=['stay_id', 'time_window'], how='outer')

        # Merge demographics (non-temporal)
        if not demographics.empty:
            self.features = self.features.merge(demographics, on='stay_id', how='left')

        logger.info(f"\n✓ Feature extraction complete: {len(self.features):,} observations")
        logger.info(f"  Columns: {len(self.features.columns)}")
        logger.info(f"  Features: {self.features.columns.tolist()}")

    def _clean_data(self) -> Dict:
        """Clean data (outliers and missing values)"""
        self.features_clean, cleaning_stats = self.cleaner.clean_data(
            self.features,
            temporal=True
        )
        return cleaning_stats

    def _split_data(self) -> Dict:
        """Split data into train/val/test sets"""
        logger.info("Splitting data into train/val/test sets...")

        split_config = self.config.get('data_split', {})
        train_ratio = split_config.get('train', 0.70)
        val_ratio = split_config.get('validation', 0.15)
        test_ratio = split_config.get('test', 0.15)
        random_seed = split_config.get('random_seed', 42)

        # Split by ICU stay (not by time window)
        unique_stays = self.features_clean['stay_id'].unique()
        n_stays = len(unique_stays)

        # Shuffle
        np.random.seed(random_seed)
        shuffled_stays = np.random.permutation(unique_stays)

        # Split
        n_train = int(n_stays * train_ratio)
        n_val = int(n_stays * val_ratio)

        train_stays = shuffled_stays[:n_train]
        val_stays = shuffled_stays[n_train:n_train+n_val]
        test_stays = shuffled_stays[n_train+n_val:]

        # Create splits
        self.train_data = {
            'features': self.features_clean[self.features_clean['stay_id'].isin(train_stays)].copy()
        }
        self.val_data = {
            'features': self.features_clean[self.features_clean['stay_id'].isin(val_stays)].copy()
        }
        self.test_data = {
            'features': self.features_clean[self.features_clean['stay_id'].isin(test_stays)].copy()
        }

        stats = {
            'total_stays': n_stays,
            'train_stays': len(train_stays),
            'val_stays': len(val_stays),
            'test_stays': len(test_stays),
            'train_observations': len(self.train_data['features']),
            'val_observations': len(self.val_data['features']),
            'test_observations': len(self.test_data['features'])
        }

        logger.info(f"\n✓ Data split complete:")
        logger.info(f"  Train: {stats['train_stays']:>6,} stays ({stats['train_observations']:>8,} observations)")
        logger.info(f"  Val:   {stats['val_stays']:>6,} stays ({stats['val_observations']:>8,} observations)")
        logger.info(f"  Test:  {stats['test_stays']:>6,} stays ({stats['test_observations']:>8,} observations)")

        return stats

    def _normalize_features(self):
        """Normalize features (fit on train, transform all)"""
        logger.info("Normalizing features...")

        # Fit on training data
        self.normalizer.fit(self.train_data['features'])

        # Transform all splits
        self.train_data['features_normalized'] = self.normalizer.transform(self.train_data['features'])
        self.val_data['features_normalized'] = self.normalizer.transform(self.val_data['features'])
        self.test_data['features_normalized'] = self.normalizer.transform(self.test_data['features'])

        # Save normalizer
        normalizer_path = self.processed_dir / 'normalizer.pkl'
        self.normalizer.save(str(normalizer_path))

        logger.info(f"✓ Normalization complete")
        logger.info(f"  Normalizer saved to {normalizer_path}")

    def _save_processed_data(self):
        """Save all processed data"""
        logger.info("Saving processed data...")

        # Save train/val/test splits
        for split_name, split_data in [('train', self.train_data), ('val', self.val_data), ('test', self.test_data)]:
            # Save raw features
            features_path = self.processed_dir / f'{split_name}_features.csv'
            split_data['features'].to_csv(features_path, index=False)
            logger.info(f"✓ Saved {split_name} features to {features_path}")

            # Save normalized features
            features_norm_path = self.processed_dir / f'{split_name}_features_normalized.csv'
            split_data['features_normalized'].to_csv(features_norm_path, index=False)
            logger.info(f"✓ Saved {split_name} normalized features to {features_norm_path}")

        # Save cohort
        cohort_path = self.processed_dir / 'cohort.csv'
        self.cohort.to_csv(cohort_path, index=False)
        logger.info(f"✓ Saved cohort to {cohort_path}")

        logger.info(f"\n✓ All processed data saved to {self.processed_dir}")

    def _save_intermediate(self, name: str, data: pd.DataFrame):
        """Save intermediate results"""
        filepath = self.intermediate_dir / f'{name}.csv'
        data.to_csv(filepath, index=False)
        logger.info(f"  Saved intermediate: {filepath}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/preprocessing.log'),
            logging.StreamHandler()
        ]
    )

    # Load configuration
    from utils.config_loader import ConfigLoader
    config = ConfigLoader().config

    # Initialize and run pipeline
    data_path = config.get('data_source', {}).get('base_path', 'data/raw/mimic-iv-3.1')
    pipeline = MIMICPreprocessingPipeline(config, data_path)

    # Run full pipeline
    stats = pipeline.run_full_pipeline(
        save_intermediate=True,
        validate_each_stage=True
    )

    print("\n" + "="*80)
    print("PREPROCESSING STATISTICS")
    print("="*80)
    for key, value in stats.items():
        print(f"{key}: {value}")
