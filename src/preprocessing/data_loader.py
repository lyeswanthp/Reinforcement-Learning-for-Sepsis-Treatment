"""
MIMIC-IV Data Loader
Handles loading all 31 CSV files from MIMIC-IV v3.1 dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from functools import lru_cache
import warnings

logger = logging.getLogger(__name__)


class MIMICDataLoader:
    """
    Comprehensive data loader for MIMIC-IV v3.1 dataset
    Handles all 31 CSV files (22 hosp + 9 icu)
    """

    # Define all expected files
    HOSP_FILES = [
        'admissions', 'd_hcpcs', 'd_icd_diagnoses', 'd_icd_procedures',
        'd_labitems', 'diagnoses_icd', 'drgcodes', 'emar', 'emar_detail',
        'hcpcsevents', 'labevents', 'microbiologyevents', 'omr', 'patients',
        'pharmacy', 'poe', 'poe_detail', 'prescriptions', 'procedures_icd',
        'provider', 'services', 'transfers'
    ]

    ICU_FILES = [
        'caregiver', 'chartevents', 'd_items', 'datetimeevents', 'icustays',
        'ingredientevents', 'inputevents', 'outputevents', 'procedureevents'
    ]

    # Date/time columns that need parsing
    DATETIME_COLUMNS = {
        'admissions': ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime'],
        'transfers': ['intime', 'outtime'],
        'icustays': ['intime', 'outtime'],
        'chartevents': ['charttime', 'storetime'],
        'datetimeevents': ['charttime', 'storetime'],
        'inputevents': ['starttime', 'endtime', 'storetime'],
        'outputevents': ['charttime', 'storetime'],
        'procedureevents': ['starttime', 'endtime', 'storetime'],
        'labevents': ['charttime', 'storetime'],
        'microbiologyevents': ['chartdate', 'charttime', 'storedate', 'storetime'],
        'prescriptions': ['starttime', 'stoptime'],
        'emar': ['charttime', 'scheduletime', 'storetime'],
        'emar_detail': ['charttime'],
        'poe': ['ordertime'],
        'poe_detail': ['field_value'],
        'pharmacy': ['starttime', 'stoptime'],
        'patients': ['dod'],
    }

    def __init__(self, base_path: Union[str, Path], config: Optional[Dict] = None):
        """
        Initialize data loader

        Args:
            base_path: Base path to MIMIC-IV data (contains hosp/ and icu/ folders)
            config: Optional configuration dictionary
        """
        self.base_path = Path(base_path)
        self.config = config or {}
        self.chunk_size = self.config.get('preprocessing', {}).get('chunk_size', 100000)

        # Verify directory structure
        self.hosp_path = self.base_path / 'hosp'
        self.icu_path = self.base_path / 'icu'

        self._verify_structure()
        logger.info(f"MIMICDataLoader initialized with base path: {self.base_path}")

    def _verify_structure(self):
        """Verify that hosp/ and icu/ directories exist"""
        if not self.hosp_path.exists():
            raise FileNotFoundError(f"Hospital data directory not found: {self.hosp_path}")
        if not self.icu_path.exists():
            raise FileNotFoundError(f"ICU data directory not found: {self.icu_path}")

        logger.info("✓ Directory structure verified")

    def get_available_files(self) -> Dict[str, List[str]]:
        """
        Get list of available CSV files

        Returns:
            Dictionary with 'hosp' and 'icu' keys containing lists of available files
        """
        available = {
            'hosp': [],
            'icu': []
        }

        # Check hospital files
        for file in self.HOSP_FILES:
            file_path = self.hosp_path / f"{file}.csv"
            if file_path.exists():
                available['hosp'].append(file)

        # Check ICU files
        for file in self.ICU_FILES:
            file_path = self.icu_path / f"{file}.csv"
            if file_path.exists():
                available['icu'].append(file)

        logger.info(f"Found {len(available['hosp'])}/22 hospital files")
        logger.info(f"Found {len(available['icu'])}/9 ICU files")

        return available

    def load_table(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        chunksize: Optional[int] = None,
        nrows: Optional[int] = None,
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load a single table from CSV

        Args:
            table_name: Name of the table (e.g., 'admissions', 'chartevents')
            columns: Specific columns to load (None = all columns)
            chunksize: If specified, returns an iterator
            nrows: Number of rows to read (for testing)
            parse_dates: Whether to parse datetime columns

        Returns:
            DataFrame or iterator of DataFrames
        """
        # Determine file path
        if table_name in self.HOSP_FILES:
            file_path = self.hosp_path / f"{table_name}.csv"
        elif table_name in self.ICU_FILES:
            file_path = self.icu_path / f"{table_name}.csv"
        else:
            raise ValueError(f"Unknown table: {table_name}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Prepare datetime parsing
        date_cols = self.DATETIME_COLUMNS.get(table_name, []) if parse_dates else []

        # Load data
        logger.info(f"Loading {table_name} from {file_path}")

        try:
            df = pd.read_csv(
                file_path,
                usecols=columns,
                chunksize=chunksize,
                nrows=nrows,
                parse_dates=date_cols,
                low_memory=False
            )

            if chunksize is None:
                logger.info(f"✓ Loaded {table_name}: {len(df):,} rows, {len(df.columns)} columns")
            else:
                logger.info(f"✓ Created iterator for {table_name} with chunksize={chunksize}")

            return df

        except Exception as e:
            logger.error(f"Failed to load {table_name}: {e}")
            raise

    @lru_cache(maxsize=10)
    def load_patients(self) -> pd.DataFrame:
        """
        Load patients table with caching

        Returns:
            DataFrame with patient demographics
        """
        return self.load_table('patients')

    @lru_cache(maxsize=10)
    def load_admissions(self) -> pd.DataFrame:
        """
        Load admissions table with caching

        Returns:
            DataFrame with admission records
        """
        return self.load_table('admissions')

    @lru_cache(maxsize=10)
    def load_icustays(self) -> pd.DataFrame:
        """
        Load ICU stays table with caching

        Returns:
            DataFrame with ICU stay records
        """
        return self.load_table('icustays')

    @lru_cache(maxsize=10)
    def load_d_icd_diagnoses(self) -> pd.DataFrame:
        """Load ICD diagnosis dictionary"""
        return self.load_table('d_icd_diagnoses')

    @lru_cache(maxsize=10)
    def load_d_labitems(self) -> pd.DataFrame:
        """Load lab items dictionary"""
        return self.load_table('d_labitems')

    @lru_cache(maxsize=10)
    def load_d_items(self) -> pd.DataFrame:
        """Load chart items dictionary"""
        return self.load_table('d_items')

    def load_chartevents(
        self,
        itemids: Optional[List[int]] = None,
        subject_ids: Optional[List[int]] = None,
        chunksize: Optional[int] = None
    ) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        """
        Load chartevents (large file - use filtering)

        Args:
            itemids: Filter by specific item IDs
            subject_ids: Filter by specific subject IDs
            chunksize: Process in chunks

        Returns:
            DataFrame or chunk iterator
        """
        if chunksize is None:
            chunksize = self.chunk_size

        logger.info(f"Loading chartevents (chunksize={chunksize})")

        # Load in chunks and filter
        chunks = []
        for chunk in self.load_table('chartevents', chunksize=chunksize):
            if itemids is not None:
                chunk = chunk[chunk['itemid'].isin(itemids)]
            if subject_ids is not None:
                chunk = chunk[chunk['subject_id'].isin(subject_ids)]

            if len(chunk) > 0:
                chunks.append(chunk)

        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"✓ Loaded chartevents: {len(df):,} rows after filtering")
            return df
        else:
            logger.warning("No chartevents data after filtering")
            return pd.DataFrame()

    def load_labevents(
        self,
        itemids: Optional[List[int]] = None,
        subject_ids: Optional[List[int]] = None,
        chunksize: Optional[int] = None
    ) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        """
        Load labevents (large file - use filtering)

        Args:
            itemids: Filter by specific item IDs
            subject_ids: Filter by specific subject IDs
            chunksize: Process in chunks

        Returns:
            DataFrame or chunk iterator
        """
        if chunksize is None:
            chunksize = self.chunk_size

        logger.info(f"Loading labevents (chunksize={chunksize})")

        chunks = []
        for chunk in self.load_table('labevents', chunksize=chunksize):
            if itemids is not None:
                chunk = chunk[chunk['itemid'].isin(itemids)]
            if subject_ids is not None:
                chunk = chunk[chunk['subject_id'].isin(subject_ids)]

            if len(chunk) > 0:
                chunks.append(chunk)

        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"✓ Loaded labevents: {len(df):,} rows after filtering")
            return df
        else:
            logger.warning("No labevents data after filtering")
            return pd.DataFrame()

    def load_inputevents(
        self,
        subject_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Load input events (IV fluids, medications)

        Args:
            subject_ids: Filter by specific subject IDs

        Returns:
            DataFrame with input events
        """
        df = self.load_table('inputevents')

        if subject_ids is not None:
            df = df[df['subject_id'].isin(subject_ids)]

        logger.info(f"✓ Loaded inputevents: {len(df):,} rows")
        return df

    def load_outputevents(
        self,
        subject_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Load output events (urine output, etc.)

        Args:
            subject_ids: Filter by specific subject IDs

        Returns:
            DataFrame with output events
        """
        df = self.load_table('outputevents')

        if subject_ids is not None:
            df = df[df['subject_id'].isin(subject_ids)]

        logger.info(f"✓ Loaded outputevents: {len(df):,} rows")
        return df

    def get_file_info(self) -> pd.DataFrame:
        """
        Get information about all CSV files

        Returns:
            DataFrame with file information (name, size, row count estimate)
        """
        info_list = []

        for file in self.HOSP_FILES:
            file_path = self.hosp_path / f"{file}.csv"
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                info_list.append({
                    'category': 'hosp',
                    'table': file,
                    'size_mb': round(size_mb, 2),
                    'path': str(file_path)
                })

        for file in self.ICU_FILES:
            file_path = self.icu_path / f"{file}.csv"
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                info_list.append({
                    'category': 'icu',
                    'table': file,
                    'size_mb': round(size_mb, 2),
                    'path': str(file_path)
                })

        info_df = pd.DataFrame(info_list)
        info_df = info_df.sort_values('size_mb', ascending=False)

        logger.info(f"\nTotal data size: {info_df['size_mb'].sum():.2f} MB")

        return info_df


if __name__ == "__main__":
    # Test the data loader
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    base_path = "data/raw/mimic-iv-3.1"
    loader = MIMICDataLoader(base_path)

    # Get file information
    print("\n" + "="*80)
    print("MIMIC-IV Data Files")
    print("="*80)
    info = loader.get_file_info()
    print(info.to_string(index=False))

    # Load sample tables
    print("\n" + "="*80)
    print("Loading Sample Tables")
    print("="*80)

    try:
        patients = loader.load_patients()
        print(f"\nPatients: {len(patients):,} rows")
        print(patients.head())

        admissions = loader.load_admissions()
        print(f"\nAdmissions: {len(admissions):,} rows")
        print(admissions.head())

        icustays = loader.load_icustays()
        print(f"\nICU Stays: {len(icustays):,} rows")
        print(icustays.head())

    except FileNotFoundError as e:
        print(f"\nNote: {e}")
        print("Please ensure MIMIC-IV data is in the correct location")
