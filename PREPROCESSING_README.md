# MIMIC-IV Preprocessing Pipeline

Comprehensive preprocessing pipeline for MIMIC-IV v3.1 dataset that processes all 31 CSV files (22 hospital + 9 ICU) for reinforcement learning applications.

## Overview

This preprocessing pipeline:
1. **Loads all 31 MIMIC-IV CSV files** (hosp + icu directories)
2. **Selects sepsis cohort** using Sepsis-3 criteria
3. **Extracts 48+ features** from hospital and ICU data
4. **Cleans data** (handles outliers and missing values)
5. **Normalizes features** (z-score with log transforms)
6. **Validates data quality** at each stage
7. **Splits data** into train/validation/test sets
8. **Saves processed data** ready for RL training

## Directory Structure

```
RL/
├── data/
│   ├── raw/
│   │   └── mimic-iv-3.1/          # Your MIMIC-IV data goes here
│   │       ├── hosp/               # Hospital data (22 CSV files)
│   │       └── icu/                # ICU data (9 CSV files)
│   ├── processed/                  # Output: processed data
│   └── intermediate/               # Intermediate results (if saved)
├── configs/
│   └── config.yaml                 # Configuration file
├── src/
│   └── preprocessing/
│       ├── data_loader.py          # Loads all 31 CSV files
│       ├── cohort_selection.py     # Sepsis cohort selection
│       ├── feature_extraction_hosp.py  # Hospital features
│       ├── feature_extraction_icu.py   # ICU features
│       ├── data_cleaning.py        # Outlier/missing value handling
│       ├── normalization.py        # Z-score normalization
│       ├── data_validation.py      # Data quality validation
│       └── preprocessing_pipeline.py   # Main orchestrator
├── logs/                           # Preprocessing logs
└── run_preprocessing.py            # Main entry point
```

## Quick Start

### 1. Prepare Your Data

Place your MIMIC-IV v3.1 data in the following structure:

```
data/raw/mimic-iv-3.1/
├── hosp/
│   ├── admissions.csv
│   ├── patients.csv
│   ├── labevents.csv
│   ├── diagnoses_icd.csv
│   └── ... (18 more files)
└── icu/
    ├── icustays.csv
    ├── chartevents.csv
    ├── inputevents.csv
    └── ... (6 more files)
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing

**Basic usage:**
```bash
python run_preprocessing.py
```

**Specify data path:**
```bash
python run_preprocessing.py --data-path /path/to/your/mimic-iv-3.1
```

**Custom configuration:**
```bash
python run_preprocessing.py --config configs/custom_config.yaml
```

**Debug mode:**
```bash
python run_preprocessing.py --log-level DEBUG
```

## Configuration

Edit `configs/config.yaml` to customize:

### Data Source
```yaml
data_source:
  type: "csv"  # Use CSV files (not PostgreSQL)
  base_path: "data/raw/mimic-iv-3.1"
```

### Cohort Selection
```yaml
cohort:
  sepsis3_definition: true  # Use Sepsis-3 criteria
  min_age: 18
  max_age: 120
  min_icu_los_hours: 12
```

### Preprocessing Settings
```yaml
preprocessing:
  chunk_size: 100000  # Process large files in chunks
  missing_data:
    strategy: "forward_fill_then_median"
    max_missing_ratio: 0.5
  outliers:
    method: "iqr"  # IQR or zscore
    iqr_multiplier: 3.0
```

### Feature Normalization
```yaml
normalization:
  method: "zscore"
  continuous_features:
    log_transform:  # Log transform these before z-score
      - SpO2
      - BUN
      - Creatinine
      # ... more features
  binary_features:  # Subtract 0.5
    - gender
    - mechvent
```

### Data Split
```yaml
data_split:
  train: 0.70
  validation: 0.15
  test: 0.15
  random_seed: 42
  stratify_by_mortality: true
```

## Pipeline Stages

### Stage 1: Data Loading
- Loads all 31 CSV files from MIMIC-IV
- Parses datetime columns automatically
- Supports chunked loading for large files (chartevents, labevents)
- Caches frequently used tables

**Files loaded:**
- **Hospital (22 files):** admissions, patients, diagnoses_icd, labevents, prescriptions, etc.
- **ICU (9 files):** icustays, chartevents, inputevents, outputevents, etc.

### Stage 2: Cohort Selection
Identifies sepsis patients using:
- **Sepsis-3 criteria:**
  - Suspected infection (culture + antibiotics within 72h)
  - SOFA score increase ≥ 2 points
- **ICD codes:** Sepsis diagnosis codes (ICD-9/10)
- **Age filter:** 18-120 years
- **ICU LOS filter:** ≥12 hours

### Stage 3: Feature Extraction

**Demographics (4 features):**
- gender, age, weight, re_admission

**Vital Signs (10 features):**
- HR, SysBP, MeanBP, DiaBP, RR, Temp_C, SpO2, FiO2, GCS, mechvent

**Labs - Chemistry (11 features):**
- Potassium, Sodium, Chloride, Glucose, BUN, Creatinine, Magnesium, Calcium, SGOT, SGPT, Total_bili

**Labs - Hematology (6 features):**
- Hb, WBC_count, Platelets_count, PTT, PT, INR

**Labs - Blood Gas (6 features):**
- Arterial_pH, paO2, paCO2, Arterial_BE, HCO3, Arterial_lactate

**Fluid Balance (5 features):**
- input_total, input_4hourly, output_total, output_4hourly, cumulated_balance

**Derived Scores (4 features):**
- SOFA, SIRS, Shock_Index, PaO2_FiO2

### Stage 4: Data Cleaning
- **Outlier removal:**
  - Physiological range validation
  - IQR method (default: 3.0 × IQR)
  - Z-score method (optional)
- **Missing value imputation:**
  - Forward fill for temporal data
  - Median imputation
  - KNN imputation (optional)
- **Drop high-missing features:** >50% missing

### Stage 5: Train/Val/Test Split
- Splits by ICU stay (not by time window)
- Default: 70% train, 15% validation, 15% test
- Optional stratification by mortality

### Stage 6: Feature Normalization
- **Log transform:** For skewed features (SpO2, labs, etc.)
- **Z-score normalization:** Mean=0, Std=1
- **Binary features:** Subtract 0.5
- **Fitted on training data only**
- Saved for later use (inference)

### Stage 7: Data Validation
Validates at each stage:
- Missing values
- Data types
- Value ranges
- Duplicates
- Temporal consistency
- Feature completeness

## Output Files

After preprocessing, you'll have:

```
data/processed/
├── cohort.csv                      # Selected cohort metadata
├── train_features.csv              # Training features (raw)
├── train_features_normalized.csv   # Training features (normalized)
├── val_features.csv                # Validation features (raw)
├── val_features_normalized.csv     # Validation features (normalized)
├── test_features.csv               # Test features (raw)
├── test_features_normalized.csv    # Test features (normalized)
└── normalizer.pkl                  # Fitted normalizer (for inference)

data/intermediate/  # If --save-intermediate is used
├── cohort.csv
├── features_raw.csv
└── features_clean.csv

logs/
├── preprocessing_YYYYMMDD_HHMMSS.log
└── validation_report_YYYYMMDD_HHMMSS.txt
```

## Features in Output

Each output file contains:
- **Identifiers:** stay_id, time_window
- **48 state features** (as configured)
- **Temporal structure:** Multiple rows per ICU stay (one per time window)

Example row:
```csv
stay_id,time_window,gender,age,HR,SysBP,Temp_C,Glucose,Lactate,...
12345,0,1,65,95,110,37.2,120,2.1,...
12345,1,1,65,92,115,37.5,115,1.8,...
```

## Command Line Options

```bash
python run_preprocessing.py [OPTIONS]

Options:
  --data-path PATH        Path to MIMIC-IV data (default: from config)
  --config PATH           Configuration file (default: configs/config.yaml)
  --output-dir PATH       Output directory (default: data/processed)
  --save-intermediate     Save intermediate results (default: True)
  --no-validate           Skip validation (faster but less safe)
  --log-level LEVEL       Logging level: DEBUG, INFO, WARNING, ERROR
  --test-run              Run on subset for testing
  -h, --help              Show help message
```

## Monitoring Progress

The pipeline provides detailed logging:

```
2024-11-16 10:00:00 - INFO - ================================================================================
2024-11-16 10:00:00 - INFO - STAGE 1: DATA LOADING
2024-11-16 10:00:00 - INFO - ================================================================================
2024-11-16 10:00:05 - INFO - ✓ Loaded patients: 299,712 rows
2024-11-16 10:00:10 - INFO - ✓ Loaded admissions: 431,231 rows
...
2024-11-16 10:05:00 - INFO - ================================================================================
2024-11-16 10:05:00 - INFO - STAGE 2: COHORT SELECTION
2024-11-16 10:05:00 - INFO - ================================================================================
2024-11-16 10:05:30 - INFO - ✓ Final cohort: 15,432 ICU stays
...
```

## Troubleshooting

### Data Not Found
```
Error: Data path does not exist: data/raw/mimic-iv-3.1
```
**Solution:** Ensure MIMIC-IV data is in the correct location or specify `--data-path`

### Out of Memory
```
Error: MemoryError when loading chartevents
```
**Solution:** The pipeline uses chunked loading automatically. For very large datasets, you can:
- Reduce `chunk_size` in config
- Process on a machine with more RAM
- Use a subset for testing (`--test-run`)

### Missing Features
```
Warning: 10 expected features not extracted
```
**Solution:** This is normal if some lab values or vitals are not available in your cohort. The pipeline handles missing features gracefully.

## Advanced Usage

### Programmatic API

```python
from src.utils.config_loader import ConfigLoader
from src.preprocessing import MIMICPreprocessingPipeline

# Load config
config = ConfigLoader('configs/config.yaml').config

# Initialize pipeline
pipeline = MIMICPreprocessingPipeline(
    config=config,
    data_path='data/raw/mimic-iv-3.1'
)

# Run pipeline
stats = pipeline.run_full_pipeline(
    save_intermediate=True,
    validate_each_stage=True
)

# Access processed data
train_features = pipeline.train_data['features_normalized']
```

### Custom Feature Extraction

You can extend the feature extractors:

```python
from src.preprocessing import HospitalFeatureExtractor

class CustomHospitalExtractor(HospitalFeatureExtractor):
    def extract_custom_features(self, cohort, data):
        # Your custom logic here
        pass
```

## Performance

Expected processing time (approximate):
- Small dataset (1,000 ICU stays): 5-10 minutes
- Medium dataset (10,000 ICU stays): 30-60 minutes
- Full dataset (50,000+ ICU stays): 2-4 hours

Memory usage:
- Minimum: 8 GB RAM
- Recommended: 16 GB RAM
- Large datasets: 32 GB RAM

## Citation

If you use this preprocessing pipeline, please cite:

```bibtex
@software{mimic_iv_preprocessing,
  title={MIMIC-IV Preprocessing Pipeline for Reinforcement Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/RL}
}
```

## License

This project follows the MIMIC-IV data use agreement. Ensure you have proper access to MIMIC-IV data before using this pipeline.

## Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review the validation report
3. Open an issue on GitHub
4. Contact the maintainers

## Changelog

### v1.0.0 (2024-11-16)
- Initial release
- Support for all 31 MIMIC-IV v3.1 CSV files
- Sepsis-3 cohort selection
- 48+ feature extraction
- Comprehensive data cleaning and normalization
- Data validation at each stage
- Train/val/test splitting
