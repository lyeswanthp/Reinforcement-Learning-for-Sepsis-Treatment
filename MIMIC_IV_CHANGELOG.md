# MIMIC-IV v3.1 Changelog Summary

This is the change log for MIMIC-IV v3.1 (current version used in our project).

## Key Version Information

**Current Version**: MIMIC-IV v3.1 (Released October 2024)
**Baseline Used**: MIMIC-III v1.4

## Critical Differences from MIMIC-III

### Schema Changes (v2.0 - Major Update)
- **Core module removed**: admissions, patients, transfers now in `hosp` module
- **Neonates removed**: Separate NICU dataset
- Module structure: `hosp`, `icu` modules

### Database Statistics (v3.1)
- Patients: 364,627
- Admissions: 546,028  
- ICU stays: 94,458
- Date range: Now includes 2020-2022 stays

### Important Tables for Our Project

#### icu module
- `icustays`: ICU stay information
- `chartevents`: Time series physiological data
- `inputevents`: IV fluids and medications
- `ingredientevents`: NEW - Detailed IV ingredient info
- `outputevents`: Output measurements

#### hosp module  
- `patients`: Demographics, mortality
- `admissions`: Hospital admission info
- `labevents`: Laboratory measurements
- `microbiologyevents`: Microbiology cultures
- `omr`: NEW - Height, weight, BMI from outpatient
- `prescriptions`: Medication prescriptions

### Key Fixes in v3.1 (October 2024)
- labevents itemid values corrected to match v2.2
- Foreign key constraints fixed
- Only 8 tables modified from v3.0

## Mapping Strategy

The baseline uses MIMIC-III which had a flatter structure. We need to:
1. Map MIMIC-III table names → MIMIC-IV table names
2. Map column names (some have changed)
3. Handle new/removed features
4. Adjust itemid values for labs

