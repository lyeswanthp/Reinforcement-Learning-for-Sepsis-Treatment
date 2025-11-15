# MIMIC-III to MIMIC-IV v3.1 Feature Mapping

## Overview
This document maps the 47 baseline features from the AI Clinician (MIMIC-III) to MIMIC-IV v3.1.

## Database Structure Changes

### MIMIC-III Structure
```
mimiciii.patients
mimiciii.admissions
mimiciii.icustays
mimiciii.chartevents
mimiciii.labevents
mimiciii.inputevents_mv  (MetaVision)
mimiciii.inputevents_cv  (CareVue)
mimiciii.outputevents
mimiciii.prescriptions
mimiciii.microbiologyevents
```

### MIMIC-IV v3.1 Structure
```
mimic_hosp.patients
mimic_hosp.admissions
mimic_icu.icustays
mimic_icu.chartevents
mimic_hosp.labevents
mimic_icu.inputevents     (UNIFIED - no more _mv/_cv split)
mimic_icu.ingredientevents (NEW)
mimic_icu.outputevents
mimic_hosp.prescriptions
mimic_hosp.microbiologyevents
mimic_hosp.omr             (NEW - height, weight, BMI)
```

## Feature List (47 Baseline Features)

### Demographics (4 features)
| Feature | MIMIC-III Source | MIMIC-IV v3.1 Source | Notes |
|---------|------------------|----------------------|-------|
| gender | patients.gender | hosp.patients.gender | M→1, F→2 |
| age | Computed from dob | Computed using anchor_year | MIMIC-IV uses de-identified ages |
| re_admission | Computed from admissions | hosp.admissions | ROW_NUMBER() over admissions |
| elixhauser | public.elixhauser_quan | Need to compute | Comorbidity score |

### Vital Signs (10 features)
| Feature | MIMIC-III itemid | MIMIC-IV itemid | Source Table |
|---------|------------------|-----------------|--------------|
| HR (Heart Rate) | 211, 220045 | 220045 | icu.chartevents |
| SysBP | 51, 442, 455, 6701, 220179, 220050 | 220179, 220050 | icu.chartevents |
| MeanBP | 52, 443, 456, 6702, 220181, 220052 | 220181, 220052 | icu.chartevents |
| DiaBP | 8368, 8440, 8441, 8555, 220180, 220051 | 220180, 220051 | icu.chartevents |
| RR (Resp Rate) | 618, 615, 220210, 224690 | 220210, 224690 | icu.chartevents |
| Temp_C | 223761, 678, 223762, 676 | 223761, 223762 | icu.chartevents |
| SpO2 | 646, 220277 | 220277 | icu.chartevents |
| FiO2 | 2981, 3420, 3422, 223835, 727 | 223835, 227287 | icu.chartevents |
| GCS | 198, 226755, 226756, 226757 | 226755, 226756, 226757 | icu.chartevents |
| mechvent | Computed from vent items | icu.chartevents | Binary indicator |

### Laboratory Values (22 features)
| Feature | MIMIC-III itemid | MIMIC-IV itemid | Source Table |
|---------|------------------|-----------------|--------------|
| Potassium | 50822, 50971, 227442 | 50822, 50971 | hosp.labevents |
| Sodium | 50824, 50983, 227464 | 50824, 50983 | hosp.labevents |
| Chloride | 50806, 50902, 220602 | 50806, 50902 | hosp.labevents |
| Glucose | 50809, 50931, 225664 | 50809, 50931 | hosp.labevents |
| BUN | 51006, 225624 | 51006 | hosp.labevents |
| Creatinine | 50912, 220615 | 50912 | hosp.labevents |
| Magnesium | 50960, 220635 | 50960 | hosp.labevents |
| Calcium | 50893, 50808 | 50893, 50808 | hosp.labevents |
| Ionised_Ca | 50808 | 50808 | hosp.labevents |
| CO2_mEqL | 50804, 50803 | 50804, 50803 | hosp.labevents |
| SGOT (AST) | 50878 | 50878 | hosp.labevents |
| SGPT (ALT) | 50861 | 50861 | hosp.labevents |
| Total_bili | 50885 | 50885 | hosp.labevents |
| Albumin | 50862 | 50862 | hosp.labevents |
| Hb (Hemoglobin) | 51222, 50811 | 51222, 50811 | hosp.labevents |
| WBC | 51301, 51300 | 51301, 51300 | hosp.labevents |
| Platelets | 51265 | 51265 | hosp.labevents |
| PTT | 51275 | 51275 | hosp.labevents |
| PT | 51274 | 51274 | hosp.labevents |
| INR | 51237 | 51237 | hosp.labevents |
| Arterial_pH | 50820 | 50820 | hosp.labevents |
| paO2 | 50821 | 50821 | hosp.labevents |
| paCO2 | 50818 | 50818 | hosp.labevents |
| Arterial_BE | 50802 | 50802 | hosp.labevents |
| HCO3 | 50882 | 50882 | hosp.labevents |
| Arterial_lactate | 50813 | 50813 | hosp.labevents |

### Fluid Balance (4 features)
| Feature | Source | MIMIC-IV Source | Notes |
|---------|--------|-----------------|-------|
| input_total | inputevents_mv/cv | icu.inputevents | Cumulative input |
| input_4hourly | inputevents_mv/cv | icu.inputevents | 4-hour window |
| output_total | outputevents | icu.outputevents | Cumulative output |
| output_4hourly | outputevents | icu.outputevents | 4-hour window |

### Treatment Variables (2 features)
| Feature | Source | MIMIC-IV Source | Notes |
|---------|--------|-----------------|-------|
| max_dose_vaso | inputevents (vasopressors) | icu.inputevents | Norepinephrine equivalent |
| mechvent | chartevents | icu.chartevents | Binary 0/1 |

### Derived/Computed Features (5 features)
| Feature | Calculation | Data Required |
|---------|-------------|---------------|
| SOFA | Sepsis-related Organ Failure | Multiple vitals + labs |
| SIRS | Systemic Inflammatory Response | HR, Temp, RR, WBC |
| Shock_Index | HR / SysBP | HR, SysBP |
| PaO2_FiO2 | paO2 / FiO2 | paO2, FiO2 |
| cumulated_balance | input_total - output_total | Input/Output |

### Weight (1 feature)
| Feature | MIMIC-III Source | MIMIC-IV Source | Notes |
|---------|------------------|-----------------|-------|
| Weight_kg | chartevents (itemid 763, 224639) | icu.chartevents OR hosp.omr | omr table has baseline weights |

## Critical MIMIC-IV Changes to Handle

### 1. InputEvents Table Unification
**MIMIC-III**: Separate `inputevents_mv` and `inputevents_cv`
**MIMIC-IV**: Single `inputevents` table with all data

**Impact**: Simplifies queries but need to check for:
- Rate column handling
- Status description changes
- ItemID consistency

### 2. New OMR Table
- Contains baseline height, weight, BMI
- Can provide pre-admission values
- May improve feature quality

### 3. Ingredient Events
- New table with detailed IV composition
- Useful for calculating fluid tonicity
- Can improve fluid balance calculations

### 4. Mortality Data Enhancement
- Out-of-hospital mortality now available (up to 1 year)
- Improves reward function accuracy

### 5. ItemID Changes
According to v3.1 changelog:
- labevents itemid fixed to match v2.2
- 43 rare lab itemid were changed in v2.0 (see changelog)
- Most common labs unchanged

## Extraction Strategy for Python Pipeline

### Phase 1: Core Demographics
```sql
SELECT
    p.subject_id,
    p.gender,
    p.anchor_age,
    p.dod,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.hospital_expire_flag,
    i.stay_id,
    i.intime,
    i.outtime,
    i.los
FROM mimic_hosp.patients p
INNER JOIN mimic_hosp.admissions a ON p.subject_id = a.subject_id
INNER JOIN mimic_icu.icustays i ON a.hadm_id = i.hadm_id
WHERE i.los >= 0.5  -- At least 12 hours in ICU
```

### Phase 2: Sepsis-3 Cohort
Need to implement:
1. **Suspected Infection**: Antibiotic administration + culture/microbiology
2. **SOFA Score ≥ 2**: Calculate from vitals and labs

### Phase 3: Time-Series Data (4-hour bins)
- Aggregate chartevents into 4-hour windows
- Aggregate labevents (carry forward last value)
- Calculate fluid input/output per 4-hour window
- Calculate max vasopressor dose per 4-hour window

### Phase 4: Feature Engineering
Create 148-dimensional feature vector:
- 48 state features (from above)
- 10 action features (one-hot encoded: 5 IV bins + 5 vaso bins)
- 90 interaction features (45 clinical × 2 treatment actions)

## Next Steps

1. ✅ Create SQL extraction queries for MIMIC-IV
2. ✅ Implement Sepsis-3 cohort definition
3. ✅ Build 4-hour aggregation logic
4. ✅ Implement feature normalization (z-score)
5. ✅ Create action discretization (5×5 bins)
6. ✅ Calculate reward function (SOFA changes + mortality)
7. ✅ Implement 70/15/15 split
