# AI Clinician: Linear Q-Learning with WDR-OPE for Sepsis Treatment

A robust and interpretable framework for sepsis treatment policy optimization using reinforcement learning on MIMIC-IV v3.1 data.

## 📋 Project Overview

This project implements the methodology described in the proposal: **"A Robust and Interpretable Framework for Sepsis Treatment Policy Optimization via Approximate Reinforcement Learning and Model-Assisted Off-Policy Evaluation"**

### Key Features

- **Dataset**: MIM IC-IV v3.1 (October 2024 release)
- **Baseline**: AI Clinician by Komorowski et al. (Nature Medicine 2018)
- **Algorithm**: Linear Approximate Q-Learning with L2 Regularization
- **Evaluation**: Weighted Doubly Robust (WDR) Off-Policy Evaluation with Bootstrap CIs
- **State Space**: 148-dimensional continuous features (48 state + 10 action + 90 interactions)
- **Action Space**: 25 discrete actions (5 IV fluid bins × 5 vasopressor bins)

## 🗂️ Project Structure

```
RL/
├── baseline_ai_clinician/          # Original MATLAB baseline code (Komorowski et al.)
├── configs/
│   └── config.yaml                 # Main configuration file
├── data/
│   ├── raw/                        # Raw extracted data from MIMIC-IV
│   ├── processed/                  # Processed features and trajectories
│   └── splits/                     # Train/validation/test splits
├── src/
│   ├── data_extraction/            # MIMIC-IV SQL extraction
│   ├── preprocessing/              # Data cleaning and preprocessing
│   ├── feature_engineering/        # 148-dimensional feature construction
│   ├── rl_algorithms/              # Linear Q-learning implementation
│   ├── ope_methods/                # WDR-OPE and bootstrap
│   ├── models/                     # Dynamics and reward models
│   └── utils/                      # Utilities (config, database, logging)
├── notebooks/                      # Jupyter notebooks for analysis
├── results/                        # Model outputs and evaluations
├── logs/                           # Training and evaluation logs
├── MIMIC_III_to_IV_MAPPING.md     # Feature mapping documentation
├── MIMIC_IV_CHANGELOG.md          # MIMIC-IV version notes
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites

1. **MIMIC-IV v3.1 Access**
   - Complete CITI training
   - Get PhysioNet credentialed access
   - Download MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/

2. **PostgreSQL Database**
   - Install PostgreSQL 12+
   - Load MIMIC-IV v3.1 into database
   - Follow: https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres

3. **Python Environment**
   - Python 3.9+
   - Virtual environment (recommended)

### Installation

```bash
# Clone repository
cd RL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Update database configuration
# Edit configs/config.yaml with your database credentials
```

### Configuration

Edit `configs/config.yaml`:

```yaml
database:
  name: "your_mimic4_db"
  user: "your_username"
  password: "your_password"
  host: "localhost"
  port: 5432
```

## 📊 Data Pipeline Workflow

### 1. Cohort Selection (Sepsis-3 Criteria)
- Adult patients (age ≥ 18)
- ICU stay ≥ 12 hours
- Suspected infection (antibiotics + cultures within ±24h)
- SOFA score ≥ 2

### 2. Feature Extraction
Extracts 48 baseline features every 4 hours:

**Demographics (4)**
- Age, Gender, Weight, Re-admission

**Vital Signs (10)**
- HR, SysBP, MeanBP, DiaBP, RR, Temperature, SpO2, FiO2, GCS, Mechanical Ventilation

**Laboratory Values (22)**
- Electrolytes, Renal function, Liver function, Hematology, Coagulation, Blood gas

**Fluid Balance (4)**
- Input/Output (total and 4-hourly)

**Derived Scores (5)**
- SOFA, SIRS, Shock Index, PaO2/FiO2, Cumulative Balance

**Treatment (2)**
- Max vasopressor dose (norepinephrine equivalent)
- Mechanical ventilation status

### 3. Action Discretization
- **IV Fluids**: 5 bins based on 4-hour volume (includes 0)
- **Vasopressors**: 5 bins based on max dose (includes 0)
- **Total Actions**: 5 × 5 = 25 discrete actions

### 4. Feature Engineering (148 dimensions)
- **48** state features (z-scored)
- **10** action features (one-hot encoded)
- **90** interaction features (45 clinical states × 2 action components)

### 5. Reward Function
- **Terminal**: +15 for survival, -15 for death
- **Intermediate**: Based on SOFA score changes
  - -0.25 per point increase (worsening)
  - +0.1 per point decrease (improvement)
  - 0 for no change

### 6. Data Splitting
- **Training**: 70% (hyperparameter tuning, model learning)
- **Validation**: 15% (early stopping, model selection)
- **Test**: 15% (final OPE, locked until end)

## 🤖 Reinforcement Learning Pipeline

### Linear Approximate Q-Learning

```
Q(s, a) = w^T f(s, a)
```

Where:
- `w`: Weight vector (148 dimensions)
- `f(s, a)`: Feature vector (state + action + interactions)

**SGD Update with L2 Regularization:**

```
w_i ← w_i(1 - α·λ) + α · (target - Q(s,a)) · f_i(s,a)
target = r + γ · max_a' Q(s', a')
```

### Hyperparameter Tuning
- **Method**: Random Search (50-100 trials)
- **Search Space**:
  - Learning rate (α): log-uniform[1e-5, 1e-1]
  - Discount factor (γ): {0.90, 0.95, 0.99}
  - L2 regularization (λ): log-uniform[1e-6, 1e-2]
- **Optimization Metric**: WDR-OPE value on validation set
- **Early Stopping**: 20 epochs without improvement

## 📈 Off-Policy Evaluation (OPE)

### Primary Method: Weighted Doubly Robust (WDR)

The WDR estimator combines:
1. **Model-based estimate**: Using learned dynamics T(s,a,s') and reward R(s,a)
2. **IS-based correction**: Using importance weights to correct model bias

**Doubly Robust Property**: Unbiased if EITHER the model OR importance weights are correct.

### Confidence Intervals: Non-Parametric Bootstrap
- **Iterations**: 1,000 bootstrap samples
- **Sample Size**: Up to 25,000 trajectories (or 75% of data)
- **Method**: Resample entire patient trajectories with replacement
- **Output**: 95% CI from 2.5th and 97.5th percentiles

### Behavior Policy Softening
```
π_b'(a|s) = (1-ε)π_b(a|s) + ε(1/|A|)
ε = 0.01  # 99%/1% mix
```

## 🔍 Interpretability Analysis

### Policy Sensitivity Plots
- Define prototypical patient states
- Vary critical variables (e.g., Lactate: 1-10 mmol/L)
- Plot optimal action vs. variable value
- Present to clinicians for validation

Example: "What does the policy recommend as lactate increases?"

## 📚 Key Documentation

- **[MIMIC_III_to_IV_MAPPING.md](MIMIC_III_to_IV_MAPPING.md)**: Complete mapping from baseline (MIMIC-III) to our implementation (MIMIC-IV v3.1)
- **[MIMIC_IV_CHANGELOG.md](MIMIC_IV_CHANGELOG.md)**: MIMIC-IV version history and important changes
- **[baseline_ai_clinician/README.md](baseline_ai_clinician/README.md)**: Original baseline code documentation

## 🎯 Current Status

### ✅ Completed
1. Imported baseline AI Clinician code (Komorowski et al.)
2. Analyzed MIMIC-IV v3.1 structure and changelog
3. Created complete MIMIC-III → MIMIC-IV feature mapping
4. Set up project structure
5. Created configuration system
6. Built database connection utilities
7. Documented data pipeline requirements

### 🚧 In Progress
1. SQL extraction queries for MIMIC-IV v3.1
2. Sepsis-3 cohort definition
3. Data preprocessing pipeline

### 📋 To Do
1. Feature engineering (148 dimensions)
2. Linear Q-learning implementation
3. Dynamics model for WDR-OPE
4. WDR estimator implementation
5. Bootstrap CI generation
6. Hyperparameter tuning framework
7. Policy interpretability analysis
8. Comparison with baseline

## 📖 References

### Main Paper (Baseline)
Komorowski, M., Celi, L.A., Badawi, O. et al. The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care. *Nat Med* 24, 1716–1720 (2018). https://doi.org/10.1038/s41591-018-0213-5

### MIMIC-IV Database
Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 3.1). PhysioNet. https://doi.org/10.13026/kpb9-mt28

### Sepsis-3 Definition
Singer, M., Deutschman, C.S., Seymour, C.W., et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA* 315(8):801-810 (2016). https://doi.org/10.1001/jama.2016.0287

### Off-Policy Evaluation
Thomas, P. S., & Brunskill, E. (2016). Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning. *ICML*.

## 👥 Contributors

Your Name - Project Lead

## 📄 License

This project follows the same license as MIMIC-IV (PhysioNet Credentialed Health Data License).

## 🙏 Acknowledgments

- Matthieu Komorowski et al. for the original AI Clinician
- MIT-LCP for MIMIC-IV database
- PhysioNet for data access
