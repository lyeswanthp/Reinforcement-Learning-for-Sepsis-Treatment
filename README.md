# Offline Reinforcement Learning for Sepsis Treatment

A comparative study of Q-Learning approaches for optimal treatment policy learning from MIMIC-IV intensive care data.

## Overview

This project implements and compares multiple offline RL algorithms to learn sepsis treatment policies for vasopressor and IV fluid administration. The work addresses key challenges in offline medical RL: behavioral constraints, Q-value overestimation, and clinical interpretability.

**Key Results:**
- Developed 5+ Q-learning variants with different function approximators
- Implemented Weighted Doubly Robust (WDR) off-policy evaluation
- Clinical validation via sensitivity analysis (BP response, lactate dynamics)
- Comprehensive comparison against clinician baseline policy

## Dataset

**MIMIC-IV v3.1** (Medical Information Mart for Intensive Care)
- **Source**: ICU patient records from Beth Israel Deaconess Medical Center
- **Cohort**: Septic patients (Sepsis-3 definition)
- **Timeframe**: 4-hour decision intervals
- **Features**: 48 clinical variables (vitals, labs, fluid balance, scores)
- **Actions**: 25 combinations (5 IV fluid bins × 5 vasopressor bins)
- **Split**: 70% train / 15% validation / 15% test

## Project Structure

```
.
├── configs/
│   └── config.yaml                    # Global configuration
├── src/
│   ├── preprocessing/                 # MIMIC-IV data pipeline
│   │   ├── cohort_selection.py        # Sepsis-3 cohort extraction
│   │   ├── data_cleaning.py           # Missing data, outliers
│   │   ├── feature_extraction_icu.py  # ICU features
│   │   ├── feature_extraction_hosp.py # Hospital features
│   │   └── preprocessing_pipeline.py  # End-to-end pipeline
│   ├── mdp/                           # MDP construction
│   │   ├── action_extraction.py       # Action discretization
│   │   ├── reward_computation.py      # SOFA-based rewards
│   │   └── trajectory_builder.py      # Episode construction
│   ├── rl/                            # RL algorithms
│   │   ├── q_learning.py              # Linear Q-learning
│   │   └── policy.py                  # Policy extraction
│   └── ope/                           # Off-policy evaluation
│       ├── wdr.py                     # Weighted Doubly Robust
│       └── importance_sampling.py     # IS estimators
├── final_refined/                     # Best approach (GBM + IQL)
├── iql_multitask/                     # IQL with multi-task learning
├── hybrid_linear_rl/                  # Linear vs GBM comparison
├── gbm_entropy/                       # GBM with entropy regularization
├── simplified_action_rl/              # Neural network (9 actions)
├── run_qlearning_final.py             # Linear Q-learning (proposal baseline)
├── run_double_dqn.py                  # Double DQN
├── run_dueling_dqn.py                 # Dueling DQN
└── run_preprocessing.py               # Data preprocessing entry point
```

## Algorithms Implemented

### 1. Linear Q-Learning (Baseline)
- **Method**: Linear function approximation with Fitted Q-Iteration (FQI)
- **Features**: 148 (states + actions + interactions)
- **Regularization**: L2 (Ridge regression)
- **Advantage**: Interpretable, stable, matches proposal specification
- **File**: `run_qlearning_final.py`

### 2. Gradient Boosting Q-Learning
- **Method**: XGBoost ensemble per action
- **Features**: 41 state features
- **Updates**: 3 iterative Q-value updates
- **Advantage**: Non-linear, robust to noise, proven for offline RL
- **Directory**: `gbm_entropy/`, `hybrid_linear_rl/`

### 3. Deep Q-Networks
- **Variants**: Double DQN, Dueling DQN
- **Architecture**: 3-layer MLP [256, 256, 128]
- **Regularization**: Conservative Q-Learning (CQL), gradient clipping
- **Files**: `run_double_dqn.py`, `run_dueling_dqn.py`

### 4. Implicit Q-Learning (IQL)
- **Method**: Expectile regression (no max operator)
- **Innovation**: Multi-task learning (Q, V, BP prediction, lactate prediction)
- **Advantage**: No Q-value overestimation, clinical sensitivity
- **Directory**: `iql_multitask/`

### 5. Ensemble Approach (Final)
- **Components**: GBM baseline + IQL refinement
- **Strategy**: Use GBM stability + IQL multi-task learning
- **Directory**: `final_refined/`

## Key Features

### Behavioral Constraints
- **BCQ-style filtering**: Only allow actions with π_b(a|s) > 0.05
- **Softened behavior policy**: 99%/1% ε-greedy smoothing
- **State-dependent π_b**: Logistic regression per-state action probabilities

### Off-Policy Evaluation
- **Primary**: Weighted Doubly Robust (WDR)
- **Baseline**: Weighted Per-Decision Importance Sampling (WPDIS)
- **Confidence Intervals**: 1000-sample bootstrap
- **Metrics**: ESS (Effective Sample Size), policy value, agreement rate

### Clinical Validation
1. **Blood Pressure Test**: π(hypotensive patient) → high vasopressors
2. **Lactate Test**: π(high lactate patient) → high IV fluids
3. **Action Diversity**: Policy uses ≥7 distinct actions

### Reward Function
- **Terminal**: +15 (survival), -15 (death)
- **Intermediate**: +0.1 per SOFA decrease, -0.25 per SOFA increase
- **Discount**: γ = 0.99

## Installation

```bash
# Clone repository
git clone https://github.com/lyeswanthp/Reinforcement-Learning-for-Sepsis-Treatment.git
cd Reinforcement-Learning-for-Sepsis-Treatment

# Install dependencies (Python 3.8+)
pip install numpy pandas scikit-learn xgboost torch pyyaml tqdm
```

## Usage

### 1. Data Preprocessing

```bash
# Process MIMIC-IV raw data
python run_preprocessing.py --config configs/config.yaml

# Output: data/processed/{train,val,test}_trajectories.csv
```

**Required MIMIC-IV files** (place in `data/raw/mimic-iv-3.1/`):
- ICU: `icustays.csv`, `chartevents.csv`, `inputevents.csv`, `outputevents.csv`
- Hospital: `admissions.csv`, `patients.csv`, `labevents.csv`, `diagnoses_icd.csv`

### 2. Train RL Models

```bash
# Linear Q-learning (baseline)
python run_qlearning_final.py

# Gradient Boosting Q-learning
cd hybrid_linear_rl && python main.py

# IQL with multi-task learning
cd iql_multitask && python main.py

# Deep Q-Networks
python run_double_dqn.py  # or run_dueling_dqn.py

# Final ensemble approach
cd final_refined && python main.py
```

### 3. SLURM Submission (HPC)

```bash
# Submit to SLURM scheduler
sbatch final_refined/run.sh

# Monitor job
squeue -u $USER
tail -f logs/slurm-*.out
```

### 4. Evaluation

All training scripts automatically run WDR-OPE and clinical validation. Results saved to:
- `outputs/models_*/results.json` - WDR value, ESS, confidence intervals
- `outputs/models_*/clinical_validation.txt` - BP/lactate/diversity tests
- `outputs/models_*/policy.pkl` - Trained policy

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
mdp:
  gamma: 0.99
  time_window_hours: 4

action_space:
  iv_fluid_bins: 5
  vasopressor_bins: 5

reward:
  terminal_survival: 15
  terminal_death: -15

q_learning:
  learning_rate_range: [1e-5, 1e-1]
  l2_regularization_range: [1e-6, 1e-2]

ope:
  methods: ["wpdis", "wdr"]
  bootstrap:
    n_iterations: 1000
    confidence_level: 0.95
```

## Results

### WDR Policy Value Comparison

| Algorithm | WDR Value | 95% CI | Clinical Tests | Action Diversity |
|-----------|-----------|--------|----------------|------------------|
| Clinician Baseline | 7.56 | [7.2, 7.9] | 3/3 | 20/25 |
| Linear Q-Learning | 5.64 | [5.1, 6.2] | 0/3 | 1/9 |
| GBM Q-Learning | 6.70 | [6.3, 7.1] | 0/3 | 2/9 |
| Double DQN | 149.2 | - | 0/3 | 6/20 |
| IQL Multi-Task | *Running* | - | - | - |

**Note**: Q-value overestimation observed in deep RL methods. IQL and ensemble approaches designed to address this.

### Key Findings
1. **Overestimation Problem**: Standard DQN variants produce unrealistic Q-values (149 vs 7.5)
2. **Action Collapse**: Most methods converge to deterministic policies (1-2 actions)
3. **Clinical Insensitivity**: Models fail BP/lactate response tests without multi-task learning
4. **Importance Weight Variance**: ESS < 0.001 indicates poor behavior policy estimation

## Reproducibility

```bash
# Set random seeds (config.yaml)
random_seed: 42

# Use same MIMIC-IV version
data_source:
  version: "mimic-iv-3.1"

# Fixed data split
data_split:
  random_seed: 42
  stratify_by_mortality: true
```

## Citation

```bibtex
@misc{panchumarthi2024offline,
  title={Offline Reinforcement Learning for Sepsis Treatment: A Comparative Study of Q-Learning Approaches},
  author={Panchumarthi, Lyeswanth},
  year={2024},
  howpublished={GitHub repository},
  url={https://github.com/lyeswanthp/Reinforcement-Learning-for-Sepsis-Treatment}
}
```

## References

1. Komorowski et al. (2018) - The AI Clinician for intensive care treatment
2. Kostrikov et al. (2021) - Offline Reinforcement Learning with Implicit Q-Learning
3. Kumar et al. (2020) - Conservative Q-Learning for Offline RL
4. Johnson et al. (2023) - MIMIC-IV, a freely accessible electronic health record dataset

## License

This project uses the MIMIC-IV dataset, which requires:
- Completion of CITI "Data or Specimens Only Research" course
- PhysioNet credentialing
- Signed data use agreement

Code is released under MIT License. See [LICENSE](LICENSE) for details.

## Contact

**Lyeswanth Panchumarthi**
- GitHub: [@lyeswanthp](https://github.com/lyeswanthp)
- Repository: [Reinforcement-Learning-for-Sepsis-Treatment](https://github.com/lyeswanthp/Reinforcement-Learning-for-Sepsis-Treatment)

## Acknowledgments

- MIMIC-IV dataset provided by MIT Laboratory for Computational Physiology
- Original AI Clinician implementation by Komorowski et al.
- IQL implementation inspired by Kostrikov et al.
