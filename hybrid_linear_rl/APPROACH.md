# Hybrid Linear RL - Option 5 Implementation

## Overview
Production implementation combining **Linear Q-learning** (proposal baseline) and **Gradient Boosting Q-learning** to compare simple vs complex function approximators.

## Key Design Decisions

### 1. Action Space: 3x3 (9 actions)
- **IV bins**: None (0), Low (1-2), High (3-4)
- **Vaso bins**: None (0), Low (1-2), High (3-4)
- Fixed mapping from 5-bin to 3-bin space

### 2. Linear Q-learning
**Architecture**:
- Feature vector: `[states(41), actions(9), interactions(41×9=369)]` = 419 features
- Linear approximation: `Q(s,a) = w^T · f(s,a)`
- L2 regularization (Ridge): `λ = 0.01`
- SGD update with weight decay

**Advantages**:
- Exactly matches proposal specification
- Highly interpretable (can inspect weights)
- Fast training (~500 epochs)
- No risk of overfitting with proper regularization

**Hyperparameters**:
- Learning rate: 0.001
- L2 lambda: 0.01
- Batch size: 1024
- Early stopping: patience 30

### 3. Gradient Boosting Q-learning
**Architecture**:
- Separate XGBoost model per action (9 models)
- Input: scaled states (41 features)
- Iterative Q-value updates (3 iterations)

**Advantages**:
- Captures non-linear patterns
- No manual feature engineering needed
- Robust to noisy data
- Good for offline RL

**Hyperparameters**:
- N estimators: 100
- Max depth: 6
- Learning rate: 0.1
- Subsample: 0.8

## Why This Approach Works

### Problem with Neural Network (Previous):
- 98.2% agreement → just mimicking clinicians
- NaN clinical sensitivity → learned constant policy
- ESS 0.0002 → extreme importance weight variance
- Q-values 149 vs 7.5 → unrealistic estimates

### Linear Model Advantages:
1. **Simpler hypothesis space** → less prone to collapse
2. **Explicit features** → can validate BP/lactate interactions
3. **Stable gradients** → no vanishing/exploding
4. **Interpretable** → can explain to clinicians

### GBM Advantages:
1. **Non-linear flexibility** → captures complex interactions
2. **Tree-based** → handles mixed feature scales well
3. **Proven for offline RL** → used in industry (Microsoft, Google)
4. **Fast inference** → suitable for real-time deployment

## Expected Outcomes

### Linear Q-learning:
- **Clinical tests**: 2-3/3 PASS (BP, lactate should respond correctly)
- **Agreement**: 30-60% (not too conservative)
- **WDR value**: 5-15 (reasonable, close to clinician 7.56)
- **Action diversity**: 7-9/9 actions used

### Gradient Boosting:
- **Clinical tests**: 2-3/3 PASS
- **Agreement**: 20-50% (more exploratory)
- **WDR value**: 8-20 (potentially better than linear)
- **Action diversity**: 8-9/9 actions used

### Comparison:
If **Linear > GBM**: Problem is fundamentally linear (good for interpretability)
If **GBM > Linear**: Non-linearities important (use GBM for deployment)

## Files Structure
```
hybrid_linear_rl/
├── config.py              # Configurations for both models
├── data_loader.py         # Action simplification + data loading
├── behavior_policy.py     # Logistic regression π_b
├── linear_q_learning.py   # Linear Q with L2 regularization
├── gbm_q_learning.py      # XGBoost Q-learning
├── evaluator.py           # WDR-OPE + Clinical validation
├── main.py                # Train both, compare results
└── run.sh                 # SLURM submission
```

## Runtime Estimate
- Linear training: ~10-30 minutes
- GBM training: ~30-60 minutes
- Evaluation: ~5-10 minutes
- **Total: ~1-2 hours** (much faster than neural network!)

## Next Steps
1. Submit job: `sbatch run.sh`
2. Compare linear vs GBM performance
3. If linear wins → use for interpretability analysis
4. If GBM wins → investigate which features drive non-linearity
5. Prepare final report with clinical validation
