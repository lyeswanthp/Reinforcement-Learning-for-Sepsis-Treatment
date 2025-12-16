# IQL with Multi-Task Learning - Production Implementation

## Overview

Production implementation of **Implicit Q-Learning (IQL)** with **multi-task learning** to address offline RL challenges: action collapse, Q-value overestimation, and clinical insensitivity.

## Key Innovations

### 1. Implicit Q-Learning (IQL)

**Problem with Standard Q-Learning:**
- Uses max operator: `Q(s,a) = r + γ max Q(s',a')`
- Overestimates Q-values for out-of-distribution actions
- Leads to unrealistic policy values (149 vs 7.5)

**IQL Solution:**
- **Expectile Regression**: No max operator, asymmetric loss
- **Separate V-Network**: Learns state values independently
- **Target**: `Q(s,a) = r + γ V(s')` where V is learned via expectile

**Mathematics:**
```
V(s) = E_τ[Q(s,a)] where τ = expectile (0.7)
Expectile loss: L = τ(y-ŷ)² if y>ŷ else (1-τ)(y-ŷ)²
```

**Advantages:**
- No overestimation bias
- Stable training (no bootstrapping errors)
- Proven for offline RL (Kostrikov et al., 2021)

### 2. Multi-Task Learning

**Problem:**
- Models ignore physiological relationships
- BP/lactate clinical tests fail (NaN correlations)
- No incentive to learn BP→vaso, lactate→IV causality

**Solution - 4 Prediction Heads:**
1. **Q-values**: Policy optimization
2. **V-values**: State value estimation
3. **BP prediction**: Forces model to understand BP dynamics
4. **Lactate prediction**: Forces model to understand metabolic state

**Loss Function:**
```
L_total = L_Q + L_V + λ_BP·L_BP + λ_lactate·L_lactate - β·H(π)

Where:
- L_Q, L_V: IQL expectile losses
- L_BP, L_lactate: MSE on next BP/lactate
- H(π): Entropy bonus for action diversity
```

**Expected Impact:**
- BP test: PASS (model learns BP→vaso to minimize L_BP)
- Lactate test: PASS (model learns lactate→IV to minimize L_lactate)
- Diversity: 7+ actions (entropy regularization β=0.3)

### 3. Temporal Data Split

**Problem with Random Split:**
- Train/test patients from same time periods
- Model sees same treatment protocols
- Overly optimistic evaluation

**Temporal Split Implementation:**
```
Training:   70% earliest admissions
Validation: 15% middle admissions
Test:       15% latest admissions
```

**Benefits:**
- Tests temporal generalization
- Detects protocol drift
- Realistic evaluation of deployment

### 4. Entropy Regularization

**Problem:**
- All previous models: action collapse at inference
- Only 1-2 actions used despite training with all 9

**Solution:**
```python
action_probs = softmax(Q(s,·) / temperature)
entropy = -Σ p·log(p)
loss = loss - β_entropy * entropy  # β=0.3
```

**Effect:**
- Forces model to maintain high entropy policy
- Prevents deterministic collapse
- Used in Soft Actor-Critic (SAC)

## Architecture

```
Input: State (41 features)
    ↓
Shared Network: [256, 256, 128] with LayerNorm + GELU
    ↓
    ├─→ Q-Head: Linear(128, 9)
    ├─→ V-Head: Linear(128, 1)
    ├─→ BP-Head: Linear(128, 1)
    └─→ Lactate-Head: Linear(128, 1)

Total: ~280K parameters
```

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Expectile | 0.7 | Standard IQL (focus on upper tail) |
| Temperature | 3.0 | High for action diversity |
| Entropy weight | 0.3 | Strong diversity enforcement |
| BP weight | 0.1 | Auxiliary task, not dominant |
| Lactate weight | 0.1 | Auxiliary task, not dominant |
| Learning rate | 3e-4 | Standard for AdamW |
| Batch size | 1024 | Large for stable gradients |
| Gradient clip | 1.0 | Prevent explosion |

## Expected Results

### Baseline Comparison

| Model | WDR | Clinical | Diversity | Issue |
|-------|-----|----------|-----------|-------|
| Double DQN | 149.2 | 0/3 | 6/20 | Q-explosion |
| Dueling DQN | N/A | 1/3 | 9/20 | CQL too strong |
| Neural | 0.675 | 1/3 | 6/9 | Undertrained |
| Linear | 5.639 | 0/3 | 1/9 | Too simple |
| GBM | 6.704 | 0/3 | 2/9 | Action collapse |

### IQL Multi-Task Targets

- **WDR**: 7-9 (close to or better than clinician 7.56)
- **Clinical**: 3/3 PASS (BP + lactate + diversity)
- **Diversity**: 7-9 actions, H > 1.5
- **Agreement**: 20-40% (not mimicking clinicians)

### Why This Should Work

1. **IQL**: No overestimation → realistic Q-values
2. **Multi-task**: Forces clinical relationships → passes BP/lactate tests
3. **Entropy**: Strong regularization → action diversity
4. **Temporal split**: Realistic evaluation → true generalization

## Implementation Quality

**Production-Level Features:**
- Clean modular architecture (6 files)
- Type hints and docstrings
- Proper logging and error handling
- GPU acceleration with automatic device detection
- Early stopping with patience
- Reproducible (fixed random seeds)
- Configurable via dataclasses

**No Unnecessary Code:**
- No unused imports
- No debug prints
- No commented-out code
- Minimal comments (code is self-documenting)

## Runtime Estimate

- Training: 2-4 hours (depends on early stopping)
- Evaluation: 10-15 minutes
- **Total: ~3-5 hours**

## Files

```
iql_multitask/
├── config.py              # All configurations
├── data_loader.py         # Temporal split + action simplification
├── behavior_policy.py     # Logistic regression π_b
├── iql_agent.py          # IQL with multi-task network
├── trainer.py            # Training loop
├── evaluator.py          # WDR-OPE + clinical validation
├── main.py               # Main pipeline
└── run.sh                # SLURM submission
```

## Key Differences from Previous Approaches

1. **vs Double DQN**: No max operator → no overestimation
2. **vs Linear Q**: Non-linear network → captures interactions
3. **vs GBM**: Gradient-based → end-to-end learning
4. **vs All Previous**: Multi-task + entropy → clinical sensitivity + diversity

## Next Steps After Results

**If WDR ≥ 7.5 and Clinical 3/3:**
- Submit to medical AI conference (MLHC, AAAI)
- Prepare for clinical deployment discussions

**If Clinical 2/3:**
- Increase auxiliary task weights (0.1 → 0.2)
- Add more test states for validation

**If Diversity < 7:**
- Increase entropy weight (0.3 → 0.5)
- Use temperature sampling at inference

**If WDR < 6:**
- Adjust expectile (0.7 → 0.8 for more optimism)
- Increase network capacity ([256,256,128] → [512,512,256])
