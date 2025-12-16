"""
Off-Policy Evaluation (OPE) Module

Implements methods for evaluating RL policies from offline data:
- Weighted Doubly Robust (WDR)
- Weighted Per-Decision Importance Sampling (WPDIS)

Based on:
- Thomas & Brunskill (2016) "Data-Efficient Off-Policy Policy Evaluation"
- Komorowski et al. (2018) "The AI Clinician"

"""

from .wdr import WeightedDoublyRobust
from .importance_sampling import ImportanceSampling

__all__ = [
    'WeightedDoublyRobust',
    'ImportanceSampling',
]
