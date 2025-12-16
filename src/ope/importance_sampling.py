"""
Importance Sampling for Off-Policy Evaluation

Implements vanilla and weighted importance sampling estimators.

Author: AI Clinician Project
Date: 2024-11-16
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class ImportanceSampling:
    """
    Importance Sampling estimators (IS, WIS, WPDIS).
    """

    def __init__(self, gamma: float = 0.99):
        """
        Initialize ImportanceSampling.

        Args:
            gamma: Discount factor
        """
        self.gamma = gamma

    def weighted_per_decision_is(
        self,
        returns: np.ndarray,
        importance_ratios: np.ndarray
    ) -> float:
        """
        Weighted Per-Decision Importance Sampling (WPDIS).

        Args:
            returns: Observed returns (n_trajectories,)
            importance_ratios: Cumulative importance ratios (n_trajectories,)

        Returns:
            WPDIS estimate
        """
        weighted_returns = returns * importance_ratios
        weight_sum = np.sum(importance_ratios)

        if weight_sum > 0:
            wpdis_estimate = np.sum(weighted_returns) / weight_sum
        else:
            wpdis_estimate = 0.0

        logger.info(f"WPDIS estimate: {wpdis_estimate:.3f}")

        return wpdis_estimate
