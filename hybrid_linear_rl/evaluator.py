"""
Evaluation: WDR-OPE and clinical validation.
"""
import numpy as np
import pandas as pd
from scipy import stats
from behavior_policy import BehaviorPolicy
from config import Config
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class WDREstimator:
    """Weighted Doubly Robust estimator."""

    def __init__(self, gamma: float, max_weight: float):
        self.gamma = gamma
        self.max_weight = max_weight

    def estimate(
        self,
        df: pd.DataFrame,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        agent,
        behavior_policy: BehaviorPolicy
    ) -> Dict:
        trajectory_values = []

        for stay_id in df['stay_id'].unique():
            mask = (df['stay_id'] == stay_id).values

            traj_states = states[mask]
            traj_actions = actions[mask]
            traj_rewards = rewards[mask]
            traj_dones = dones[mask]

            q_values = agent.get_q_values(traj_states)
            pi_b = behavior_policy.get_action_probs(traj_states, traj_actions)
            pi_b = np.clip(pi_b, 1e-6, 1.0)

            target_actions = agent.select_actions(traj_states)
            pi_e = (target_actions == traj_actions).astype(float)
            pi_e = 0.99 * pi_e + 0.01 / len(set(actions))

            rho = pi_e / pi_b
            rho = np.clip(rho, 0, self.max_weight)

            T = len(traj_rewards)
            v_wdr = 0.0
            cumulative_rho = 1.0

            for t in range(T):
                q_t = q_values[t, int(traj_actions[t])]
                cumulative_rho = np.clip(cumulative_rho * rho[t], 0, self.max_weight)

                if t == 0:
                    v_wdr = q_values[t].max()

                next_q = q_values[min(t+1, T-1)].max() if not traj_dones[t] else 0
                advantage = traj_rewards[t] + self.gamma * next_q - q_t
                v_wdr += (self.gamma ** t) * cumulative_rho * advantage

            trajectory_values.append(v_wdr)

        trajectory_values = np.array(trajectory_values)

        return {
            'wdr_estimate': np.mean(trajectory_values),
            'wdr_std': np.std(trajectory_values),
            'wdr_se': np.std(trajectory_values) / np.sqrt(len(trajectory_values)),
            'trajectory_values': trajectory_values
        }


class ClinicalValidator:
    """Clinical sensitivity tests."""

    def __init__(self, agent, state_cols: list, action_simplifier):
        self.agent = agent
        self.state_cols = state_cols
        self.action_simplifier = action_simplifier
        self.feature_idx = {col: i for i, col in enumerate(state_cols)}

    def validate(self) -> Dict:
        logger.info("="*60)
        logger.info("CLINICAL SENSITIVITY ANALYSIS")
        logger.info("="*60)

        results = {}
        baseline = np.zeros((1, len(self.state_cols)))

        if 'MeanBP' in self.feature_idx:
            results['bp'] = self._test_bp(baseline)

        if 'Arterial_lactate' in self.feature_idx:
            results['lactate'] = self._test_lactate(baseline)

        results['diversity'] = self._test_diversity()

        passed = sum([
            results.get('bp', {}).get('passed', False),
            results.get('lactate', {}).get('passed', False),
            results.get('diversity', {}).get('passed', False)
        ])

        results['overall'] = {'passed': passed, 'total': 3}
        logger.info(f"Clinical validation: {passed}/3")
        return results

    def _test_bp(self, baseline: np.ndarray) -> Dict:
        idx = self.feature_idx['MeanBP']
        bp_range = np.linspace(-2, 2, 9)
        vaso_recs = []

        for bp in bp_range:
            state = baseline.copy()
            state[0, idx] = bp
            action = self.agent.select_actions(state)[0]
            _, vaso = self.action_simplifier.get_action_components(action)
            vaso_recs.append(vaso)

        corr = stats.spearmanr(bp_range, vaso_recs)
        passed = corr.correlation < -0.2 and not np.isnan(corr.correlation)

        logger.info(f"  BP: r={corr.correlation:.3f}, {'PASS' if passed else 'FAIL'}")
        return {'correlation': corr.correlation, 'passed': passed}

    def _test_lactate(self, baseline: np.ndarray) -> Dict:
        idx = self.feature_idx['Arterial_lactate']
        lac_range = np.linspace(-1, 3, 9)
        intensity_recs = []

        for lac in lac_range:
            state = baseline.copy()
            state[0, idx] = lac
            action = self.agent.select_actions(state)[0]
            iv, vaso = self.action_simplifier.get_action_components(action)
            intensity_recs.append(iv + vaso)

        corr = stats.spearmanr(lac_range, intensity_recs)
        passed = corr.correlation > 0.2 and not np.isnan(corr.correlation)

        logger.info(f"  Lactate: r={corr.correlation:.3f}, {'PASS' if passed else 'FAIL'}")
        return {'correlation': corr.correlation, 'passed': passed}

    def _test_diversity(self) -> Dict:
        random_states = np.random.randn(1000, len(self.state_cols))
        actions = self.agent.select_actions(random_states)

        n_unique = len(np.unique(actions))
        counts = np.bincount(actions, minlength=9)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        passed = n_unique >= 5 and entropy > 1.0

        logger.info(f"  Diversity: {n_unique} unique, H={entropy:.2f}, {'PASS' if passed else 'FAIL'}")
        return {'n_unique': n_unique, 'entropy': entropy, 'passed': passed}
