"""
Off-policy evaluation and validation.
"""
import numpy as np
import pandas as pd
from scipy import stats
from agent import DoubleDQNAgent
from models import BehaviorPolicy
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
        agent: DoubleDQNAgent,
        behavior_policy: BehaviorPolicy
    ) -> Dict:
        trajectory_values = []
        trajectory_weights = []

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
            pi_e = 0.99 * pi_e + 0.01 / agent.n_actions

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
            trajectory_weights.append(cumulative_rho)

        trajectory_values = np.array(trajectory_values)
        trajectory_weights = np.array(trajectory_weights)

        ess = (np.sum(trajectory_weights) ** 2) / (np.sum(trajectory_weights ** 2) + 1e-10)
        ess_ratio = ess / len(trajectory_weights)

        return {
            'wdr_estimate': np.mean(trajectory_values),
            'wdr_std': np.std(trajectory_values),
            'wdr_se': np.std(trajectory_values) / np.sqrt(len(trajectory_values)),
            'ess_ratio': ess_ratio,
            'trajectory_values': trajectory_values
        }


class ClinicalValidator:
    """Clinical sensitivity analysis."""

    def __init__(self, agent: DoubleDQNAgent, state_cols: list, action_simplifier):
        self.agent = agent
        self.state_cols = state_cols
        self.action_simplifier = action_simplifier
        self.feature_idx = {col: i for i, col in enumerate(state_cols)}

    def validate(self) -> Dict:
        logger.info("="*60)
        logger.info("CLINICAL SENSITIVITY ANALYSIS")
        logger.info("="*60)

        results = {}
        baseline_state = np.zeros((1, len(self.state_cols)))

        if 'MeanBP' in self.feature_idx:
            results['bp_sensitivity'] = self._test_bp_response(baseline_state)

        if 'Arterial_lactate' in self.feature_idx:
            results['lactate_sensitivity'] = self._test_lactate_response(baseline_state)

        results['diversity'] = self._test_action_diversity()

        tests_passed = sum([
            results.get('bp_sensitivity', {}).get('passed', False),
            results.get('lactate_sensitivity', {}).get('passed', False),
            results.get('diversity', {}).get('passed', False)
        ])

        results['overall'] = {
            'passed': tests_passed,
            'total': 3,
            'score': tests_passed / 3
        }

        logger.info(f"Clinical validation: {tests_passed}/3 tests passed")
        return results

    def _test_bp_response(self, baseline: np.ndarray) -> Dict:
        bp_idx = self.feature_idx['MeanBP']
        bp_range = np.linspace(-2, 2, 9)
        vaso_recommendations = []

        for bp_val in bp_range:
            test_state = baseline.copy()
            test_state[0, bp_idx] = bp_val
            action = self.agent.select_actions(test_state)[0]
            _, vaso_bin = self.action_simplifier.get_action_components(action)
            vaso_recommendations.append(vaso_bin)

        corr = stats.spearmanr(bp_range, vaso_recommendations)
        passed = corr.correlation < -0.2 and not np.isnan(corr.correlation)

        logger.info(f"  BP sensitivity: r={corr.correlation:.3f}, {'PASS' if passed else 'FAIL'}")

        return {
            'correlation': corr.correlation,
            'p_value': corr.pvalue,
            'passed': passed
        }

    def _test_lactate_response(self, baseline: np.ndarray) -> Dict:
        lac_idx = self.feature_idx['Arterial_lactate']
        lac_range = np.linspace(-1, 3, 9)
        intensity_recommendations = []

        for lac_val in lac_range:
            test_state = baseline.copy()
            test_state[0, lac_idx] = lac_val
            action = self.agent.select_actions(test_state)[0]
            iv_bin, vaso_bin = self.action_simplifier.get_action_components(action)
            intensity_recommendations.append(iv_bin + vaso_bin)

        corr = stats.spearmanr(lac_range, intensity_recommendations)
        passed = corr.correlation > 0.2 and not np.isnan(corr.correlation)

        logger.info(f"  Lactate sensitivity: r={corr.correlation:.3f}, {'PASS' if passed else 'FAIL'}")

        return {
            'correlation': corr.correlation,
            'p_value': corr.pvalue,
            'passed': passed
        }

    def _test_action_diversity(self) -> Dict:
        random_states = np.random.randn(1000, len(self.state_cols))
        random_actions = self.agent.select_actions(random_states)

        n_unique = len(np.unique(random_actions))
        action_counts = np.bincount(random_actions, minlength=self.agent.n_actions)
        action_probs = action_counts / action_counts.sum()
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        max_entropy = np.log(self.agent.n_actions)

        passed = n_unique >= 5 and entropy > 1.0

        logger.info(f"  Action diversity: {n_unique} unique, H={entropy:.2f}, {'PASS' if passed else 'FAIL'}")

        return {
            'n_unique': n_unique,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'passed': passed
        }


class KomorowskiValidator:
    """Komorowski-style mortality validation."""

    def __init__(self, agent: DoubleDQNAgent, action_simplifier):
        self.agent = agent
        self.action_simplifier = action_simplifier

    def validate(self, df: pd.DataFrame, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> Dict:
        logger.info("="*60)
        logger.info("KOMOROWSKI VALIDATION")
        logger.info("="*60)

        ai_actions = self.agent.select_actions(states)

        agreement_results = self._validate_agreement(df, actions, ai_actions, rewards)
        distance_results = self._validate_distance(df, actions, ai_actions, rewards)

        overall_passed = (
            agreement_results['passed'] or distance_results['passed']
        )

        logger.info(f"  Agreement test: {'PASS' if agreement_results['passed'] else 'FAIL'}")
        logger.info(f"  Distance test: {'PASS' if distance_results['passed'] else 'FAIL'}")

        return {
            'agreement': agreement_results,
            'distance': distance_results,
            'overall_passed': overall_passed
        }

    def _validate_agreement(self, df: pd.DataFrame, clin_actions: np.ndarray, ai_actions: np.ndarray, rewards: np.ndarray) -> Dict:
        trajectory_data = []

        for stay_id in df['stay_id'].unique():
            mask = (df['stay_id'] == stay_id).values
            traj_rewards = rewards[mask]
            traj_clin = clin_actions[mask]
            traj_ai = ai_actions[mask]

            terminal_reward = traj_rewards[-1]
            died = terminal_reward < 0
            agreement = np.mean(traj_clin == traj_ai)

            trajectory_data.append({
                'died': died,
                'agreement': agreement
            })

        traj_df = pd.DataFrame(trajectory_data)
        corr = stats.spearmanr(traj_df['agreement'], traj_df['died'])
        passed = corr.correlation < 0 and corr.pvalue < 0.05

        return {
            'correlation': corr.correlation,
            'p_value': corr.pvalue,
            'passed': passed
        }

    def _validate_distance(self, df: pd.DataFrame, clin_actions: np.ndarray, ai_actions: np.ndarray, rewards: np.ndarray) -> Dict:
        distances = []

        for i in range(len(clin_actions)):
            clin_iv, clin_vaso = self.action_simplifier.get_action_components(clin_actions[i])
            ai_iv, ai_vaso = self.action_simplifier.get_action_components(ai_actions[i])
            dist = np.sqrt((clin_iv - ai_iv)**2 + (clin_vaso - ai_vaso)**2)
            distances.append(dist)

        distances = np.array(distances)

        trajectory_data = []
        for stay_id in df['stay_id'].unique():
            mask = (df['stay_id'] == stay_id).values
            traj_rewards = rewards[mask]
            traj_dist = distances[mask]

            terminal_reward = traj_rewards[-1]
            died = terminal_reward < 0
            mean_distance = np.mean(traj_dist)

            trajectory_data.append({
                'died': died,
                'mean_distance': mean_distance
            })

        traj_df = pd.DataFrame(trajectory_data)
        corr = stats.spearmanr(traj_df['mean_distance'], traj_df['died'])
        passed = corr.correlation > 0 and corr.pvalue < 0.05

        return {
            'correlation': corr.correlation,
            'p_value': corr.pvalue,
            'passed': passed
        }
