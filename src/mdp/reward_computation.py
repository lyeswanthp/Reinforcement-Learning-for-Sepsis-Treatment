"""
Reward Computation Module

Computes rewards for the RL agent based on SOFA score changes and mortality.

Reward Structure (from Komorowski et al.):
- Terminal reward (survival): +15
- Terminal reward (death): -15
- Intermediate reward (SOFA decrease): +0.1 per point
- Intermediate reward (SOFA increase): -0.25 per point
- Intermediate reward (no change): 0

Author: AI Clinician Project
Date: 2024-11-16
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RewardComputer:
    """
    Computes rewards for reinforcement learning based on SOFA scores and mortality.

    Implements the reward function from the AI Clinician paper.
    """

    def __init__(self, config: Dict):
        """
        Initialize RewardComputer.

        Args:
            config: Configuration dictionary containing reward settings
        """
        self.config = config
        reward_config = config.get('reward', {})

        # Reward parameters (from config)
        self.terminal_survival = reward_config.get('terminal_survival', 15)
        self.terminal_death = reward_config.get('terminal_death', -15)
        self.intermediate_sofa_decrease = reward_config.get('intermediate_sofa_decrease', 0.1)
        self.intermediate_sofa_increase = reward_config.get('intermediate_sofa_increase', -0.25)
        self.intermediate_no_change = reward_config.get('intermediate_no_change', 0)

        logger.info("RewardComputer initialized")
        logger.info(f"  Terminal survival reward: {self.terminal_survival}")
        logger.info(f"  Terminal death reward: {self.terminal_death}")
        logger.info(f"  SOFA decrease reward: {self.intermediate_sofa_decrease} per point")
        logger.info(f"  SOFA increase penalty: {self.intermediate_sofa_increase} per point")

    def compute_sofa_score(
        self,
        features: pd.DataFrame,
        required_components: bool = True
    ) -> pd.DataFrame:
        """
        Compute SOFA (Sequential Organ Failure Assessment) score.

        SOFA Components (0-4 points each, max 24 total):
        1. Respiration: PaO2/FiO2 ratio
        2. Coagulation: Platelets
        3. Liver: Bilirubin
        4. Cardiovascular: MAP and vasopressor use
        5. CNS: GCS
        6. Renal: Creatinine (and urine output)

        Args:
            features: DataFrame with clinical features
            required_components: If True, require all components (may result in missing SOFA)

        Returns:
            DataFrame with columns: stay_id, time_window, sofa_score, sofa_*
        """
        logger.info("Computing SOFA scores...")

        df = features.copy()

        # Initialize component scores
        df['sofa_respiration'] = 0
        df['sofa_coagulation'] = 0
        df['sofa_liver'] = 0
        df['sofa_cardiovascular'] = 0
        df['sofa_cns'] = 0
        df['sofa_renal'] = 0

        # 1. Respiration (PaO2/FiO2)
        if 'PaO2_FiO2' in df.columns:
            df['sofa_respiration'] = df['PaO2_FiO2'].apply(self._sofa_respiration)
        elif 'paO2' in df.columns and 'FiO2_1' in df.columns:
            # Compute PaO2/FiO2 ratio
            df['PaO2_FiO2'] = df['paO2'] / (df['FiO2_1'] / 100)  # FiO2 is often stored as percentage
            df['sofa_respiration'] = df['PaO2_FiO2'].apply(self._sofa_respiration)
        elif not required_components:
            logger.warning("  Missing PaO2/FiO2 - respiration score = 0")
        else:
            logger.error("  Cannot compute SOFA respiration - missing PaO2/FiO2")

        # 2. Coagulation (Platelets)
        if 'Platelets_count' in df.columns:
            df['sofa_coagulation'] = df['Platelets_count'].apply(self._sofa_coagulation)
        elif not required_components:
            logger.warning("  Missing Platelets - coagulation score = 0")
        else:
            logger.error("  Cannot compute SOFA coagulation - missing Platelets")

        # 3. Liver (Bilirubin)
        if 'Total_bili' in df.columns:
            df['sofa_liver'] = df['Total_bili'].apply(self._sofa_liver)
        elif not required_components:
            logger.warning("  Missing Total_bili - liver score = 0")
        else:
            logger.error("  Cannot compute SOFA liver - missing Total_bili")

        # 4. Cardiovascular (MAP + vasopressors)
        if 'MeanBP' in df.columns or 'max_dose_vaso' in df.columns:
            df['sofa_cardiovascular'] = df.apply(self._sofa_cardiovascular, axis=1)
        elif not required_components:
            logger.warning("  Missing MeanBP/vasopressors - cardiovascular score = 0")
        else:
            logger.error("  Cannot compute SOFA cardiovascular - missing MeanBP/vasopressors")

        # 5. CNS (GCS)
        if 'GCS' in df.columns:
            df['sofa_cns'] = df['GCS'].apply(self._sofa_cns)
        elif not required_components:
            logger.warning("  Missing GCS - CNS score = 0")
        else:
            logger.error("  Cannot compute SOFA CNS - missing GCS")

        # 6. Renal (Creatinine)
        if 'Creatinine' in df.columns:
            df['sofa_renal'] = df['Creatinine'].apply(self._sofa_renal)
        elif not required_components:
            logger.warning("  Missing Creatinine - renal score = 0")
        else:
            logger.error("  Cannot compute SOFA renal - missing Creatinine")

        # Total SOFA score
        df['SOFA'] = (
            df['sofa_respiration'] +
            df['sofa_coagulation'] +
            df['sofa_liver'] +
            df['sofa_cardiovascular'] +
            df['sofa_cns'] +
            df['sofa_renal']
        )

        logger.info(f"✓ SOFA scores computed for {len(df):,} observations")
        logger.info(f"  SOFA range: {df['SOFA'].min():.1f} - {df['SOFA'].max():.1f}")
        logger.info(f"  SOFA mean: {df['SOFA'].mean():.1f} ± {df['SOFA'].std():.1f}")

        return df

    def _sofa_respiration(self, pao2_fio2: float) -> int:
        """SOFA respiration component (PaO2/FiO2 ratio)."""
        if pd.isna(pao2_fio2):
            return 0
        if pao2_fio2 >= 400:
            return 0
        elif pao2_fio2 >= 300:
            return 1
        elif pao2_fio2 >= 200:
            return 2
        elif pao2_fio2 >= 100:
            return 3
        else:
            return 4

    def _sofa_coagulation(self, platelets: float) -> int:
        """SOFA coagulation component (platelets × 10³/μL)."""
        if pd.isna(platelets):
            return 0
        if platelets >= 150:
            return 0
        elif platelets >= 100:
            return 1
        elif platelets >= 50:
            return 2
        elif platelets >= 20:
            return 3
        else:
            return 4

    def _sofa_liver(self, bilirubin: float) -> int:
        """SOFA liver component (bilirubin mg/dL)."""
        if pd.isna(bilirubin):
            return 0
        if bilirubin < 1.2:
            return 0
        elif bilirubin < 2.0:
            return 1
        elif bilirubin < 6.0:
            return 2
        elif bilirubin < 12.0:
            return 3
        else:
            return 4

    def _sofa_cardiovascular(self, row: pd.Series) -> int:
        """
        SOFA cardiovascular component (MAP + vasopressors).

        Scoring:
        0: MAP ≥ 70 mmHg
        1: MAP < 70 mmHg
        2: Dopamine ≤ 5 or dobutamine (any dose)
        3: Dopamine > 5 OR epinephrine ≤ 0.1 OR norepinephrine ≤ 0.1
        4: Dopamine > 15 OR epinephrine > 0.1 OR norepinephrine > 0.1
        """
        mean_bp = row.get('MeanBP', np.nan)
        vaso_dose = row.get('max_dose_vaso', 0)

        # Convert to norepinephrine-equivalent (assumed in max_dose_vaso)
        if pd.notna(vaso_dose) and vaso_dose > 0:
            if vaso_dose > 0.1:
                return 4
            else:
                return 3
        elif pd.notna(mean_bp):
            if mean_bp < 70:
                return 1
            else:
                return 0
        else:
            return 0

    def _sofa_cns(self, gcs: float) -> int:
        """SOFA CNS component (Glasgow Coma Scale)."""
        if pd.isna(gcs):
            return 0
        if gcs >= 15:
            return 0
        elif gcs >= 13:
            return 1
        elif gcs >= 10:
            return 2
        elif gcs >= 6:
            return 3
        else:
            return 4

    def _sofa_renal(self, creatinine: float) -> int:
        """SOFA renal component (creatinine mg/dL)."""
        if pd.isna(creatinine):
            return 0
        if creatinine < 1.2:
            return 0
        elif creatinine < 2.0:
            return 1
        elif creatinine < 3.5:
            return 2
        elif creatinine < 5.0:
            return 3
        else:
            return 4

    def compute_rewards(
        self,
        trajectories: pd.DataFrame,
        cohort: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute rewards for all state transitions.

        Args:
            trajectories: DataFrame with columns: stay_id, time_window, SOFA, ...
            cohort: DataFrame with mortality information (stay_id, hospital_expire_flag)

        Returns:
            DataFrame with added 'reward' column
        """
        logger.info("Computing rewards...")

        df = trajectories.copy()

        # Merge with mortality data
        mortality_data = cohort[['stay_id', 'hospital_expire_flag']].drop_duplicates()
        df = df.merge(mortality_data, on='stay_id', how='left')

        # Sort by stay and time
        df = df.sort_values(['stay_id', 'time_window']).reset_index(drop=True)

        # Compute SOFA change (current - previous)
        df['sofa_prev'] = df.groupby('stay_id')['SOFA'].shift(1)
        df['sofa_change'] = df['SOFA'] - df['sofa_prev']

        # Identify terminal states (last time window for each stay)
        df['is_terminal'] = False
        last_time_windows = df.groupby('stay_id')['time_window'].transform('max')
        df.loc[df['time_window'] == last_time_windows, 'is_terminal'] = True

        # Initialize rewards
        df['reward'] = 0.0

        # Terminal rewards
        terminal_mask = df['is_terminal']
        survived_mask = terminal_mask & (df['hospital_expire_flag'] == 0)
        died_mask = terminal_mask & (df['hospital_expire_flag'] == 1)

        df.loc[survived_mask, 'reward'] = self.terminal_survival
        df.loc[died_mask, 'reward'] = self.terminal_death

        logger.info(f"  Terminal rewards: {survived_mask.sum():,} survivors, {died_mask.sum():,} deaths")

        # Intermediate rewards (based on SOFA change)
        intermediate_mask = ~terminal_mask & df['sofa_prev'].notna()

        # SOFA decreased (improvement)
        improved_mask = intermediate_mask & (df['sofa_change'] < 0)
        df.loc[improved_mask, 'reward'] = (
            df.loc[improved_mask, 'sofa_change'].abs() * self.intermediate_sofa_decrease
        )

        # SOFA increased (worsening)
        worsened_mask = intermediate_mask & (df['sofa_change'] > 0)
        df.loc[worsened_mask, 'reward'] = (
            -df.loc[worsened_mask, 'sofa_change'] * abs(self.intermediate_sofa_increase)
        )

        # No change
        no_change_mask = intermediate_mask & (df['sofa_change'] == 0)
        df.loc[no_change_mask, 'reward'] = self.intermediate_no_change

        logger.info(f"  Intermediate rewards: {improved_mask.sum():,} improved, {worsened_mask.sum():,} worsened, {no_change_mask.sum():,} no change")
        logger.info(f"✓ Rewards computed")
        logger.info(f"  Reward range: {df['reward'].min():.2f} to {df['reward'].max():.2f}")
        logger.info(f"  Mean reward: {df['reward'].mean():.3f}")

        return df

    def compute_returns(
        self,
        trajectories: pd.DataFrame,
        gamma: float = 0.99
    ) -> pd.DataFrame:
        """
        Compute discounted returns (cumulative future rewards) for each state.

        Args:
            trajectories: DataFrame with rewards
            gamma: Discount factor

        Returns:
            DataFrame with added 'return' column
        """
        logger.info(f"Computing returns (γ={gamma})...")

        df = trajectories.copy()

        # Sort by stay and time
        df = df.sort_values(['stay_id', 'time_window']).reset_index(drop=True)

        # Compute returns using reverse cumulative sum with discount
        def compute_stay_returns(stay_df):
            """Compute returns for a single ICU stay."""
            rewards = stay_df['reward'].values
            n = len(rewards)
            returns = np.zeros(n)

            # Backward pass: compute discounted returns
            returns[-1] = rewards[-1]
            for t in range(n-2, -1, -1):
                returns[t] = rewards[t] + gamma * returns[t+1]

            stay_df['return'] = returns
            return stay_df

        # Apply to each stay
        df = df.groupby('stay_id', group_keys=False).apply(compute_stay_returns)

        logger.info(f"✓ Returns computed")
        logger.info(f"  Return range: {df['return'].min():.2f} to {df['return'].max():.2f}")
        logger.info(f"  Mean return: {df['return'].mean():.3f}")

        return df


def main():
    """Example usage of RewardComputer."""
    from src.utils.config_loader import ConfigLoader

    # Load config
    config = ConfigLoader('configs/config.yaml').config

    # Initialize computer
    computer = RewardComputer(config)

    logger.info("RewardComputer initialized")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
