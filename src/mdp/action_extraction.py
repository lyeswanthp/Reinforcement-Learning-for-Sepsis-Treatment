"""
Action Extraction Module

Extracts and discretizes actions (IV fluids + vasopressors) from MIMIC-IV data.
Creates a discrete action space of 25 actions (5 IV bins × 5 vasopressor bins).

Author: AI Clinician Project
Date: 2024-11-16
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ActionExtractor:
    """
    Extracts and discretizes medical actions from MIMIC-IV data.

    Action Space: 25 discrete actions
    - IV Fluids: 5 bins (0-4)
    - Vasopressors: 5 bins (0-4)
    - Total: 5 × 5 = 25 action combinations

    The bins are created using percentile-based discretization on the training data.
    """

    def __init__(self, config: Dict):
        """
        Initialize ActionExtractor.

        Args:
            config: Configuration dictionary containing action_space settings
        """
        self.config = config
        self.n_iv_bins = config.get('action_space', {}).get('iv_fluid_bins', 5)
        self.n_vaso_bins = config.get('action_space', {}).get('vasopressor_bins', 5)
        self.n_actions = self.n_iv_bins * self.n_vaso_bins

        # Will be fitted on training data
        self.iv_bins: Optional[np.ndarray] = None
        self.vaso_bins: Optional[np.ndarray] = None

        self.fitted = False

        logger.info(f"ActionExtractor initialized: {self.n_iv_bins} IV bins × {self.n_vaso_bins} vaso bins = {self.n_actions} actions")

    def load_actions_from_events(
        self,
        inputevents_path: str,
        cohort: pd.DataFrame,
        time_window_hours: int = 4
    ) -> pd.DataFrame:
        """
        Load and aggregate actions from MIMIC-IV inputevents.

        Args:
            inputevents_path: Path to inputevents.csv
            cohort: DataFrame with cohort ICU stays
            time_window_hours: Time window for aggregation (default: 4 hours)

        Returns:
            DataFrame with columns: stay_id, time_window, iv_fluids, vasopressor_dose
        """
        logger.info(f"Loading actions from {inputevents_path}...")

        # Load inputevents
        logger.info("  Reading inputevents file...")
        inputevents = pd.read_csv(
            inputevents_path,
            usecols=['stay_id', 'starttime', 'endtime', 'itemid', 'amount', 'rate', 'amountuom', 'rateuom'],
            parse_dates=['starttime', 'endtime']
        )

        # Filter for cohort stays
        cohort_stay_ids = set(cohort['stay_id'].unique())
        inputevents = inputevents[inputevents['stay_id'].isin(cohort_stay_ids)]
        logger.info(f"  Filtered to {len(inputevents):,} events for cohort")

        # Define item IDs for IV fluids and vasopressors
        # These are common MIMIC-IV item IDs - adjust based on your dataset
        iv_fluid_items = self._get_iv_fluid_item_ids()
        vasopressor_items = self._get_vasopressor_item_ids()

        # Extract IV fluids
        logger.info("  Extracting IV fluid administration...")
        iv_events = inputevents[inputevents['itemid'].isin(iv_fluid_items)].copy()
        logger.info(f"    Found {len(iv_events):,} IV fluid events")

        # Extract vasopressors
        logger.info("  Extracting vasopressor administration...")
        vaso_events = inputevents[inputevents['itemid'].isin(vasopressor_items)].copy()
        logger.info(f"    Found {len(vaso_events):,} vasopressor events")

        # Get ICU stay times for windowing
        stay_times = cohort[['stay_id', 'intime', 'outtime']].drop_duplicates()
        stay_times['intime'] = pd.to_datetime(stay_times['intime'])
        stay_times['outtime'] = pd.to_datetime(stay_times['outtime'])

        # Aggregate IV fluids by time window
        logger.info(f"  Aggregating actions into {time_window_hours}-hour windows...")
        iv_aggregated = self._aggregate_iv_fluids(iv_events, stay_times, time_window_hours)
        vaso_aggregated = self._aggregate_vasopressors(vaso_events, stay_times, time_window_hours)

        # Merge IV and vasopressor data
        actions = pd.merge(
            iv_aggregated,
            vaso_aggregated,
            on=['stay_id', 'time_window'],
            how='outer'
        )

        # Fill missing values with 0 (no administration)
        actions['iv_fluids'] = actions['iv_fluids'].fillna(0)
        actions['vasopressor_dose'] = actions['vasopressor_dose'].fillna(0)

        logger.info(f"✓ Extracted actions: {len(actions):,} stay-time windows")
        logger.info(f"  IV fluids range: {actions['iv_fluids'].min():.0f} - {actions['iv_fluids'].max():.0f} mL")
        logger.info(f"  Vasopressor range: {actions['vasopressor_dose'].min():.4f} - {actions['vasopressor_dose'].max():.4f} μg/kg/min")

        return actions

    def _get_iv_fluid_item_ids(self) -> List[int]:
        """Get item IDs for IV fluids from MIMIC-IV."""
        # Common IV fluid item IDs in MIMIC-IV
        # This list should be verified against your d_items table
        return [
            # Crystalloids
            220949,  # 0.9% Normal Saline
            220950,  # D5W
            225158,  # NaCl 0.9%
            225828,  # Lactated Ringers
            225159,  # NaCl 0.45%
            225161,  # Dextrose 5%
            225944,  # Sterile Water
            # Colloids
            220864,  # Albumin 5%
            220862,  # Albumin 25%
            225174,  # Hetastarch
            # Other fluids
            226364,  # OR Crystalloid Intake
            226365,  # OR Cell Saver Intake
        ]

    def _get_vasopressor_item_ids(self) -> Dict[int, str]:
        """
        Get item IDs for vasopressors from MIMIC-IV.

        Returns:
            Dictionary mapping item ID to drug name
        """
        return {
            221906: 'norepinephrine',  # Norepinephrine (mcg/min)
            221289: 'epinephrine',     # Epinephrine (mcg/min)
            221749: 'phenylephrine',   # Phenylephrine (mcg/min)
            222315: 'vasopressin',     # Vasopressin (units/min)
            221662: 'dopamine',        # Dopamine (mcg/kg/min)
            221653: 'dobutamine',      # Dobutamine (mcg/kg/min)
        }

    def _aggregate_iv_fluids(
        self,
        iv_events: pd.DataFrame,
        stay_times: pd.DataFrame,
        time_window_hours: int
    ) -> pd.DataFrame:
        """
        Aggregate IV fluid amounts by time window.

        Args:
            iv_events: DataFrame with IV fluid events
            stay_times: DataFrame with ICU stay times
            time_window_hours: Time window size

        Returns:
            DataFrame with stay_id, time_window, iv_fluids (mL)
        """
        if len(iv_events) == 0:
            return pd.DataFrame(columns=['stay_id', 'time_window', 'iv_fluids'])

        # Merge with stay times
        iv_events = iv_events.merge(stay_times, on='stay_id', how='left')

        # Calculate time window for each event (based on starttime)
        iv_events['hours_since_intime'] = (
            iv_events['starttime'] - iv_events['intime']
        ).dt.total_seconds() / 3600

        iv_events['time_window'] = (
            iv_events['hours_since_intime'] / time_window_hours
        ).astype(int)

        # Filter out events outside ICU stay or negative time windows
        iv_events = iv_events[
            (iv_events['time_window'] >= 0) &
            (iv_events['starttime'] <= iv_events['outtime'])
        ]

        # Aggregate by stay and time window
        # Sum the amounts (in mL)
        iv_aggregated = iv_events.groupby(['stay_id', 'time_window']).agg({
            'amount': 'sum'  # Total IV fluids in mL
        }).reset_index()

        iv_aggregated.rename(columns={'amount': 'iv_fluids'}, inplace=True)

        return iv_aggregated

    def _aggregate_vasopressors(
        self,
        vaso_events: pd.DataFrame,
        stay_times: pd.DataFrame,
        time_window_hours: int
    ) -> pd.DataFrame:
        """
        Aggregate vasopressor doses by time window.

        Converts all vasopressors to norepinephrine-equivalent doses.

        Args:
            vaso_events: DataFrame with vasopressor events
            stay_times: DataFrame with ICU stay times
            time_window_hours: Time window size

        Returns:
            DataFrame with stay_id, time_window, vasopressor_dose (μg/kg/min NE-equivalent)
        """
        if len(vaso_events) == 0:
            return pd.DataFrame(columns=['stay_id', 'time_window', 'vasopressor_dose'])

        # Merge with stay times
        vaso_events = vaso_events.merge(stay_times, on='stay_id', how='left')

        # Calculate time window
        vaso_events['hours_since_intime'] = (
            vaso_events['starttime'] - vaso_events['intime']
        ).dt.total_seconds() / 3600

        vaso_events['time_window'] = (
            vaso_events['hours_since_intime'] / time_window_hours
        ).astype(int)

        # Filter out events outside ICU stay
        vaso_events = vaso_events[
            (vaso_events['time_window'] >= 0) &
            (vaso_events['starttime'] <= vaso_events['outtime'])
        ]

        # Convert to norepinephrine equivalents (NE-eq)
        # Equivalence factors based on clinical literature
        vaso_events['ne_equivalent'] = vaso_events.apply(
            self._convert_to_ne_equivalent, axis=1
        )

        # Aggregate by stay and time window (max dose in window)
        vaso_aggregated = vaso_events.groupby(['stay_id', 'time_window']).agg({
            'ne_equivalent': 'max'  # Max dose during window
        }).reset_index()

        vaso_aggregated.rename(columns={'ne_equivalent': 'vasopressor_dose'}, inplace=True)

        return vaso_aggregated

    def _convert_to_ne_equivalent(self, row: pd.Series) -> float:
        """
        Convert vasopressor dose to norepinephrine equivalent.

        Equivalence factors:
        - Norepinephrine: 1.0
        - Epinephrine: 1.0 (similar potency)
        - Phenylephrine: 0.1 (weaker)
        - Vasopressin: 2.0 (convert units/min to mcg/kg/min equivalent)
        - Dopamine: 0.01 (much weaker at typical doses)
        - Dobutamine: 0 (not a vasopressor)

        Args:
            row: Row from vasopressor events

        Returns:
            Norepinephrine-equivalent dose (μg/kg/min)
        """
        vasopressor_map = self._get_vasopressor_item_ids()
        drug_name = vasopressor_map.get(row['itemid'], 'unknown')

        rate = row.get('rate', 0)
        if pd.isna(rate) or rate == 0:
            return 0.0

        # Equivalence factors
        equivalence = {
            'norepinephrine': 1.0,
            'epinephrine': 1.0,
            'phenylephrine': 0.1,
            'vasopressin': 2.0,
            'dopamine': 0.01,
            'dobutamine': 0.0,  # Not a vasopressor
        }

        factor = equivalence.get(drug_name, 1.0)
        return rate * factor

    def fit(self, actions: pd.DataFrame, subset: str = 'train') -> 'ActionExtractor':
        """
        Fit action bins on training data using percentile-based discretization.

        Args:
            actions: DataFrame with columns: stay_id, time_window, iv_fluids, vasopressor_dose
            subset: Name of subset being fitted (for logging)

        Returns:
            self
        """
        logger.info(f"Fitting action bins on {subset} data ({len(actions):,} observations)...")

        # Extract continuous action values (exclude zeros for percentile calculation)
        iv_values = actions[actions['iv_fluids'] > 0]['iv_fluids'].values
        vaso_values = actions[actions['vasopressor_dose'] > 0]['vasopressor_dose'].values

        logger.info(f"  IV fluids: {len(iv_values):,} non-zero observations")
        logger.info(f"  Vasopressors: {len(vaso_values):,} non-zero observations")

        # Create bins using percentiles
        # Bin 0: always for zero (no administration)
        # Bins 1-4: percentile-based on non-zero values

        if len(iv_values) > 0:
            # For IV fluids: [0, 25th, 50th, 75th, 100th percentiles]
            percentiles = [0, 25, 50, 75, 100]
            iv_percentile_values = np.percentile(iv_values, percentiles)

            # Create bins: [0, ..., infinity]
            self.iv_bins = np.array([0] + list(iv_percentile_values[1:]) + [np.inf])

            logger.info(f"  IV fluid bins: {self.iv_bins[:-1]}")
        else:
            self.iv_bins = np.array([0, np.inf])
            logger.warning("  No IV fluid data - using default bins")

        if len(vaso_values) > 0:
            vaso_percentile_values = np.percentile(vaso_values, percentiles)
            self.vaso_bins = np.array([0] + list(vaso_percentile_values[1:]) + [np.inf])

            logger.info(f"  Vasopressor bins: {self.vaso_bins[:-1]}")
        else:
            self.vaso_bins = np.array([0, np.inf])
            logger.warning("  No vasopressor data - using default bins")

        self.fitted = True
        logger.info("✓ Action bins fitted successfully")

        return self

    def transform(self, actions: pd.DataFrame) -> pd.DataFrame:
        """
        Transform continuous actions to discrete action indices.

        Args:
            actions: DataFrame with continuous actions (iv_fluids, vasopressor_dose)

        Returns:
            DataFrame with added columns: iv_bin, vaso_bin, action
        """
        if not self.fitted:
            raise ValueError("ActionExtractor must be fitted before transform. Call fit() first.")

        logger.info(f"Discretizing {len(actions):,} actions...")

        actions = actions.copy()

        # Discretize IV fluids
        actions['iv_bin'] = np.digitize(actions['iv_fluids'], self.iv_bins[1:-1], right=False)

        # Discretize vasopressors
        actions['vaso_bin'] = np.digitize(actions['vasopressor_dose'], self.vaso_bins[1:-1], right=False)

        # Combine into single action index
        # action = iv_bin * n_vaso_bins + vaso_bin
        actions['action'] = actions['iv_bin'] * self.n_vaso_bins + actions['vaso_bin']

        # Ensure action is in valid range [0, n_actions-1]
        actions['action'] = actions['action'].clip(0, self.n_actions - 1)

        logger.info(f"✓ Discretization complete")
        logger.info(f"  Action distribution:")
        action_counts = actions['action'].value_counts().sort_index()
        for action_id, count in action_counts.head(10).items():
            iv_bin = action_id // self.n_vaso_bins
            vaso_bin = action_id % self.n_vaso_bins
            logger.info(f"    Action {action_id} (IV={iv_bin}, Vaso={vaso_bin}): {count:,} ({count/len(actions)*100:.1f}%)")

        if len(action_counts) > 10:
            logger.info(f"    ... and {len(action_counts)-10} more actions")

        return actions

    def fit_transform(self, actions: pd.DataFrame, subset: str = 'train') -> pd.DataFrame:
        """
        Fit action bins and transform in one step.

        Args:
            actions: DataFrame with continuous actions
            subset: Name of subset (for logging)

        Returns:
            DataFrame with discrete actions
        """
        return self.fit(actions, subset).transform(actions)

    def save_bins(self, output_path: str):
        """
        Save fitted action bins to file.

        Args:
            output_path: Path to save bins (pickle format)
        """
        if not self.fitted:
            raise ValueError("Cannot save unfitted ActionExtractor")

        import pickle

        bins_data = {
            'iv_bins': self.iv_bins,
            'vaso_bins': self.vaso_bins,
            'n_iv_bins': self.n_iv_bins,
            'n_vaso_bins': self.n_vaso_bins,
            'n_actions': self.n_actions,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(bins_data, f)

        logger.info(f"✓ Action bins saved to {output_path}")

    def load_bins(self, input_path: str):
        """
        Load fitted action bins from file.

        Args:
            input_path: Path to load bins from
        """
        import pickle

        with open(input_path, 'rb') as f:
            bins_data = pickle.load(f)

        self.iv_bins = bins_data['iv_bins']
        self.vaso_bins = bins_data['vaso_bins']
        self.n_iv_bins = bins_data['n_iv_bins']
        self.n_vaso_bins = bins_data['n_vaso_bins']
        self.n_actions = bins_data['n_actions']

        self.fitted = True

        logger.info(f"✓ Action bins loaded from {input_path}")

    def get_action_description(self, action: int) -> str:
        """
        Get human-readable description of an action.

        Args:
            action: Discrete action index (0 to n_actions-1)

        Returns:
            String description of the action
        """
        if not self.fitted:
            raise ValueError("ActionExtractor must be fitted first")

        iv_bin = action // self.n_vaso_bins
        vaso_bin = action % self.n_vaso_bins

        # Get bin ranges
        if iv_bin < len(self.iv_bins) - 1:
            iv_range = f"{self.iv_bins[iv_bin]:.0f}-{self.iv_bins[iv_bin+1]:.0f} mL"
        else:
            iv_range = f">{self.iv_bins[-2]:.0f} mL"

        if vaso_bin < len(self.vaso_bins) - 1:
            vaso_range = f"{self.vaso_bins[vaso_bin]:.3f}-{self.vaso_bins[vaso_bin+1]:.3f} μg/kg/min"
        else:
            vaso_range = f">{self.vaso_bins[-2]:.3f} μg/kg/min"

        return f"Action {action}: IV={iv_range}, Vaso={vaso_range}"


def main():
    """Example usage of ActionExtractor."""
    from src.utils.config_loader import ConfigLoader

    # Load config
    config = ConfigLoader('configs/config.yaml').config

    # Initialize extractor
    extractor = ActionExtractor(config)

    logger.info("ActionExtractor initialized")
    logger.info(f"Action space: {extractor.n_actions} discrete actions")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
