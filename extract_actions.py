#!/usr/bin/env python3
"""
Extract Actions from MIMIC-IV inputevents

Extracts IV fluid and vasopressor administration, discretizes into action bins.

Usage:
    python extract_actions.py --cohort data/processed/cohort.csv \
                               --inputevents-path /path/to/inputevents.csv \
                               --output data/processed/
"""

import argparse
import pandas as pd
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.mdp import ActionExtractor
from src.utils.config_loader import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Extract actions from MIMIC-IV inputevents')
    parser.add_argument('--cohort', required=True, help='Path to cohort.csv')
    parser.add_argument('--inputevents-path', required=True, help='Path to inputevents.csv')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("ACTION EXTRACTION")
    logger.info("="*80)

    # Load config and cohort
    logger.info(f"Loading configuration from {args.config}...")
    config = ConfigLoader(args.config).config

    logger.info(f"Loading cohort from {args.cohort}...")
    cohort = pd.read_csv(args.cohort)
    logger.info(f"  Cohort size: {len(cohort):,} ICU stays")

    # Extract actions
    logger.info("\nExtracting actions from inputevents...")
    extractor = ActionExtractor(config)

    actions = extractor.load_actions_from_events(
        args.inputevents_path,
        cohort,
        time_window_hours=config.get('mdp', {}).get('time_window_hours', 4)
    )

    # Load processed features to get train/val/test splits
    logger.info("\nLoading processed features to determine splits...")
    train_features = pd.read_csv(f"{args.output}/train_features.csv")
    val_features = pd.read_csv(f"{args.output}/val_features.csv")
    test_features = pd.read_csv(f"{args.output}/test_features.csv")

    # Split actions by stay_id
    train_stays = set(train_features['stay_id'].unique())
    val_stays = set(val_features['stay_id'].unique())
    test_stays = set(test_features['stay_id'].unique())

    logger.info(f"  Train stays: {len(train_stays):,}")
    logger.info(f"  Val stays: {len(val_stays):,}")
    logger.info(f"  Test stays: {len(test_stays):,}")

    train_actions = actions[actions['stay_id'].isin(train_stays)].copy()
    val_actions = actions[actions['stay_id'].isin(val_stays)].copy()
    test_actions = actions[actions['stay_id'].isin(test_stays)].copy()

    logger.info(f"\n  Train actions: {len(train_actions):,} observations")
    logger.info(f"  Val actions: {len(val_actions):,} observations")
    logger.info(f"  Test actions: {len(test_actions):,} observations")

    # Fit action bins on training data
    logger.info("\nDiscretizing actions...")
    train_actions = extractor.fit_transform(train_actions, 'train')
    val_actions = extractor.transform(val_actions)
    test_actions = extractor.transform(test_actions)

    # Save
    logger.info("\nSaving action data...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_actions.to_csv(output_dir / "train_actions.csv", index=False)
    val_actions.to_csv(output_dir / "val_actions.csv", index=False)
    test_actions.to_csv(output_dir / "test_actions.csv", index=False)
    extractor.save_bins(str(output_dir / "action_bins.pkl"))

    logger.info(f"  ✓ Saved train_actions.csv")
    logger.info(f"  ✓ Saved val_actions.csv")
    logger.info(f"  ✓ Saved test_actions.csv")
    logger.info(f"  ✓ Saved action_bins.pkl")

    logger.info("\n" + "="*80)
    logger.info("ACTION EXTRACTION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()