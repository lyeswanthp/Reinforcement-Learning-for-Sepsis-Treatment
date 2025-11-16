#!/usr/bin/env python3
"""
Run MIMIC-IV Preprocessing Pipeline
Main entry point for preprocessing all MIMIC-IV CSV files
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import ConfigLoader
from src.preprocessing.preprocessing_pipeline import MIMICPreprocessingPipeline


def setup_logging(log_dir: Path, log_level: str = 'INFO'):
    """
    Setup logging configuration

    Args:
        log_dir: Directory for log files
        log_level: Logging level
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'preprocessing_{timestamp}.log'

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")

    return logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='MIMIC-IV Preprocessing Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to MIMIC-IV data directory (contains hosp/ and icu/ folders). '
             'If not specified, uses path from config file.'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for processed data. If not specified, uses config value.'
    )

    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        default=True,
        help='Save intermediate processing results'
    )

    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation at each stage (faster but less safe)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    parser.add_argument(
        '--test-run',
        action='store_true',
        help='Run on subset of data for testing (faster)'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Load configuration
    try:
        config_loader = ConfigLoader(args.config)
        config = config_loader.config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Setup logging
    log_dir = Path(config.get('output', {}).get('log_dir', 'logs'))
    logger = setup_logging(log_dir, args.log_level)

    logger.info("="*80)
    logger.info("MIMIC-IV PREPROCESSING PIPELINE")
    logger.info("="*80)
    logger.info(f"Configuration file: {args.config}")

    # Determine data path
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = config.get('data_source', {}).get('base_path', 'data/raw/mimic-iv-3.1')

    logger.info(f"Data path: {data_path}")

    # Verify data path exists
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        logger.error(f"Data path does not exist: {data_path}")
        logger.error("Please provide the correct path to your MIMIC-IV data using --data-path")
        sys.exit(1)

    # Check for hosp and icu directories
    hosp_dir = data_path_obj / 'hosp'
    icu_dir = data_path_obj / 'icu'

    if not hosp_dir.exists():
        logger.error(f"Hospital data directory not found: {hosp_dir}")
        sys.exit(1)

    if not icu_dir.exists():
        logger.error(f"ICU data directory not found: {icu_dir}")
        sys.exit(1)

    logger.info(f"✓ Found hospital data directory: {hosp_dir}")
    logger.info(f"✓ Found ICU data directory: {icu_dir}")

    # Update output directory if specified
    if args.output_dir:
        config['output']['processed_dir'] = args.output_dir

    try:
        # Initialize pipeline
        logger.info("\nInitializing preprocessing pipeline...")
        pipeline = MIMICPreprocessingPipeline(config, data_path)

        # Run pipeline
        logger.info("\nStarting preprocessing pipeline...\n")
        stats = pipeline.run_full_pipeline(
            save_intermediate=args.save_intermediate,
            validate_each_stage=not args.no_validate
        )

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info("\nPipeline Statistics:")
        logger.info("-"*40)
        for key, value in stats.items():
            if 'duration' in key:
                logger.info(f"{key:30s}: {value:>10.2f} seconds ({value/60:>6.2f} min)")
            elif isinstance(value, dict):
                logger.info(f"\n{key}:")
                for k, v in value.items():
                    logger.info(f"  {k:28s}: {v:>10,}" if isinstance(v, int) else f"  {k:28s}: {v}")
            else:
                logger.info(f"{key:30s}: {value}")

        logger.info("\n" + "="*80)
        logger.info("Output files saved to:")
        logger.info(f"  {config['output']['processed_dir']}")
        logger.info("="*80)

        return 0

    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error("PREPROCESSING FAILED")
        logger.error(f"{'='*80}")
        logger.error(f"Error: {e}")

        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
