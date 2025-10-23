#!/usr/bin/env python3
"""
Precompute dataset normalization statistics (mean and std) and save them to disk.
This script should be run once before training to avoid recalculating statistics each time.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from dataset.dataset import TCDataset, load_config

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def precompute_statistics(config_path: str, output_dir: str):
    """
    Precompute normalization statistics from training data and save to JSON files.

    Args:
        config_path: Path to configuration file
        output_dir: Directory to save statistics files
    """
    logger = setup_logging()

    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create training dataset to compute statistics
    logger.info("Creating training dataset to compute statistics...")
    logger.info(f"Training years: {config['data']['train_years']}")

    train_dataset = TCDataset(
        config,
        years=config['data']['train_years'],
        split='train'
    )

    # Extract statistics
    coord_stats = train_dataset.coord_stats
    intensity_stats = train_dataset.intensity_stats

    # Save coordinate statistics
    coord_stats_path = os.path.join(output_dir, 'coord_stats.json')
    with open(coord_stats_path, 'w', encoding='utf-8') as f:
        json.dump(coord_stats, f, indent=2)
    logger.info(f"Saved coordinate statistics to: {coord_stats_path}")
    logger.info(f"  lat_std: {coord_stats['lat_std']:.6f}")
    logger.info(f"  lon_std: {coord_stats['lon_std']:.6f}")
    logger.info(f"  lat_mean: {coord_stats['lat_mean']:.6f}")
    logger.info(f"  lon_mean: {coord_stats['lon_mean']:.6f}")

    # Save intensity statistics
    intensity_stats_path = os.path.join(output_dir, 'intensity_stats.json')
    with open(intensity_stats_path, 'w', encoding='utf-8') as f:
        json.dump(intensity_stats, f, indent=2)
    logger.info(f"Saved intensity statistics to: {intensity_stats_path}")
    logger.info(f"  wind_mean: {intensity_stats['wind_mean']:.6f} kt")
    logger.info(f"  wind_std: {intensity_stats['wind_std']:.6f} kt")
    logger.info(f"  pres_mean: {intensity_stats['pres_mean']:.6f} hPa")
    logger.info(f"  pres_std: {intensity_stats['pres_std']:.6f} hPa")

    logger.info("Statistics precomputation completed successfully!")

    return coord_stats, intensity_stats

def main():
    parser = argparse.ArgumentParser(description='Precompute dataset statistics')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='Directory to save statistics (default: dataset)')

    args = parser.parse_args()

    precompute_statistics(args.config, args.output_dir)

if __name__ == '__main__':
    main()
