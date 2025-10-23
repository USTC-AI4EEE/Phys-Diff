#!/usr/bin/env python3
"""
Precompute ERA5 channel-wise normalization statistics.

Computes global mean and std for each channel using Welford's online algorithm
to avoid memory overflow during training data preprocessing.
"""

import os
import sys
import numpy as np
import json
import logging
from tqdm import tqdm
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from dataset.dataset import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_env_channel_statistics(config, output_dir='dataset'):
    """
    Compute global statistics for all 69 ERA5 channels.

    Uses Welford's online algorithm to avoid loading all data into memory.

    Args:
        config: Configuration dictionary
        output_dir: Output directory

    Returns:
        stats: Statistics dictionary
    """

    tc_cropped_path = config['data']['tc_cropped_path']
    years = config['data']['train_years']

    logger.info(f"Computing environment channel statistics from training years: {years}")
    logger.info(f"Data path: {tc_cropped_path}")

    # Welford's online algorithm accumulators
    n_channels = 69
    channel_means = np.zeros(n_channels, dtype=np.float64)
    channel_m2 = np.zeros(n_channels, dtype=np.float64)
    channel_counts = np.zeros(n_channels, dtype=np.int64)

    total_files = 0
    processed_files = 0

    logger.info("Scanning for ERA5 files...")
    for year in years:
        year_dir = os.path.join(tc_cropped_path, str(year))
        if not os.path.exists(year_dir):
            logger.warning(f"Year directory not found: {year_dir}")
            continue

        for sid in os.listdir(year_dir):
            sid_dir = os.path.join(year_dir, sid)
            if not os.path.isdir(sid_dir):
                continue

            for timestamp_dir in os.listdir(sid_dir):
                timestamp_path = os.path.join(sid_dir, timestamp_dir)
                if not os.path.isdir(timestamp_path):
                    continue

                era5_file = os.path.join(timestamp_path, "ERA5_data.npy")
                if os.path.exists(era5_file):
                    total_files += 1

    logger.info(f"Found {total_files} ERA5 files to process")

    pbar = tqdm(total=total_files, desc="Computing statistics")

    for year in years:
        year_dir = os.path.join(tc_cropped_path, str(year))
        if not os.path.exists(year_dir):
            continue

        for sid in os.listdir(year_dir):
            sid_dir = os.path.join(year_dir, sid)
            if not os.path.isdir(sid_dir):
                continue

            for timestamp_dir in os.listdir(sid_dir):
                timestamp_path = os.path.join(sid_dir, timestamp_dir)
                if not os.path.isdir(timestamp_path):
                    continue

                era5_file = os.path.join(timestamp_path, "ERA5_data.npy")

                if os.path.exists(era5_file):
                    try:
                        era5_data = np.load(era5_file)

                        if era5_data.shape != (69, 80, 80):
                            logger.warning(f"Unexpected shape {era5_data.shape} in {era5_file}, skipping")
                            continue

                        # Update channel statistics using Welford's algorithm
                        for ch in range(n_channels):
                            channel_data = era5_data[ch].flatten()

                            for value in channel_data:
                                channel_counts[ch] += 1
                                delta = value - channel_means[ch]
                                channel_means[ch] += delta / channel_counts[ch]
                                delta2 = value - channel_means[ch]
                                channel_m2[ch] += delta * delta2

                        processed_files += 1
                        pbar.update(1)

                    except Exception as e:
                        logger.error(f"Error processing {era5_file}: {e}")
                        continue

    pbar.close()

    if processed_files == 0:
        raise ValueError("No ERA5 files were processed successfully!")

    logger.info(f"Successfully processed {processed_files}/{total_files} files")

    channel_stds = np.sqrt(channel_m2 / channel_counts)

    # Validate statistics
    for ch in range(n_channels):
        if channel_counts[ch] == 0:
            logger.warning(f"Channel {ch} has no data!")
        elif channel_stds[ch] < 1e-8:
            logger.warning(f"Channel {ch} has very small std: {channel_stds[ch]}")

    stats = {
        'channel_means': channel_means.tolist(),
        'channel_stds': channel_stds.tolist(),
        'channel_counts': channel_counts.tolist(),
        'n_channels': n_channels,
        'n_files_processed': int(processed_files),
        'train_years': years
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'env_channel_stats.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved channel statistics to {output_path}")

    logger.info("=" * 80)
    logger.info("ENVIRONMENT CHANNEL STATISTICS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Number of channels: {n_channels}")
    logger.info(f"Total samples per channel: {channel_counts[0]:,}")
    logger.info(f"Mean range: [{channel_means.min():.4f}, {channel_means.max():.4f}]")
    logger.info(f"Std range: [{channel_stds.min():.4f}, {channel_stds.max():.4f}]")
    logger.info("=" * 80)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Precompute environment channel statistics')
    parser.add_argument('--config', type=str, default='configs/config_wo_fengwu.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='Output directory for statistics file')
    args = parser.parse_args()

    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    stats = compute_env_channel_statistics(config, args.output_dir)

    logger.info("Environment statistics computation completed successfully!")

    output_path = os.path.join(args.output_dir, 'env_channel_stats.json')
    with open(output_path, 'r', encoding='utf-8') as f:
        loaded_stats = json.load(f)

    logger.info(f"Verification: Successfully loaded statistics from {output_path}")
    logger.info(f"  - Channels: {loaded_stats['n_channels']}")
    logger.info(f"  - Files processed: {loaded_stats['n_files_processed']}")
    logger.info(f"  - Training years: {loaded_stats['train_years']}")


if __name__ == '__main__':
    main()
