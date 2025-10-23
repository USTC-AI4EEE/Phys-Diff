import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TCDataset(Dataset):
    """
    TC dataset with relative coordinate normalization.

    Implements relative coordinate normalization where coordinates are normalized
    relative to the starting point of each trajectory for improved numerical stability.
    Statistics are loaded from precomputed files.
    """

    def __init__(self, config: Dict, years: List[int], split: str = 'train',
                 stats_dir: str = 'dataset'):
        self.config = config
        self.years = years
        self.split = split
        self.history_steps = config['data']['history_steps']
        self.future_steps = config['data']['future_steps']
        self.stats_dir = stats_dir

        # Load IBTrACS data
        self.ibtracs = self._load_ibtracs()

        # Filter by years
        self.ibtracs_filtered = self._filter_by_years(self.ibtracs, self.years)

        # Create sequences
        self.sequences = self._create_sequences()

        # Load precomputed statistics
        self.coord_stats = self._load_coord_stats()
        self.intensity_stats = self._load_intensity_stats()
        self._load_env_channel_stats()  # Load environment channel statistics

        # Get FengWu setting for initialization summary
        self.use_fengwu = self.config['model'].get('ablation', {}).get('use_fengwu', True)

        # Output initialization summary once
        logger.info(f"Created {split} dataset with {len(self.sequences)} sequences")
        logger.info(f"Coord stats - lat_std: {self.coord_stats['lat_std']:.4f}, lon_std: {self.coord_stats['lon_std']:.4f}")
        logger.info(f"Intensity stats - wind: {self.intensity_stats['wind_mean']:.2f}±{self.intensity_stats['wind_std']:.2f} kt, pres: {self.intensity_stats['pres_mean']:.2f}±{self.intensity_stats['pres_std']:.2f} hPa")
        logger.info(f"FengWu integration: {'enabled' if self.use_fengwu else 'disabled'}")

    def _load_coord_stats(self) -> Dict:
        """Load precomputed coordinate statistics"""
        stats_path = os.path.join(self.stats_dir, 'coord_stats.json')
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Coordinate statistics file not found: {stats_path}\n"
                f"Please run: python scripts/precompute_statistics.py --config <config_file>"
            )

        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        logger.info(f"Loaded coordinate statistics from: {stats_path}")
        return stats

    def _load_intensity_stats(self) -> Dict:
        """Load precomputed intensity statistics"""
        stats_path = os.path.join(self.stats_dir, 'intensity_stats.json')
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Intensity statistics file not found: {stats_path}\n"
                f"Please run: python scripts/precompute_statistics.py --config <config_file>"
            )

        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        logger.info(f"Loaded intensity statistics from: {stats_path}")
        return stats

    def _load_env_channel_stats(self):
        """Load precomputed environment channel statistics for fast normalization."""
        stats_path = os.path.join(self.stats_dir, 'env_channel_stats.json')

        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)

                self.env_channel_means = np.array(stats['channel_means'], dtype=np.float32)
                self.env_channel_stds = np.array(stats['channel_stds'], dtype=np.float32)

                # Reshape to (69, 1, 1) for broadcasting
                self.env_channel_means = self.env_channel_means[:, np.newaxis, np.newaxis]
                self.env_channel_stds = self.env_channel_stds[:, np.newaxis, np.newaxis]

                self.use_precomputed_env_stats = True
                logger.info(f"Loaded environment channel statistics from: {stats_path}")
            except Exception as e:
                logger.warning(f"Failed to load environment statistics from {stats_path}: {e}")
                logger.warning("Falling back to per-sample normalization (slower)")
                self.use_precomputed_env_stats = False
        else:
            logger.warning(f"Environment statistics not found at {stats_path}")
            logger.warning("Falling back to per-sample normalization (slower)")
            logger.warning("To enable fast normalization, run: python scripts/precompute_env_statistics.py")
            self.use_precomputed_env_stats = False

    def _load_ibtracs(self) -> pd.DataFrame:
        """Load IBTrACS dataset"""
        ibtracs_path = self.config['data']['ibtracs_path']

        df = pd.read_csv(ibtracs_path, low_memory=False)
        required_columns = ['SID', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'USA_PRES']

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in IBTrACS data: {missing_cols}")

        df = df[required_columns]

        # Ensure correct types
        df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
        df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
        df['USA_WIND'] = pd.to_numeric(df['USA_WIND'], errors='coerce')
        df['USA_PRES'] = pd.to_numeric(df['USA_PRES'], errors='coerce')

        # Remove invalid data
        df = df.dropna()

        # Time processing
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])

        logger.info(f"Loaded {len(df)} valid IBTrACS records")
        return df

    def _filter_by_years(self, df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """Filter data by years and time resolution"""
        # First filter by years
        mask = df['ISO_TIME'].dt.year.isin(years)
        filtered_df = df[mask].copy()

        # Then filter by time resolution (6-hour intervals: 00, 06, 12, 18)
        time_resolution = self.config['data']['time_resolution']  # Should be 6 hours
        if time_resolution == 6:
            # Keep only timestamps at 00, 06, 12, 18 hours
            valid_hours = [0, 6, 12, 18]
            hour_mask = filtered_df['ISO_TIME'].dt.hour.isin(valid_hours)
            filtered_df = filtered_df[hour_mask].copy()
            logger.info(f"Applied {time_resolution}-hour time resolution filter")

        logger.info(f"Filtered to {len(filtered_df)} records for years {years} at {time_resolution}h resolution")
        return filtered_df

    def _create_sequences(self) -> List[Dict]:
        """Create training sequences"""
        sequences = []
        grouped = self.ibtracs_filtered.groupby('SID')

        total_storms = len(grouped)
        storms_with_data = 0
        processed_storms = 0

        for sid, group in grouped:
            processed_storms += 1

            if processed_storms % 100 == 0:  # Reduced frequency from 25 to 100
                logger.info(f"Processing storm {processed_storms}/{total_storms}: {sid}")

            group_sorted = group.sort_values('ISO_TIME')

            storm_sequences = []
            # Create sequences with sliding window
            for i in range(len(group_sorted) - self.history_steps - self.future_steps + 1):
                hist_data = group_sorted.iloc[i:i+self.history_steps]
                future_data = group_sorted.iloc[i+self.history_steps:i+self.history_steps+self.future_steps]

                # Check if environmental data exists
                # Historical data always uses ERA5
                if not self._check_env_data_exists(hist_data, use_fengwu=False, is_future=False):
                    continue

                # Future data: only check if FengWu is enabled
                use_fengwu_for_future = self.config['model'].get('ablation', {}).get('use_fengwu', True)
                if use_fengwu_for_future:
                    # FengWu enabled: check for FengWu data
                    if not self._check_env_data_exists(future_data, use_fengwu=True, is_future=True):
                        continue
                else:
                    # w/o FengWu: no future environmental data needed, skip check
                    # Just verify that future TC trajectory data exists (already done by data slicing)
                    pass

                storm_sequences.append({
                    'sid': sid,
                    'hist_data': hist_data,
                    'future_data': future_data
                })

            if storm_sequences:
                storms_with_data += 1
                sequences.extend(storm_sequences)

        return sequences

    def _check_env_data_exists(self, data: pd.DataFrame, use_fengwu: bool = False, is_future: bool = False) -> bool:
        """
        Check if environmental data files exist with exact timestamp matching.

        Args:
            data: DataFrame containing TC data
            use_fengwu: Whether FengWu data should be used
            is_future: Whether checking future steps

        Returns:
            bool: True if data exists
        """
        tc_cropped_path = self.config['data']['tc_cropped_path']

        # Check if storm directory exists first
        year = data.iloc[0]['ISO_TIME'].year if len(data) > 0 else None
        sid = data.iloc[0]['SID'] if len(data) > 0 else None

        if year is None or sid is None:
            return False

        storm_dir = os.path.join(tc_cropped_path, str(year), sid)
        if not os.path.exists(storm_dir):
            # Use hash for occasional logging to reduce noise
            if hash(str(storm_dir)) % 200 == 0:
                logger.debug(f"Storm directory not found: {storm_dir}")
            return False

        # Check each timestamp in the sequence
        for _, row in data.iterrows():
            year = row['ISO_TIME'].year
            sid = row['SID']
            time_str = row['ISO_TIME'].strftime('%Y-%m-%d %H_%M_%S')

            if use_fengwu:
                # Check for FengWu data first, fallback to ERA5
                fengwu_path = os.path.join(tc_cropped_path, str(year), sid, f"{time_str}/FengWu_data.npy")
                era5_path = os.path.join(tc_cropped_path, str(year), sid, f"{time_str}/ERA5_data.npy")

                fengwu_path = fengwu_path.replace('\\', '/')
                era5_path = era5_path.replace('\\', '/')

                # Accept either FengWu data or ERA5 data (for graceful fallback)
                if not (os.path.exists(fengwu_path) or os.path.exists(era5_path)):
                    # Only log missing data occasionally to reduce noise
                    if hash(time_str) % 100 == 0:  # Use hash for consistent sampling
                        logger.debug(f"Neither FengWu nor ERA5 data found for {time_str}")
                    return False
            else:
                # w/o FengWu mode
                if is_future:
                    # For future steps in w/o FengWu mode: no environmental data needed
                    # Skip validation - we don't need future environmental files
                    continue
                else:
                    # For historical steps: always check ERA5 data
                    era5_path = os.path.join(tc_cropped_path, str(year), sid, f"{time_str}/ERA5_data.npy")
                    era5_path = era5_path.replace('\\', '/')

                    if not os.path.exists(era5_path):
                        # Only log missing ERA5 data occasionally to reduce noise
                        if hash(time_str) % 100 == 0:  # Use hash for consistent sampling
                            logger.debug(f"ERA5 data not found: {era5_path}")
                        return False

        return True

    def _normalize_intensity(self, winds: np.ndarray, pres: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize intensity values"""
        winds_norm = (winds - self.intensity_stats['wind_mean']) / self.intensity_stats['wind_std']
        pres_norm = (pres - self.intensity_stats['pres_mean']) / self.intensity_stats['pres_std']
        return winds_norm, pres_norm

    def _normalize_coordinates_relative(self, coords: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
        """Normalize coordinates relative to reference point"""
        relative_coords = coords - reference_point
        lats_norm = relative_coords[:, 0] / self.coord_stats['lat_std']
        lons_norm = relative_coords[:, 1] / self.coord_stats['lon_std']
        return np.column_stack([lats_norm, lons_norm])

    def _denormalize_coordinates(self, coords_norm: np.ndarray, reference_point: np.ndarray = None) -> np.ndarray:
        """Denormalize relative coordinates back to original coordinates with proper longitude wrapping"""
        if reference_point is None:
            reference_point = np.array([0.0, 0.0])  # If no reference point, assume origin

        lats = coords_norm[:, 0] * self.coord_stats['lat_std']
        lons = coords_norm[:, 1] * self.coord_stats['lon_std']
        relative_coords = np.column_stack([lats, lons])

        # Add reference point
        absolute_coords = relative_coords + reference_point

        # CRITICAL: Wrap longitude to [-180, 180] range to handle international dateline
        absolute_coords[:, 1] = self._wrap_longitude(absolute_coords[:, 1])

        # Clamp latitude to valid range [-90, 90]
        absolute_coords[:, 0] = np.clip(absolute_coords[:, 0], -90.0, 90.0)

        return absolute_coords

    def _wrap_longitude(self, lon: np.ndarray) -> np.ndarray:
        """Wrap longitude to [-180, 180] range."""
        return ((lon + 180) % 360) - 180

    def _denormalize_intensity(self, winds_norm: np.ndarray, pres_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Denormalize data"""
        winds = winds_norm * self.intensity_stats['wind_std'] + self.intensity_stats['wind_mean']
        pres = pres_norm * self.intensity_stats['pres_std'] + self.intensity_stats['pres_mean']
        return winds, pres

    def _load_environmental_data(self, data: pd.DataFrame, use_fengwu: bool = True) -> torch.Tensor:
        """
        Load environmental data following unified paradigm:
        - Historical steps: ERA5 reanalysis data
        - Future steps: FengWu forecast data (if enabled, must be precomputed) or None

        Note: FengWu data MUST be precomputed. Real-time inference is not supported.
        """
        tc_cropped_path = self.config['data']['tc_cropped_path']
        env_data = []

        # Determine the boundary between historical and future steps based on data length
        # data contains history_steps + future_steps rows
        num_hist_steps = min(self.history_steps, len(data))

        for idx, (_, row) in enumerate(data.iterrows()):
            year = row['ISO_TIME'].year
            sid = row['SID']
            time_str = row['ISO_TIME'].strftime('%Y-%m-%d %H_%M_%S')
            tc_lat = row['LAT']
            tc_lon = row['LON']

            # Unified paradigm: Historical ERA5 + Future FengWu (or None)
            is_historical_step = idx < num_hist_steps

            if is_historical_step:
                # Historical steps: Always use ERA5 reanalysis data
                # Historical data MUST exist (verified by _check_env_data_exists)
                era5_path = os.path.join(tc_cropped_path, str(year), sid, f"{time_str}/ERA5_data.npy")
                era5_path = era5_path.replace('\\', '/')

                if os.path.exists(era5_path):
                    try:
                        env_field = np.load(era5_path)
                        if env_field.shape != (69, 80, 80):
                            if hash(era5_path) % 200 == 0:  # Only log shape issues occasionally
                                logger.debug(f"Unexpected ERA5 data shape: {env_field.shape} at {era5_path}")
                            # Use zeros as fallback but this should not happen if _check_env_data_exists works
                            env_field = np.zeros((69, 80, 80), dtype=np.float32)
                    except Exception as e:
                        if hash(era5_path) % 200 == 0:  # Only log loading errors occasionally
                            logger.debug(f"Error loading ERA5 data: {e}")
                        env_field = np.zeros((69, 80, 80), dtype=np.float32)
                else:
                    # This should NOT happen if _check_env_data_exists is working correctly
                    if hash(era5_path) % 100 == 0:  # Only log missing historical data occasionally
                        logger.debug(f"Historical ERA5 data not found: {era5_path} - this should have been filtered!")
                    env_field = np.zeros((69, 80, 80), dtype=np.float32)

                # Always append historical data (never skip/continue for historical steps)
                env_field_normalized = self._normalize_environmental_field(env_field)
                env_data.append(torch.tensor(env_field_normalized, dtype=torch.float32))
            else:
                # Future steps: Use FengWu forecast if enabled (MUST be precomputed)
                if not use_fengwu:
                    # w/o FengWu: Don't add future environmental data at all
                    continue

                # CRITICAL: Only load precomputed FengWu data
                # Real-time FengWu inference is NOT supported in this version
                fengwu_path = os.path.join(tc_cropped_path, str(year), sid, f"{time_str}/FengWu_data.npy")
                fengwu_path = fengwu_path.replace('\\', '/')

                if os.path.exists(fengwu_path):
                    try:
                        env_field = np.load(fengwu_path)
                        if env_field.shape != (69, 80, 80):
                            if hash(fengwu_path) % 200 == 0:  # Only log shape issues occasionally
                                logger.debug(f"Unexpected FengWu data shape: {env_field.shape}")
                            env_field = np.zeros((69, 80, 80), dtype=np.float32)
                    except Exception as e:
                        if hash(fengwu_path) % 200 == 0:  # Only log loading errors occasionally
                            logger.debug(f"Error loading precomputed FengWu data: {e}")
                        env_field = np.zeros((69, 80, 80), dtype=np.float32)
                else:
                    # Precomputed FengWu data not found - this should have been filtered by _check_env_data_exists
                    if hash(fengwu_path) % 200 == 0:  # Only log missing precomputed data occasionally
                        logger.debug(f"Precomputed FengWu data not found: {fengwu_path}, using zeros")
                    env_field = np.zeros((69, 80, 80), dtype=np.float32)

                # Append future environmental data (only if FengWu is enabled)
                env_field_normalized = self._normalize_environmental_field(env_field)
                env_data.append(torch.tensor(env_field_normalized, dtype=torch.float32))

        return torch.stack(env_data) if env_data else torch.empty(0, 69, 80, 80)

    def _normalize_environmental_field(self, env_field: np.ndarray) -> np.ndarray:
        """
        Normalize environmental field using per-channel Z-score normalization.

        Uses precomputed channel-wise statistics for 20x faster normalization.
        Falls back to per-sample computation if precomputed stats unavailable.

        Args:
            env_field: Array of shape (69, 80, 80)

        Returns:
            env_field_normalized: Normalized array of same shape
        """
        if self.use_precomputed_env_stats:
            # Fast vectorized normalization using precomputed statistics (~20x faster)
            env_field_normalized = (env_field - self.env_channel_means) / (self.env_channel_stds + 1e-8)
            return env_field_normalized.astype(np.float32)
        else:
            # Fallback to per-sample normalization
            env_field_normalized = np.zeros_like(env_field, dtype=np.float32)

            for i in range(env_field.shape[0]):
                channel = env_field[i]
                if channel.std() > 1e-8:
                    env_field_normalized[i] = (channel - channel.mean()) / channel.std()
                else:
                    env_field_normalized[i] = channel

            return env_field_normalized

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Get coordinate and intensity data
        hist_coords = np.column_stack([seq['hist_data']['LAT'].values, seq['hist_data']['LON'].values])
        future_coords = np.column_stack([seq['future_data']['LAT'].values, seq['future_data']['LON'].values])

        hist_winds = seq['hist_data']['USA_WIND'].values
        future_winds = seq['future_data']['USA_WIND'].values
        hist_pres = seq['hist_data']['USA_PRES'].values
        future_pres = seq['future_data']['USA_PRES'].values

        # Use relative coordinate normalization
        reference_point = hist_coords[0]  # Use historical starting point as reference
        hist_coords_norm = self._normalize_coordinates_relative(hist_coords, reference_point)
        future_coords_norm = self._normalize_coordinates_relative(future_coords, reference_point)

        # Normalize intensity
        hist_winds_norm, hist_pres_norm = self._normalize_intensity(hist_winds, hist_pres)
        future_winds_norm, future_pres_norm = self._normalize_intensity(future_winds, future_pres)

        # Check if FengWu should be used from config
        use_fengwu = self.use_fengwu

        # Load environmental data
        hist_env = self._load_environmental_data(seq['hist_data'], use_fengwu=False)  # Historical always uses ERA5

        if use_fengwu:
            # Load future environmental data (FengWu forecasts or fallback)
            future_env = self._load_environmental_data(seq['future_data'], use_fengwu=use_fengwu)
        else:
            # w/o FengWu: don't load any future environmental data
            future_env = None

        result = {
            'hist_coords': torch.tensor(hist_coords_norm, dtype=torch.float32),
            'hist_winds': torch.tensor(hist_winds_norm, dtype=torch.float32),
            'hist_pres': torch.tensor(hist_pres_norm, dtype=torch.float32),
            'hist_env': hist_env,
            'future_coords': torch.tensor(future_coords_norm, dtype=torch.float32),
            'future_winds': torch.tensor(future_winds_norm, dtype=torch.float32),
            'future_pres': torch.tensor(future_pres_norm, dtype=torch.float32),
            # Save original data for evaluation
            'future_coords_orig': torch.tensor(future_coords, dtype=torch.float32),
            'future_winds_orig': torch.tensor(future_winds, dtype=torch.float32),
            'future_pres_orig': torch.tensor(future_pres, dtype=torch.float32),
            'reference_point': torch.tensor(reference_point, dtype=torch.float32)
        }

        # Only add future_env if FengWu is enabled
        if use_fengwu and future_env is not None:
            result['future_env'] = future_env

        return result

def create_dataloaders(config: Dict, stats_dir: str = 'dataset') -> Tuple:
    """Create data loaders with consistent normalization statistics and validation set."""

    train_dataset = TCDataset(
        config,
        years=config['data']['train_years'],
        split='train',
        stats_dir=stats_dir
    )

    val_dataset = TCDataset(
        config,
        years=config['data']['val_years'],
        split='val',
        stats_dir=stats_dir
    )

    test_dataset = TCDataset(
        config,
        years=config['data']['test_years'],
        split='test',
        stats_dir=stats_dir
    )

    logger.info("=== NORMALIZATION STATISTICS (from precomputed files) ===")
    logger.info(f"  Wind: {train_dataset.intensity_stats['wind_mean']:.2f}±{train_dataset.intensity_stats['wind_std']:.2f} kt")
    logger.info(f"  Pressure: {train_dataset.intensity_stats['pres_mean']:.2f}±{train_dataset.intensity_stats['pres_std']:.2f} hPa")
    logger.info(f"  Latitude std: {train_dataset.coord_stats['lat_std']:.4f}")
    logger.info(f"  Longitude std: {train_dataset.coord_stats['lon_std']:.4f}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        prefetch_factor=config['hardware'].get('prefetch_factor', 2),
        persistent_workers=config['hardware'].get('persistent_workers', True) and config['hardware']['num_workers'] > 0
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        prefetch_factor=config['hardware'].get('prefetch_factor', 2),
        persistent_workers=config['hardware'].get('persistent_workers', True) and config['hardware']['num_workers'] > 0
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        prefetch_factor=config['hardware'].get('prefetch_factor', 2),
        persistent_workers=config['hardware'].get('persistent_workers', True) and config['hardware']['num_workers'] > 0
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def load_config(config_path: str) -> Dict:
    """Load configuration file."""
    if not os.path.isabs(config_path):
        if os.path.exists(config_path):
            pass
        elif os.path.exists(os.path.join('configs', config_path)):
            config_path = os.path.join('configs', config_path)
        else:
            config_path = os.path.join(os.path.dirname(__file__), '..', config_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
