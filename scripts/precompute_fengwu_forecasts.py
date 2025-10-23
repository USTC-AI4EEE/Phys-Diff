#!/usr/bin/env python3
"""
Precompute FengWu-based future TC environment fields for all TC sequences in the dataset.

This script generates future TC environment fields using FengWu global weather forecasts
and saves them in the same structure as the existing ERA5 data. The generated data replaces
the need for real-time ERA5 data for future time steps, allowing the training pipeline to
use FengWu-predicted TC environments instead of observational data.

Key functionality:
1. Uses FengWu model to generate global weather forecasts
2. Dynamically tracks TC positions in the forecast using vorticity analysis
3. Crops TC environment fields (10° radius) around tracked positions
4. Saves results as FengWu_data.npy files with FengWu marker files
5. Maintains compatibility with existing TC_ERA5_crop directory structure
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import yaml
from scipy import ndimage
from skimage.feature import peak_local_max

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from dataset.dataset import load_config
from utils.fengwu_inference import FengWuInference, FengWuDataProcessor

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class TCDynamicTracker:
    """
    Dynamic TC position tracking and environment field extraction.

    Tracks tropical cyclone position in evolving FengWu forecast field
    using vorticity-based detection.
    """

    def __init__(self,
                 crop_radius_deg: float = 10.0,
                 spatial_resolution: float = 0.25,
                 vorticity_threshold: float = 5.0,
                 search_radius_deg: float = 5.0,
                 era5_var_indices: Optional[Dict[str, int]] = None):
        """
        Args:
            crop_radius_deg: Radius in degrees for cropping TC environment (default 10°)
            spatial_resolution: Grid spatial resolution in degrees (default 0.25°)
            vorticity_threshold: Vorticity threshold for TC detection (10^-5 s^-1)
            search_radius_deg: Search radius around predicted TC position (degrees)
            era5_var_indices: Mapping of ERA5 variable names to channel indices
        """
        self.crop_radius_deg = crop_radius_deg
        self.spatial_resolution = spatial_resolution
        self.vorticity_threshold = vorticity_threshold
        self.search_radius_deg = search_radius_deg
        self.era5_var_indices = era5_var_indices or self._default_era5_indices()

        # Precompute grid sizes for efficiency
        self.crop_size = int(2 * crop_radius_deg / spatial_resolution)  # 80x80 for 10° radius

    @staticmethod
    def _default_era5_indices() -> Dict[str, int]:
        """Default ERA5 variable to channel index mapping"""
        return {
            'u10': 0, 'v10': 1, 't2m': 2, 'msl': 3,
            'z_50': 4, 'q_50': 5, 'u_50': 6, 'v_50': 7, 't_50': 8,
            'z_100': 9, 'q_100': 10, 'u_100': 11, 'v_100': 12, 't_100': 13,
            'z_150': 14, 'q_150': 15, 'u_150': 16, 'v_150': 17, 't_150': 18,
            'z_200': 19, 'q_200': 20, 'u_200': 21, 'v_200': 22, 't_200': 23,
            'z_250': 24, 'q_250': 25, 'u_250': 26, 'v_250': 27, 't_250': 28,
            'z_300': 29, 'q_300': 30, 'u_300': 31, 'v_300': 32, 't_300': 33,
            'z_400': 34, 'q_400': 35, 'u_400': 36, 'v_400': 37, 't_400': 38,
            'z_500': 39, 'q_500': 40, 'u_500': 41, 'v_500': 42, 't_500': 43,
            'z_600': 44, 'q_600': 45, 'u_600': 46, 'v_600': 47, 't_600': 48,
            'z_700': 49, 'q_700': 50, 'u_700': 51, 'v_700': 52, 't_700': 53,
            'z_850': 54, 'q_850': 55, 'u_850': 56, 'v_850': 57, 't_850': 58,
            'z_925': 59, 'q_925': 60, 'u_925': 61, 'v_925': 62, 't_925': 63,
            'z_1000': 64, 'q_1000': 65, 'u_1000': 66, 'v_1000': 67, 't_1000': 68,
        }

    def _lat_lon_to_grid_indices(self, lat: float, lon: float,
                                 grid_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert latitude/longitude to grid indices.

        ERA5 convention: latitude [90, -90], longitude [0, 360]

        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (0 to 360 or -180 to 180)
            grid_shape: Global grid shape (721, 1440)

        Returns:
            (lat_idx, lon_idx) grid indices
        """
        # Convert longitude to [0, 360]
        if lon < 0:
            lon = 360 + lon

        # ERA5 grid: lat from 90 to -90 (N to S), lon from 0 to 360 (E)
        lat_idx = int((90 - lat) / self.spatial_resolution)
        lon_idx = int(lon / self.spatial_resolution)

        # Clamp to valid range
        lat_idx = np.clip(lat_idx, 0, grid_shape[0] - 1)
        lon_idx = np.clip(lon_idx, 0, grid_shape[1] - 1)

        return lat_idx, lon_idx

    def _grid_indices_to_lat_lon(self, lat_idx: int, lon_idx: int) -> Tuple[float, float]:
        """
        Convert grid indices back to latitude/longitude.

        Returns:
            (lat, lon) where lat is -90 to 90, lon is 0 to 360
        """
        lat = 90 - lat_idx * self.spatial_resolution
        lon = lon_idx * self.spatial_resolution
        return lat, lon

    def compute_vorticity_at_850hpa(self, forecast_data: np.ndarray) -> np.ndarray:
        """
        Compute relative vorticity at 850 hPa from forecast data.

        Args:
            forecast_data: ERA5 forecast [69, H, W]

        Returns:
            Vorticity field [H, W] in units of 10^-5 s^-1
        """
        # Extract u and v wind at 850 hPa
        u_850_idx = self.era5_var_indices.get('u_850', 56)
        v_850_idx = self.era5_var_indices.get('v_850', 57)

        u_wind = forecast_data[u_850_idx]  # [H, W]
        v_wind = forecast_data[v_850_idx]  # [H, W]

        # Compute vorticity using finite differences
        # ζ = ∂v/∂x - ∂u/∂y
        dy = self.spatial_resolution * 111.32 * 1000  # meters per degree latitude
        dx = self.spatial_resolution * 111.32 * 1000 * np.cos(np.radians(45))  # approximate for mid-latitudes

        dv_dx = np.gradient(v_wind, axis=1) / dx
        du_dy = np.gradient(u_wind, axis=0) / dy

        vorticity = (dv_dx - du_dy) * 1e5  # Convert to 10^-5 s^-1

        return vorticity

    def find_tc_center(self, forecast_data: np.ndarray,
                      initial_lat: float, initial_lon: float,
                      grid_shape: Tuple[int, int] = (721, 1440)) -> Optional[Tuple[float, float]]:
        """
        Find TC center position in forecast data using vorticity-based detection.

        Args:
            forecast_data: ERA5 forecast [69, H, W]
            initial_lat: Initial/predicted TC latitude
            initial_lon: Initial/predicted TC longitude
            grid_shape: Global grid shape

        Returns:
            (lat, lon) of detected TC center, or None if TC not found
        """
        try:
            # Compute vorticity field
            vorticity = self.compute_vorticity_at_850hpa(forecast_data)

            # Convert initial position to grid indices
            center_lat_idx, center_lon_idx = self._lat_lon_to_grid_indices(
                initial_lat, initial_lon, grid_shape
            )

            # Define search region around predicted position
            search_size = int(self.search_radius_deg / self.spatial_resolution)

            lat_start = max(0, center_lat_idx - search_size)
            lat_end = min(grid_shape[0], center_lat_idx + search_size)
            lon_start = max(0, center_lon_idx - search_size)
            lon_end = min(grid_shape[1], center_lon_idx + search_size)

            # Extract vorticity in search region
            vorticity_region = vorticity[lat_start:lat_end, lon_start:lon_end]

            # Find local extrema (both positive and negative vorticity)
            if vorticity_region.size == 0:
                return None

            # Find peaks in absolute vorticity
            abs_vorticity = np.abs(vorticity_region)
            peaks = peak_local_max(abs_vorticity, min_distance=3, threshold_rel=0.1)

            if len(peaks) == 0:
                # If no clear peak, use global maximum in region
                max_idx = np.unravel_index(
                    np.argmax(abs_vorticity),
                    abs_vorticity.shape
                )
                peaks = [max_idx]

            # Find peak closest to initial position with sufficient vorticity
            best_peak = None
            best_vorticity = 0

            for peak in peaks:
                peak_lat_idx = lat_start + peak[0]
                peak_lon_idx = lon_start + peak[1]
                peak_vorticity = abs(vorticity[peak_lat_idx, peak_lon_idx])

                if peak_vorticity > self.vorticity_threshold:
                    if peak_vorticity > best_vorticity:
                        best_vorticity = peak_vorticity
                        best_peak = (peak_lat_idx, peak_lon_idx)

            if best_peak is None:
                logging.debug(f"No TC center found with vorticity >= {self.vorticity_threshold}")
                return None

            # Convert back to lat/lon
            detected_lat, detected_lon = self._grid_indices_to_lat_lon(
                best_peak[0], best_peak[1]
            )

            logging.debug(f"TC detected at ({detected_lat:.2f}, {detected_lon:.2f}) "
                         f"with vorticity {best_vorticity:.2f}")

            return detected_lat, detected_lon

        except Exception as e:
            logging.warning(f"Error finding TC center: {e}")
            return None

    def crop_tc_environment(self, forecast_data: np.ndarray,
                           tc_lat: float, tc_lon: float,
                           grid_shape: Tuple[int, int] = (721, 1440)) -> Optional[np.ndarray]:
        """
        Crop TC environment field around TC center.

        Args:
            forecast_data: ERA5 forecast [69, H, W]
            tc_lat: TC center latitude
            tc_lon: TC center longitude
            grid_shape: Global grid shape

        Returns:
            Cropped TC environment [69, crop_size, crop_size] or None if out of bounds
        """
        try:
            # Convert TC position to grid indices
            center_lat_idx, center_lon_idx = self._lat_lon_to_grid_indices(
                tc_lat, tc_lon, grid_shape
            )

            # Calculate crop boundaries
            crop_half = self.crop_size // 2

            lat_start = center_lat_idx - crop_half
            lat_end = center_lat_idx + crop_half
            lon_start = center_lon_idx - crop_half
            lon_end = center_lon_idx + crop_half

            # Check if crop is within bounds
            if (lat_start < 0 or lat_end > grid_shape[0] or
                lon_start < 0 or lon_end > grid_shape[1]):
                logging.debug(f"TC position ({tc_lat:.2f}, {tc_lon:.2f}) too close to boundary")
                return None

            # Crop the environment
            cropped = forecast_data[:, lat_start:lat_end, lon_start:lon_end]

            if cropped.shape != (forecast_data.shape[0], self.crop_size, self.crop_size):
                logging.warning(f"Unexpected cropped shape {cropped.shape}")
                return None

            return cropped

        except Exception as e:
            logging.warning(f"Error cropping TC environment: {e}")
            return None


def load_global_era5_data(era5_paths: List[str], timestamp: datetime) -> Optional[np.ndarray]:
    """
    Load global ERA5 data for a specific timestamp

    Args:
        era5_paths: List of paths to search for ERA5 data
        timestamp: Target timestamp

    Returns:
        Global ERA5 data [69, 721, 1440] or None if not found
    """
    time_str = timestamp.strftime('%Y%m%d_%H')

    for era5_path in era5_paths:
        # Try different file naming conventions
        possible_names = [
            f"{time_str}.npy",
            f"era5_{time_str}.npy",
            f"{timestamp.strftime('%Y-%m-%d_%H_%M_%S')}.npy"
        ]

        for filename in possible_names:
            full_path = os.path.join(era5_path, filename)
            if os.path.exists(full_path):
                try:
                    data = np.load(full_path)
                    if data.shape == (69, 721, 1440):
                        return data
                    else:
                        logging.warning(f"Unexpected ERA5 shape {data.shape} in {full_path}")
                except Exception as e:
                    logging.warning(f"Error loading {full_path}: {e}")

    # If no file found, issue warning but do NOT generate dummy data
    logging.warning(f"Global ERA5 data not found for {timestamp}")
    return None

def get_tc_sequences_metadata(config: Dict) -> List[Dict]:
    """
    Extract TC sequence metadata from IBTrACS data
    
    Returns:
        List of sequence metadata dictionaries
    """
    logger = logging.getLogger(__name__)
    
    # Load IBTrACS data
    logger.info("Loading IBTrACS data...")
    ibtracs_path = config['data']['ibtracs_path']
    df = pd.read_csv(ibtracs_path, low_memory=False)
    
    # Clean and filter data
    required_columns = ['SID', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'USA_PRES']
    df = df[required_columns].copy()
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
    df = df.dropna()
    
    # Filter by years and time resolution
    all_years = config['data']['train_years'] + config['data']['test_years']
    df_filtered = df[df['ISO_TIME'].dt.year.isin(all_years)]
    
    # Apply 6-hour time resolution filter
    time_resolution = config['data']['time_resolution']
    if time_resolution == 6:
        valid_hours = [0, 6, 12, 18]
        hour_mask = df_filtered['ISO_TIME'].dt.hour.isin(valid_hours)
        df_filtered = df_filtered[hour_mask].copy()
    
    logger.info(f"Filtered to {len(df_filtered)} records")
    
    # Create sequences
    sequences = []
    history_steps = config['data']['history_steps']
    future_steps = config['data']['future_steps']
    
    grouped = df_filtered.groupby('SID')
    
    for sid, group in tqdm(grouped, desc="Processing storms"):
        group_sorted = group.sort_values('ISO_TIME')
        
        # Create sequences with sliding window
        for i in range(len(group_sorted) - history_steps - future_steps + 1):
            hist_data = group_sorted.iloc[i:i+history_steps]
            future_data = group_sorted.iloc[i+history_steps:i+history_steps+future_steps]
            
            # Check if we have environmental data directory for this storm
            year = future_data.iloc[0]['ISO_TIME'].year
            storm_dir = os.path.join(config['data']['tc_cropped_path'], str(year), sid)
            
            if not os.path.exists(storm_dir):
                continue
                
            sequences.append({
                'sid': sid,
                'sequence_idx': len(sequences),
                'year': year,
                'hist_times': hist_data['ISO_TIME'].tolist(),
                'future_times': future_data['ISO_TIME'].tolist(),
                'hist_positions': list(zip(hist_data['LAT'].values, hist_data['LON'].values)),
                'future_positions': list(zip(future_data['LAT'].values, future_data['LON'].values))
            })
    
    logger.info(f"Created {len(sequences)} sequences for FengWu preprocessing")
    return sequences

def precompute_sequence_fengwu_data(sequence: Dict, config: Dict, fengwu: FengWuInference,
                                   tc_tracker: Optional[TCDynamicTracker] = None) -> bool:
    """
    Precompute FengWu-based future TC environment fields for a single TC sequence with dynamic TC tracking.

    This function:
    1. Loads historical ERA5 data (last 2 timesteps)
    2. Generates FengWu forecasts in an autoregressive manner
    3. Dynamically tracks the TC position in each forecast step using vorticity
    4. Crops the TC environment field (10° radius = 80×80 tensor)
    5. Saves results as FengWu_data.npy with FengWu marker files
    6. Outputs zero tensor if TC disappears or cannot be tracked

    Args:
        sequence: Sequence metadata dictionary
        config: Project configuration
        fengwu: FengWu inference engine
        tc_tracker: TCDynamicTracker instance (created if None)

    Returns:
        Success flag
    """
    if tc_tracker is None:
        tc_tracker = TCDynamicTracker(
            crop_radius_deg=10.0,
            spatial_resolution=0.25,
            vorticity_threshold=5.0,
            search_radius_deg=5.0
        )

    try:
        # Load historical ERA5 data (last 2 time steps for FengWu input)
        era5_paths = config['data']['era5_paths']
        historical_era5 = []

        # Use last 2 historical time steps as input for FengWu
        input_times = sequence['hist_times'][-2:]

        for timestamp in input_times:
            era5_data = load_global_era5_data(era5_paths, timestamp)
            if era5_data is None:
                logging.warning(f"Cannot load ERA5 data for {timestamp}, skipping sequence")
                return False
            historical_era5.append(era5_data)

        if len(historical_era5) < 2:
            logging.warning(f"Insufficient historical data for sequence {sequence['sequence_idx']}")
            return False

        # Stack historical data for FengWu input
        input_data = np.stack(historical_era5, axis=0)  # [2, 69, 721, 1440]

        # Generate FengWu forecasts in autoregressive manner
        num_future_steps = len(sequence['future_times'])
        global_forecasts = fengwu.autoregressive_inference(input_data, num_future_steps)

        # Prepare save directory
        tc_cropped_path = config['data']['tc_cropped_path']

        # Track TC position and crop environment for each future time step
        for step, (future_time, future_pos) in enumerate(zip(sequence['future_times'],
                                                              sequence['future_positions'])):
            tc_lat_gt, tc_lon_gt = future_pos  # Ground truth position from IBTrACS

            try:
                # Get the forecast at this step
                forecast_step = global_forecasts[step]  # [69, 721, 1440]

                # Step 1: Attempt to find TC center dynamically using vorticity
                detected_pos = tc_tracker.find_tc_center(
                    forecast_step,
                    tc_lat_gt,
                    tc_lon_gt
                )

                if detected_pos is not None:
                    tc_lat_detected, tc_lon_detected = detected_pos
                    logging.debug(f"Step {step}: TC tracked at ({tc_lat_detected:.2f}, {tc_lon_detected:.2f})")

                    # Step 2: Crop TC environment around detected position
                    cropped_forecast = tc_tracker.crop_tc_environment(
                        forecast_step,
                        tc_lat_detected,
                        tc_lon_detected
                    )

                    if cropped_forecast is not None:
                        # Successfully cropped TC environment
                        output_data = cropped_forecast
                        logging.debug(f"Step {step}: Successfully cropped TC environment, shape {output_data.shape}")
                    else:
                        # Detected TC is too close to boundary
                        logging.warning(f"Step {step}: Detected TC too close to boundary, using zeros")
                        output_data = np.zeros((69, 80, 80), dtype=np.float32)
                else:
                    # TC not found - use zero tensor
                    logging.warning(f"Step {step}: TC disappeared or cannot be tracked, using zeros")
                    output_data = np.zeros((69, 80, 80), dtype=np.float32)

                # Step 3: Save cropped forecast data as future TC environment
                year = future_time.year
                sid = sequence['sid']
                time_str = future_time.strftime('%Y-%m-%d %H_%M_%S')

                save_dir = os.path.join(tc_cropped_path, str(year), sid, time_str)
                os.makedirs(save_dir, exist_ok=True)

                # Save as FengWu_data.npy to distinguish from ERA5 reanalysis data
                # This represents the future TC environment field generated from FengWu forecast
                fengwu_forecast_path = os.path.join(save_dir, 'FengWu_data.npy')

                # Save FengWu-generated TC environment data (or zeros if TC lost)
                np.save(fengwu_forecast_path, output_data.astype(np.float32))
                logging.debug(f"Saved FengWu-generated TC environment for step {step} to {fengwu_forecast_path}")

                # Also save a marker file to indicate this is FengWu-generated data
                fengwu_marker_path = os.path.join(save_dir, 'FENGWU_GENERATED.txt')
                with open(fengwu_marker_path, 'w') as f:
                    f.write(f"Generated from FengWu forecast at {datetime.now()}\n")
                    f.write(f"Original TC position (IBTrACS): {tc_lat_gt:.3f}, {tc_lon_gt:.3f}\n")
                    if detected_pos is not None:
                        f.write(f"Detected TC position: {tc_lat_detected:.3f}, {tc_lon_detected:.3f}\n")
                    else:
                        f.write("TC not detected - using zero field\n")

            except Exception as e:
                logging.error(f"Error processing step {step}: {e}")
                # Save zero tensor for this step as fallback
                year = future_time.year
                sid = sequence['sid']
                time_str = future_time.strftime('%Y-%m-%d %H_%M_%S')
                save_dir = os.path.join(tc_cropped_path, str(year), sid, time_str)
                os.makedirs(save_dir, exist_ok=True)

                # Save as FengWu_data.npy (zero field due to processing error)
                fengwu_forecast_path = os.path.join(save_dir, 'FengWu_data.npy')
                np.save(fengwu_forecast_path, np.zeros((69, 80, 80), dtype=np.float32))

                # Create error marker file
                error_marker_path = os.path.join(save_dir, 'FENGWU_ERROR.txt')
                with open(error_marker_path, 'w') as f:
                    f.write(f"FengWu processing error at {datetime.now()}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write("Using zero field as fallback\n")

        return True

    except Exception as e:
        logging.error(f"Failed to precompute FengWu data for sequence {sequence['sequence_idx']}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Precompute FengWu forecasts for TC dataset')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--max_sequences', type=int, default=None,
                       help='Maximum number of sequences to process (for testing)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing FengWu data files')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting FengWu forecast precomputation with dynamic TC tracking")

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Initialize FengWu inference
    try:
        fengwu = FengWuInference()
        logger.info("FengWu model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load FengWu model: {e}")
        logger.error("Please ensure fengwu.onnx is available in scripts/ directory")
        sys.exit(1)

    # Initialize TC dynamic tracker (reuse for all sequences)
    tc_tracker = TCDynamicTracker(
        crop_radius_deg=10.0,
        spatial_resolution=0.25,
        vorticity_threshold=5.0,
        search_radius_deg=5.0
    )
    logger.info("TC dynamic tracker initialized")

    # Get TC sequences metadata
    sequences = get_tc_sequences_metadata(config)

    if args.max_sequences:
        sequences = sequences[:args.max_sequences]
        logger.info(f"Limited to {len(sequences)} sequences for testing")

    # Process sequences
    success_count = 0
    skip_count = 0
    failure_count = 0

    for sequence in tqdm(sequences, desc="Precomputing FengWu forecasts with dynamic tracking"):
        # Check if already exists (unless overwrite is specified)
        if not args.overwrite:
            # Check if all FengWu-generated data files already exist for this sequence
            all_exist = True
            for future_time in sequence['future_times']:
                year = future_time.year
                sid = sequence['sid']
                time_str = future_time.strftime('%Y-%m-%d %H_%M_%S')
                fengwu_file = os.path.join(
                    config['data']['tc_cropped_path'],
                    str(year), sid, time_str, 'FengWu_data.npy'
                )
                fengwu_marker = os.path.join(
                    config['data']['tc_cropped_path'],
                    str(year), sid, time_str, 'FENGWU_GENERATED.txt'
                )
                # Only skip if both FengWu_data.npy exists AND it was generated by FengWu
                if not (os.path.exists(fengwu_file) and os.path.exists(fengwu_marker)):
                    all_exist = False
                    break

            if all_exist:
                skip_count += 1
                continue

        # Precompute FengWu data for this sequence with dynamic TC tracking
        if precompute_sequence_fengwu_data(sequence, config, fengwu, tc_tracker):
            success_count += 1
        else:
            failure_count += 1

        # Log progress periodically
        if (success_count + skip_count + failure_count) % 50 == 0:
            logger.info(f"Processed {success_count + skip_count + failure_count} sequences: "
                       f"{success_count} successful, {skip_count} skipped, {failure_count} failed")

    # Final summary
    total_processed = success_count + skip_count + failure_count
    logger.info(f"\nFengWu future TC environment field generation completed!")
    logger.info(f"Total sequences: {len(sequences)}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Skipped (already exist): {skip_count}")
    logger.info(f"Failed: {failure_count}")
    logger.info(f"Overall success rate: {100 * success_count / max(total_processed, 1):.1f}%")
    logger.info(f"\nKey features:")
    logger.info(f"  - Generates future TC environment fields from FengWu global forecasts")
    logger.info(f"  - Dynamic TC position tracking using vorticity-based detection")
    logger.info(f"  - Environment field extraction: 10° radius (80×80 tensor at 0.25° resolution)")
    logger.info(f"  - Saves as FengWu_data.npy with FENGWU_GENERATED.txt marker files")
    logger.info(f"  - Zero tensor output when TC cannot be tracked or disappears")
    logger.info(f"  - No dummy data generation for missing ERA5 files")
    logger.info(f"  - Maintains compatibility with existing TC_ERA5_crop structure")

if __name__ == '__main__':
    main()