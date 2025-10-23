import os
import numpy as np
import onnxruntime as ort
import torch
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FengWuInference:
    """
    FengWu weather prediction model inference wrapper
    Generates future weather fields using autoregressive inference
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize FengWu inference
        
        Args:
            model_path: path to FengWu ONNX model
        """
        if model_path is None:
            # Try to find fengwu.onnx in common locations
            possible_paths = [
                "scripts/fengwu.onnx",
                "fengwu.onnx",
                os.path.join(os.path.dirname(__file__), "../../scripts/fengwu.onnx"),
                os.path.abspath("scripts/fengwu.onnx")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = os.path.abspath(path)
                    break
                    
        if model_path is None:
            raise FileNotFoundError("FengWu model file not found. Please specify model_path or place fengwu.onnx in scripts/ directory")
        
        # Convert to absolute path for reliability
        self.model_path = os.path.abspath(model_path)
        self.session = None
        self.input_names = None
        self.output_names = None
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load FengWu ONNX model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"FengWu model not found at {self.model_path}")
        
        # Set the behavior of onnxruntime (from scripts/inference.py)
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = 1
        
        # Set the behavior of cuda provider
        cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested'}
        
        try:
            # Create inference session with optimized settings
            providers = [('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, sess_options=options, providers=providers)
            
            # Get input and output names
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"FengWu model loaded successfully")
            logger.info(f"Input names: {self.input_names}")
            logger.info(f"Output names: {self.output_names}")
            
        except Exception as e:
            logger.error(f"Failed to load FengWu model: {e}")
            raise
    
    def preprocess_input(self, era5_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Preprocess ERA5 data for FengWu inference (adapted from scripts/inference.py)
        
        Args:
            era5_data: ERA5 data with shape [2, 69, 721, 1440] (2 time steps)
            
        Returns:
            preprocessed: dictionary with preprocessed inputs for FengWu
        """
        if era5_data.shape[0] != 2:
            raise ValueError(f"FengWu requires 2 input time steps, got {era5_data.shape[0]}")
        
        # Load normalization statistics (from scripts/inference.py approach)
        try:
            # Try to load normalization files if available
            data_mean_path = os.path.join(os.path.dirname(self.model_path), "data_mean.npy")
            data_std_path = os.path.join(os.path.dirname(self.model_path), "data_std.npy")
            
            if os.path.exists(data_mean_path) and os.path.exists(data_std_path):
                data_mean = np.load(data_mean_path)[:, np.newaxis, np.newaxis]
                data_std = np.load(data_std_path)[:, np.newaxis, np.newaxis]
                
                # Apply normalization as in scripts/inference.py
                era5_normalized = (era5_data - data_mean) / data_std
            else:
                logger.warning("Normalization files not found, using raw data")
                era5_normalized = era5_data
        except Exception as e:
            logger.warning(f"Failed to load normalization: {e}, using raw data")
            era5_normalized = era5_data
        
        # Ensure float32 type
        era5_normalized = era5_normalized.astype(np.float32)
        
        # FengWu expects 4D input: [2, 69, 721, 1440] (no batch dimension)
        if len(era5_normalized.shape) == 5:
            era5_normalized = era5_normalized[0]  # Remove batch dimension
        
        # Concatenate time steps as expected by FengWu (from scripts/inference.py)
        input_tensor = np.concatenate((era5_normalized[0], era5_normalized[1]), axis=0)[np.newaxis, :, :, :]
        
        # Create input dictionary
        inputs = {
            self.input_names[0]: input_tensor  # [1, 138, 721, 1440]
        }
        
        return inputs
    
    def postprocess_output(self, outputs: List[np.ndarray]) -> np.ndarray:
        """
        Postprocess FengWu outputs (adapted from scripts/inference.py)
        
        Args:
            outputs: raw outputs from FengWu model
            
        Returns:
            processed: processed output with shape [69, 721, 1440]
        """
        # Extract the main output (future weather field)
        output = outputs[0] if isinstance(outputs, list) else outputs
        
        # Apply denormalization if normalization files are available
        try:
            data_mean_path = os.path.join(os.path.dirname(self.model_path), "data_mean.npy")
            data_std_path = os.path.join(os.path.dirname(self.model_path), "data_std.npy")
            
            if os.path.exists(data_mean_path) and os.path.exists(data_std_path):
                data_mean = np.load(data_mean_path)[:, np.newaxis, np.newaxis]
                data_std = np.load(data_std_path)[:, np.newaxis, np.newaxis]
                
                # Extract first 69 channels and denormalize (from scripts/inference.py)
                if len(output.shape) == 4:  # [1, 138, 721, 1440]
                    output_raw = output[0, :69]  # [69, 721, 1440]
                    output_denormalized = (output_raw * data_std) + data_mean
                else:
                    output_denormalized = output
            else:
                # No normalization files, use raw output
                if len(output.shape) == 4:
                    output_denormalized = output[0, :69]  # [69, 721, 1440]
                else:
                    output_denormalized = output
        except Exception as e:
            logger.warning(f"Failed to denormalize output: {e}")
            if len(output.shape) == 4:
                output_denormalized = output[0, :69]  # [69, 721, 1440]
            else:
                output_denormalized = output
        
        return output_denormalized
    
    def single_step_inference(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform single step inference
        
        Args:
            input_data: input ERA5 data [2, 69, 721, 1440] (2 time steps)
            
        Returns:
            prediction: predicted next time step [69, 721, 1440]
        """
        # Preprocess input
        inputs = self.preprocess_input(input_data)
        
        try:
            # Run inference
            outputs = self.session.run(self.output_names, inputs)
            
            # Postprocess output
            prediction = self.postprocess_output(outputs)
            
            return prediction
            
        except Exception as e:
            logger.error(f"FengWu inference failed: {e}")
            raise
    
    def autoregressive_inference(self, 
                                initial_data: np.ndarray, 
                                num_steps: int) -> np.ndarray:
        """
        Perform autoregressive inference to generate multiple future steps
        
        Args:
            initial_data: initial ERA5 data [2, 69, 721, 1440] (2 time steps)
            num_steps: number of future steps to predict
            
        Returns:
            predictions: predicted future steps [num_steps, 69, 721, 1440]
        """
        predictions = []
        
        # Current input: [t-1, t]
        current_input = initial_data.copy()
        
        for step in range(num_steps):
            logger.debug(f"Generating step {step + 1}/{num_steps}")
            
            # Predict next time step
            next_step = self.single_step_inference(current_input)  # [69, 721, 1440]
            predictions.append(next_step)
            
            # Update input for next iteration: [t, t+1]
            current_input = np.stack([current_input[1], next_step], axis=0)
        
        return np.stack(predictions, axis=0)  # [num_steps, 69, 721, 1440]
    
    def crop_tc_region(self, 
                      global_field: np.ndarray, 
                      tc_lat: float, 
                      tc_lon: float,
                      radius: float = 10.0) -> np.ndarray:
        """
        Crop TC region from global field
        
        Args:
            global_field: global weather field [69, 721, 1440]
            tc_lat: TC center latitude
            tc_lon: TC center longitude
            radius: radius in degrees for cropping
            
        Returns:
            cropped: cropped field [69, 80, 80] (assuming 0.25° resolution)
        """
        # ERA5 grid parameters
        lat_resolution = 0.25  # degrees
        lon_resolution = 0.25  # degrees
        
        # Global grid coordinates
        lats = np.arange(90, -90.25, -lat_resolution)  # [90, 89.75, ..., -90]
        lons = np.arange(0, 360, lon_resolution)  # [0, 0.25, ..., 359.75]
        
        # Convert TC position to grid indices
        lat_idx = np.argmin(np.abs(lats - tc_lat))
        
        # Handle longitude wrapping
        if tc_lon < 0:
            tc_lon += 360
        lon_idx = np.argmin(np.abs(lons - tc_lon))
        
        # Calculate crop region (±radius degrees)
        radius_pixels = int(radius / lat_resolution)  # ~40 pixels for 10 degrees
        
        # Calculate crop bounds
        lat_start = max(0, lat_idx - radius_pixels)
        lat_end = min(len(lats), lat_idx + radius_pixels)
        lon_start = lon_idx - radius_pixels
        lon_end = lon_idx + radius_pixels
        
        # Handle longitude wrapping
        if lon_start < 0:
            # Wrap around
            left_part = global_field[:, lat_start:lat_end, lon_start + len(lons):]
            right_part = global_field[:, lat_start:lat_end, :lon_end]
            cropped_field = np.concatenate([left_part, right_part], axis=2)
        elif lon_end > len(lons):
            # Wrap around
            left_part = global_field[:, lat_start:lat_end, lon_start:]
            right_part = global_field[:, lat_start:lat_end, :lon_end - len(lons)]
            cropped_field = np.concatenate([left_part, right_part], axis=2)
        else:
            # Normal case
            cropped_field = global_field[:, lat_start:lat_end, lon_start:lon_end]
        
        # Resize to standard 80x80 if needed
        target_size = 80
        if cropped_field.shape[1] != target_size or cropped_field.shape[2] != target_size:
            # Simple resize by taking every nth pixel or padding
            h, w = cropped_field.shape[1:]
            if h > target_size:
                stride_h = h // target_size
                cropped_field = cropped_field[:, ::stride_h, :][:, :target_size, :]
            if w > target_size:
                stride_w = w // target_size
                cropped_field = cropped_field[:, :, ::stride_w][:, :, :target_size]
            
            # Pad if too small
            if cropped_field.shape[1] < target_size:
                pad_h = target_size - cropped_field.shape[1]
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                cropped_field = np.pad(cropped_field, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode='edge')
            
            if cropped_field.shape[2] < target_size:
                pad_w = target_size - cropped_field.shape[2]
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                cropped_field = np.pad(cropped_field, ((0, 0), (0, 0), (pad_left, pad_right)), mode='edge')
        
        return cropped_field
    
    def generate_tc_forecast(self,
                           historical_era5: List[np.ndarray],
                           tc_trajectory: List[Tuple[float, float]],
                           num_future_steps: int) -> List[np.ndarray]:
        """
        Generate FengWu forecast for TC region
        
        Args:
            historical_era5: list of historical ERA5 global fields
            tc_trajectory: list of (lat, lon) positions for historical times
            num_future_steps: number of future steps to predict
            
        Returns:
            tc_forecasts: list of cropped forecast fields [69, 80, 80]
        """
        if len(historical_era5) < 2:
            raise ValueError("Need at least 2 historical time steps for FengWu")
        
        # Use last 2 time steps as input
        input_data = np.stack(historical_era5[-2:], axis=0)  # [2, 69, 721, 1440]
        
        # Generate global forecasts
        logger.info(f"Generating {num_future_steps} FengWu forecast steps...")
        global_forecasts = self.autoregressive_inference(input_data, num_future_steps)
        
        # Crop TC regions for each forecast step
        tc_forecasts = []
        last_tc_position = tc_trajectory[-1]  # Use last known position
        
        for step in range(num_future_steps):
            # For simplicity, assume TC moves slowly and use last known position
            # In practice, you might want to predict TC movement or use persistence
            tc_lat, tc_lon = last_tc_position
            
            cropped_forecast = self.crop_tc_region(
                global_forecasts[step],
                tc_lat, 
                tc_lon
            )
            
            tc_forecasts.append(cropped_forecast)
        
        logger.info(f"Generated {len(tc_forecasts)} TC forecast fields")
        return tc_forecasts

class FengWuDataProcessor:
    """
    Processor for integrating FengWu forecasts into the training pipeline
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Check if FengWu should be used and model path is specified
        model_path = self.config['model']['fengwu'].get('model_path')
        use_precomputed = self.config['model']['fengwu'].get('use_precomputed', True)
        
        try:
            if model_path is not None and not use_precomputed:
                # Initialize FengWu with specified model path
                self.fengwu = FengWuInference(model_path)
                self.fengwu_available = True
                logger.info(f"FengWu model loaded successfully from {model_path}")
            else:
                # FengWu not configured for runtime inference
                self.fengwu = None
                self.fengwu_available = False
                if use_precomputed:
                    logger.info("FengWu configured for precomputed data only")
                else:
                    logger.info("FengWu model path not specified, FengWu disabled")
        except Exception as e:
            logger.warning(f"Failed to load FengWu model: {e}")
            self.fengwu = None
            self.fengwu_available = False
        
    def load_historical_era5(self, 
                           era5_paths: List[str], 
                           times: List[datetime]) -> List[np.ndarray]:
        """
        Load historical ERA5 data for FengWu input
        
        Args:
            era5_paths: paths to ERA5 data directories
            times: list of datetime objects for required times
            
        Returns:
            era5_data: list of ERA5 global fields
        """
        era5_data = []
        
        for time in times:
            # Format time string as expected by file naming
            time_str = time.strftime('%Y-%m-%d %H_%M_%S')
            
            # Search for file in available paths
            found = False
            for path in era5_paths:
                file_path = os.path.join(path, f"{time_str}.npy")
                if os.path.exists(file_path):
                    data = np.load(file_path)  # [69, 721, 1440]
                    era5_data.append(data)
                    found = True
                    break
            
            if not found:
                logger.warning(f"ERA5 data not found for time {time_str}")
                # Use last available data as fallback
                if era5_data:
                    era5_data.append(era5_data[-1].copy())
                else:
                    raise FileNotFoundError(f"No ERA5 data available for {time_str}")
        
        return era5_data
    
    def generate_future_fengwu_fields(self,
                                    sid: str,
                                    hist_times: List[datetime],
                                    future_times: List[datetime],
                                    tc_positions: List[Tuple[float, float]]) -> List[np.ndarray]:
        """
        Generate future FengWu fields for a specific TC
        
        Args:
            sid: storm ID
            hist_times: historical times
            future_times: future times to predict
            tc_positions: historical TC positions
            
        Returns:
            future_fields: list of future FengWu fields [69, 80, 80]
        """
        try:
            # Load historical ERA5 data
            era5_paths = self.config['data']['era5_paths']
            historical_era5 = self.load_historical_era5(era5_paths, hist_times)
            
            # Generate forecasts
            future_fields = self.fengwu.generate_tc_forecast(
                historical_era5,
                tc_positions,
                len(future_times)
            )
            
            return future_fields
            
        except Exception as e:
            logger.error(f"Failed to generate FengWu forecast for {sid}: {e}")
            # Return dummy data as fallback
            dummy_field = np.zeros((69, 80, 80), dtype=np.float32)
            return [dummy_field.copy() for _ in future_times]
    
    def precompute_fengwu_forecasts(self, 
                                  dataset_samples: List[Dict],
                                  save_dir: str):
        """
        Precompute FengWu forecasts for all dataset samples
        
        Args:
            dataset_samples: list of dataset samples with metadata
            save_dir: directory to save precomputed forecasts
        """
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Precomputing FengWu forecasts for {len(dataset_samples)} samples...")
        
        for i, sample in enumerate(dataset_samples):
            logger.info(f"Processing sample {i+1}/{len(dataset_samples)}: {sample['sid']}")
            
            try:
                # Generate FengWu forecast
                future_fields = self.generate_future_fengwu_fields(
                    sample['sid'],
                    sample['hist_times'],
                    sample['future_times'],
                    sample['tc_positions']
                )
                
                # Save forecast
                save_path = os.path.join(save_dir, f"{sample['sid']}_{i:06d}.npy")
                np.save(save_path, np.stack(future_fields, axis=0))
                
            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")
                continue
        
        logger.info("FengWu forecast precomputation completed!")

# Example usage and testing
def test_fengwu_inference():
    """Test FengWu inference with dummy data"""
    try:
        fengwu = FengWuInference("fengwu_v2.onnx")
        
        # Create dummy input data
        dummy_input = np.random.randn(2, 69, 721, 1440).astype(np.float32)
        
        # Test single step inference
        prediction = fengwu.single_step_inference(dummy_input)
        print(f"Single step prediction shape: {prediction.shape}")
        
        # Test autoregressive inference
        predictions = fengwu.autoregressive_inference(dummy_input, 3)
        print(f"Autoregressive predictions shape: {predictions.shape}")
        
        # Test TC region cropping
        tc_field = fengwu.crop_tc_region(prediction, 25.0, 120.0)
        print(f"TC cropped field shape: {tc_field.shape}")
        
        print("FengWu inference test passed!")
        
    except Exception as e:
        print(f"FengWu inference test failed: {e}")

if __name__ == "__main__":
    test_fengwu_inference()