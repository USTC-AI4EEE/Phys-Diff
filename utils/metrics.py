import torch
import numpy as np
from typing import Dict, List, Tuple
import math

class TCMetricsSimple:
    """
    Professional tropical cyclone evaluation metrics
    Displays results in clear, standardized format
    """
    
    def __init__(self, config: Dict):
        self.config = config
    
    def wrap_longitude(self, lon: np.ndarray) -> np.ndarray:
        """
        Wrap longitude to [-180, 180) range
        """
        return ((lon + 180) % 360) - 180
    
    def haversine_distance(self, lat1: np.ndarray, lon1: np.ndarray, 
                          lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """
        Calculate great circle distance between points in kilometers
        Using Haversine formula for accurate spherical distance
        
        Args:
            lat1, lon1, lat2, lon2: coordinates in DEGREES
        Returns:
            distance in kilometers
        """
        # Wrap longitudes to [-180, 180) range
        lon1 = self.wrap_longitude(lon1)
        lon2 = self.wrap_longitude(lon2)
        
        # Convert degrees to radians for trigonometric functions
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip to avoid numerical issues
        
        # Earth radius in kilometers
        R = 6371.0
        return R * c
    
    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAE"""
        return np.mean(np.abs(y_true - y_pred))
    
    def root_mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = None) -> float:
        """
        Calculate MAPE with proper epsilon for physical quantities
        """
        if epsilon is None:
            # Use adaptive epsilon based on the scale of the true values
            epsilon = np.maximum(np.abs(y_true).mean() * 0.01, 1e-6)
        
        denominator = np.abs(y_true) + epsilon
        return np.mean(np.abs((y_true - y_pred) / denominator)) * 100
    
    def symmetric_mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate symmetric MAPE (sMAPE) - more robust for values near zero
        """
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        # Avoid division by zero
        mask = denominator > 1e-8
        smape = np.zeros_like(numerator)
        smape[mask] = numerator[mask] / denominator[mask]
        return np.mean(smape) * 100
    
    def denormalize_predictions(self, 
                              predictions: Dict[str, torch.Tensor],
                              dataset,
                              reference_points: torch.Tensor = None) -> Dict[str, np.ndarray]:
        """
        Denormalize predictions back to original scale
        """
        denormalized = {}
        
        # Denormalize coordinates
        if hasattr(dataset, '_denormalize_coordinates'):
            coords_tensor = predictions['coords'].cpu().numpy() if hasattr(predictions['coords'], 'cpu') else predictions['coords']
            
            # Handle reference points for coordinate denormalization
            if reference_points is not None:
                ref_points_np = reference_points.cpu().numpy() if hasattr(reference_points, 'cpu') else reference_points
                # For batch processing, denormalize each sequence with its reference point
                batch_size, seq_len, _ = coords_tensor.shape
                coords_denorm = []
                for i in range(batch_size):
                    coords_denorm_i = dataset._denormalize_coordinates(coords_tensor[i], ref_points_np[i])
                    coords_denorm.append(coords_denorm_i)
                coords_denorm = np.array(coords_denorm)
            else:
                # Fallback to origin reference point
                coords_denorm = dataset._denormalize_coordinates(coords_tensor)
            
            denormalized['coords'] = coords_denorm
        else:
            denormalized['coords'] = predictions['coords'].cpu().numpy() if hasattr(predictions['coords'], 'cpu') else predictions['coords']
        
        # Denormalize intensity values
        if hasattr(dataset, '_denormalize_intensity'):
            winds_tensor = predictions['winds'].cpu().numpy() if hasattr(predictions['winds'], 'cpu') else predictions['winds']
            pres_tensor = predictions['pres'].cpu().numpy() if hasattr(predictions['pres'], 'cpu') else predictions['pres']
            winds_denorm, pres_denorm = dataset._denormalize_intensity(winds_tensor, pres_tensor)
            denormalized['winds'] = winds_denorm
            denormalized['pres'] = pres_denorm
        else:
            denormalized['winds'] = predictions['winds'].cpu().numpy() if hasattr(predictions['winds'], 'cpu') else predictions['winds']
            denormalized['pres'] = predictions['pres'].cpu().numpy() if hasattr(predictions['pres'], 'cpu') else predictions['pres']
        
        return denormalized
    
    def compute_coordinate_errors(self, 
                                pred_coords: np.ndarray, 
                                target_coords: np.ndarray) -> Dict[str, float]:
        """
        Compute coordinate-specific errors using proper spherical distance
        
        Args:
            pred_coords: [N, 2] predicted coordinates in degrees (lat, lon)
            target_coords: [N, 2] target coordinates in degrees (lat, lon)
        """
        # Ensure we have numpy arrays
        if hasattr(pred_coords, 'cpu'):
            pred_coords = pred_coords.cpu().numpy()
        if hasattr(target_coords, 'cpu'):
            target_coords = target_coords.cpu().numpy()
            
        # Flatten for computation
        pred_flat = pred_coords.reshape(-1, 2)
        target_flat = target_coords.reshape(-1, 2)
        
        # Extract coordinates
        pred_lat, pred_lon = pred_flat[:, 0], pred_flat[:, 1]
        target_lat, target_lon = target_flat[:, 0], target_flat[:, 1]
        
        # Calculate spherical distance errors (km)
        spherical_errors_km = self.haversine_distance(
            target_lat, target_lon, pred_lat, pred_lon
        )
        
        # Individual coordinate errors in degrees
        lat_errors = np.abs(pred_lat - target_lat)
        
        # Longitude errors with proper wrapping
        lon_diff = self.wrap_longitude(pred_lon - target_lon)
        lon_errors = np.abs(lon_diff)
        
        return {
            'track_mae_km': np.mean(spherical_errors_km),
            'track_rmse_km': np.sqrt(np.mean(spherical_errors_km ** 2)),
            'track_max_km': np.max(spherical_errors_km),
            'lat_mae': np.mean(lat_errors),
            'lon_mae': np.mean(lon_errors),
            'spherical_errors_km': spherical_errors_km  # For detailed analysis
        }
    
    def compute_intensity_errors(self,
                               pred_winds: np.ndarray,
                               target_winds: np.ndarray,
                               pred_pres: np.ndarray,
                               target_pres: np.ndarray) -> Dict[str, float]:
        """
        Compute intensity-specific errors on physical quantities
        
        Args:
            pred_winds: predicted wind speeds in kt
            target_winds: target wind speeds in kt
            pred_pres: predicted pressure in hPa
            target_pres: target pressure in hPa
        """
        # Ensure we have numpy arrays
        if hasattr(pred_winds, 'cpu'):
            pred_winds = pred_winds.cpu().numpy()
        if hasattr(target_winds, 'cpu'):
            target_winds = target_winds.cpu().numpy()
        if hasattr(pred_pres, 'cpu'):
            pred_pres = pred_pres.cpu().numpy()
        if hasattr(target_pres, 'cpu'):
            target_pres = target_pres.cpu().numpy()
            
        # Flatten arrays
        pred_winds_flat = pred_winds.flatten()
        target_winds_flat = target_winds.flatten()
        pred_pres_flat = pred_pres.flatten()
        target_pres_flat = target_pres.flatten()
        
        # Wind speed errors (kt)
        wind_mae = self.mean_absolute_error(target_winds_flat, pred_winds_flat)
        wind_rmse = self.root_mean_squared_error(target_winds_flat, pred_winds_flat)
        wind_mape = self.mean_absolute_percentage_error(target_winds_flat, pred_winds_flat)
        wind_smape = self.symmetric_mean_absolute_percentage_error(target_winds_flat, pred_winds_flat)
        wind_max_error = np.max(np.abs(pred_winds_flat - target_winds_flat))
        
        # Pressure errors (hPa)
        pres_mae = self.mean_absolute_error(target_pres_flat, pred_pres_flat)
        pres_rmse = self.root_mean_squared_error(target_pres_flat, pred_pres_flat)
        pres_mape = self.mean_absolute_percentage_error(target_pres_flat, pred_pres_flat)
        pres_smape = self.symmetric_mean_absolute_percentage_error(target_pres_flat, pred_pres_flat)
        pres_max_error = np.max(np.abs(pred_pres_flat - target_pres_flat))
        
        return {
            'wind_mae': wind_mae,
            'wind_rmse': wind_rmse,
            'wind_mape': wind_mape,
            'wind_smape': wind_smape,
            'wind_max_error': wind_max_error,
            'pres_mae': pres_mae,
            'pres_rmse': pres_rmse,
            'pres_mape': pres_mape,
            'pres_smape': pres_smape,
            'pres_max_error': pres_max_error
        }
    
    def compute_all_metrics(self,
                          predictions: Dict[str, torch.Tensor],
                          targets: Dict[str, torch.Tensor],
                          dataset,
                          reference_points: torch.Tensor = None) -> Dict[str, float]:
        """
        Compute all evaluation metrics
        """
        # Denormalize if dataset supports it
        try:
            pred_denorm = self.denormalize_predictions(predictions, dataset, reference_points)
            target_denorm = self.denormalize_predictions(targets, dataset, reference_points)
        except:
            # Fall back to using tensor values directly
            pred_denorm = {k: v.cpu().numpy() for k, v in predictions.items()}
            target_denorm = {k: v.cpu().numpy() for k, v in targets.items()}
        
        # Compute coordinate errors
        coord_errors = self.compute_coordinate_errors(
            pred_denorm['coords'], target_denorm['coords']
        )
        
        # Compute intensity errors
        intensity_errors = self.compute_intensity_errors(
            pred_denorm['winds'], target_denorm['winds'],
            pred_denorm['pres'], target_denorm['pres']
        )
        
        # Combine all metrics
        all_metrics = {
            **coord_errors,
            **intensity_errors
        }
        
        return all_metrics
    
    def print_metrics(self, metrics: Dict, epoch: int = None, split: str = 'eval'):
        """
        Print metrics in a formatted way with correct units
        """
        header = f"=== {split.upper()} METRICS"
        if epoch is not None:
            header += f" (Epoch {epoch})"
        header += " ==="
        
        print(header)
        
        # Track errors in km (spherical distance)
        print(f"Track MAE: {metrics.get('track_mae_km', 0):.2f} km")
        print(f"Track RMSE: {metrics.get('track_rmse_km', 0):.2f} km")
        print(f"Latitude MAE: {metrics.get('lat_mae', 0):.3f} degrees")
        print(f"Longitude MAE: {metrics.get('lon_mae', 0):.3f} degrees")
        
        # Wind speed errors
        print(f"Wind MAE: {metrics.get('wind_mae', 0):.2f} kt")
        print(f"Wind RMSE: {metrics.get('wind_rmse', 0):.2f} kt")
        print(f"Wind MAPE: {metrics.get('wind_mape', 0):.1f}%")
        print(f"Wind sMAPE: {metrics.get('wind_smape', 0):.1f}%")
        
        # Pressure errors
        print(f"Pressure MAE: {metrics.get('pres_mae', 0):.2f} hPa")
        print(f"Pressure RMSE: {metrics.get('pres_rmse', 0):.2f} hPa")
        print(f"Pressure MAPE: {metrics.get('pres_mape', 0):.1f}%")
        print(f"Pressure sMAPE: {metrics.get('pres_smape', 0):.1f}%")
        
        print("=" * len(header))
    
    def evaluate_tc_predictions(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Professional tropical cyclone evaluation with standardized metrics
        
        Args:
            predictions: [batch, time, features] - model predictions
            targets: [batch, time, features] - ground truth
            Features order: [lat, lon, pressure, wind_speed]
        
        Returns:
            Dict with professional TC metrics
        """
        # Extract features (assuming order: lat, lon, pressure, wind)
        pred_lat = predictions[:, :, 0]
        pred_lon = predictions[:, :, 1] 
        pred_pressure = predictions[:, :, 2]
        pred_wind = predictions[:, :, 3]
        
        target_lat = targets[:, :, 0]
        target_lon = targets[:, :, 1]
        target_pressure = targets[:, :, 2]
        target_wind = targets[:, :, 3]
        
        # Flatten for computation
        pred_lat_flat = pred_lat.flatten()
        pred_lon_flat = pred_lon.flatten()
        pred_pressure_flat = pred_pressure.flatten()
        pred_wind_flat = pred_wind.flatten()
        
        target_lat_flat = target_lat.flatten()
        target_lon_flat = target_lon.flatten()
        target_pressure_flat = target_pressure.flatten()
        target_wind_flat = target_wind.flatten()
        
        # Wrap longitudes to [-180, 180) range
        pred_lon_flat = self.wrap_longitude(pred_lon_flat)
        target_lon_flat = self.wrap_longitude(target_lon_flat)
        
        # Calculate trajectory distance errors using proper spherical distance (km)
        track_distances = self.haversine_distance(
            target_lat_flat, target_lon_flat,
            pred_lat_flat, pred_lon_flat
        )
        
        # Calculate pressure errors (hPa)
        pressure_errors = np.abs(pred_pressure_flat - target_pressure_flat)
        
        # Calculate wind speed errors (convert from kt to m/s if needed)
        # Assuming input is in kt, convert to m/s: 1 kt = 0.514444 m/s
        wind_kt_to_ms = 0.514444
        wind_errors_ms = np.abs(pred_wind_flat - target_wind_flat) * wind_kt_to_ms
        wind_errors_kt = np.abs(pred_wind_flat - target_wind_flat)
        
        # Compute statistics
        results = {
            # Trajectory (Distance) - spherical distance error in km
            'trajectory_distance_mae_km': np.mean(track_distances),
            'trajectory_distance_rmse_km': np.sqrt(np.mean(track_distances ** 2)),
            'trajectory_distance_max_km': np.max(track_distances),
            
            # Pressure (MSLP) - absolute error in hPa  
            'pressure_mae_hpa': np.mean(pressure_errors),
            'pressure_rmse_hpa': np.sqrt(np.mean(pressure_errors ** 2)),
            'pressure_max_error_hpa': np.max(pressure_errors),
            
            # Wind Speed (MSW) - absolute error in m/s and kt
            'wind_speed_mae_ms': np.mean(wind_errors_ms),
            'wind_speed_rmse_ms': np.sqrt(np.mean(wind_errors_ms ** 2)),
            'wind_speed_mae_kt': np.mean(wind_errors_kt),
            'wind_speed_rmse_kt': np.sqrt(np.mean(wind_errors_kt ** 2)),
            'wind_speed_max_error_ms': np.max(wind_errors_ms),
            
            # Additional statistics
            'num_predictions': len(track_distances),
            'mean_true_pressure_hpa': np.mean(target_pressure_flat),
            'mean_true_wind_kt': np.mean(target_wind_flat),
            'mean_true_wind_ms': np.mean(target_wind_flat) * wind_kt_to_ms
        }
        
        return results
    
    def print_evaluation_results(self, results: Dict[str, float]):
        """
        Print evaluation results in professional format
        """
        print("\n" + "="*80)
        print("                     TROPICAL CYCLONE PREDICTION EVALUATION")
        print("="*80)
        
        print(f"\n[TRAJECTORY] Distance Error Analysis")
        print(f"   Prediction of typhoon center position (lon, lat)")
        print(f"   Spherical distance error from true position:")
        print(f"   * Mean Absolute Error (MAE):  {results['trajectory_distance_mae_km']:.2f} km")
        print(f"   * Root Mean Square Error:     {results['trajectory_distance_rmse_km']:.2f} km") 
        print(f"   * Maximum Error:              {results['trajectory_distance_max_km']:.2f} km")
        
        print(f"\n[PRESSURE] MSLP (Minimum Sea Level Pressure)")
        print(f"   Prediction of central minimum pressure")
        print(f"   Absolute error from true pressure:")
        print(f"   * Mean Absolute Error (MAE):  {results['pressure_mae_hpa']:.2f} hPa")
        print(f"   * Root Mean Square Error:     {results['pressure_rmse_hpa']:.2f} hPa")
        print(f"   * Maximum Error:              {results['pressure_max_error_hpa']:.2f} hPa")
        print(f"   * Mean True Pressure:         {results['mean_true_pressure_hpa']:.2f} hPa")
        
        print(f"\n[WIND SPEED] MSW (Maximum Sustained Wind)")
        print(f"   Prediction of maximum sustained wind speed")
        print(f"   Absolute error from true wind speed:")
        print(f"   * Mean Absolute Error (MAE):  {results['wind_speed_mae_ms']:.2f} m/s  ({results['wind_speed_mae_kt']:.2f} kt)")
        print(f"   * Root Mean Square Error:     {results['wind_speed_rmse_ms']:.2f} m/s  ({results['wind_speed_rmse_kt']:.2f} kt)")
        print(f"   * Maximum Error:              {results['wind_speed_max_error_ms']:.2f} m/s")
        print(f"   * Mean True Wind Speed:       {results['mean_true_wind_ms']:.2f} m/s  ({results['mean_true_wind_kt']:.2f} kt)")
        
        print(f"\n[SUMMARY] Statistics")
        print(f"   * Total Predictions Evaluated: {results['num_predictions']:,}")
        
        # Performance assessment
        trajectory_quality = "Excellent" if results['trajectory_distance_mae_km'] < 50 else \
                           "Good" if results['trajectory_distance_mae_km'] < 100 else \
                           "Fair" if results['trajectory_distance_mae_km'] < 200 else "Needs Improvement"
        
        pressure_quality = "Excellent" if results['pressure_mae_hpa'] < 5 else \
                          "Good" if results['pressure_mae_hpa'] < 10 else \
                          "Fair" if results['pressure_mae_hpa'] < 20 else "Needs Improvement"
        
        wind_quality = "Excellent" if results['wind_speed_mae_ms'] < 2.5 else \
                      "Good" if results['wind_speed_mae_ms'] < 5 else \
                      "Fair" if results['wind_speed_mae_ms'] < 10 else "Needs Improvement"
        
        print(f"\n[PERFORMANCE ASSESSMENT]")
        print(f"   * Trajectory Prediction:      {trajectory_quality}")
        print(f"   * Pressure Prediction:        {pressure_quality}") 
        print(f"   * Wind Speed Prediction:      {wind_quality}")
        
        print("="*80)