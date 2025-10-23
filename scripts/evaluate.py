import os
import sys
import yaml
import torch
import numpy as np
import argparse
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import project modules
from dataset.dataset import load_config, create_dataloaders
from models.denoising_network import PhysDiffModel
from utils.losses import CombinedLoss
from utils.metrics import TCMetricsSimple as TCMetrics

def setup_logging(config: Dict):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_model(checkpoint_path: str, config: Dict, device: torch.device) -> PhysDiffModel:
    """Load trained model from checkpoint"""
    model = PhysDiffModel(config).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def generate_predictions(model: PhysDiffModel,
                        dataloader,
                        device: torch.device,
                        logger: logging.Logger) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[Dict]]:
    """Generate predictions for entire dataset"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_metadata = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Generating Predictions')
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            try:
                # Generate predictions using sampling (return denormalized results)
                predictions = model.sample(
                    hist_coords=batch['hist_coords'],
                    hist_winds=batch['hist_winds'],
                    hist_pres=batch['hist_pres'],
                    hist_env=batch['hist_env'],
                    future_env=batch.get('future_env'),  # Use .get() for optional future_env
                    num_steps=batch['future_coords'].shape[1],
                    return_denormalized=True,
                    dataset=dataloader.dataset,
                    reference_points=batch['reference_point']
                )
                
                targets = {
                    'coords': batch['future_coords_orig'],  # Use original (unnormalized) for evaluation
                    'winds': batch['future_winds_orig'],
                    'pres': batch['future_pres_orig']
                }
                
                # Store results
                all_predictions.append({
                    'coords': predictions['coords'].cpu(),
                    'winds': predictions['winds'].cpu(),
                    'pres': predictions['pres'].cpu()
                })
                
                all_targets.append({
                    'coords': targets['coords'].cpu(),
                    'winds': targets['winds'].cpu(),
                    'pres': targets['pres'].cpu()
                })
                
                # Store metadata
                all_metadata.extend([{
                    'sid': sid,
                    'hist_times': hist_times,
                    'future_times': future_times
                } for sid, hist_times, future_times in zip(
                    batch['sid'], batch['hist_times'], batch['future_times']
                )])
                
            except RuntimeError as e:
                logger.error(f"Prediction error: {e}")
                continue
    
    # Concatenate results
    combined_predictions = {
        'coords': torch.cat([p['coords'] for p in all_predictions], dim=0),
        'winds': torch.cat([p['winds'] for p in all_predictions], dim=0),
        'pres': torch.cat([p['pres'] for p in all_predictions], dim=0)
    }
    
    combined_targets = {
        'coords': torch.cat([t['coords'] for t in all_targets], dim=0),
        'winds': torch.cat([t['winds'] for t in all_targets], dim=0),
        'pres': torch.cat([t['pres'] for t in all_targets], dim=0)
    }
    
    
    return combined_predictions, combined_targets, all_metadata

def compare_with_baselines(predictions: Dict[str, torch.Tensor],
                         targets: Dict[str, torch.Tensor],
                         hist_data: Dict[str, torch.Tensor],
                         metrics: TCMetrics,
                         logger: logging.Logger) -> Dict[str, Dict]:
    """Compare model predictions with baseline methods"""
    benchmark = TCBenchmark()
    
    # Convert to numpy for baseline methods
    hist_data_np = {k: v.numpy() for k, v in hist_data.items()}
    targets_np = {k: v.numpy() for k, v in targets.items()}
    num_steps = targets_np['coords'].shape[1]
    
    # Persistence baseline
    persistence_pred = benchmark.persistence_forecast(hist_data_np, num_steps)
    persistence_metrics = metrics.compute_all_metrics(
        {k: torch.from_numpy(v) for k, v in persistence_pred.items()},
        targets,
        None  # Skip denormalization as we're using original scale
    )
    
    # Linear trend baseline
    trend_pred = benchmark.linear_trend_forecast(hist_data_np, num_steps)
    trend_metrics = metrics.compute_all_metrics(
        {k: torch.from_numpy(v) for k, v in trend_pred.items()},
        targets,
        None
    )
    
    # Model metrics
    model_metrics = metrics.compute_all_metrics(predictions, targets, None)
    
    # Compare results
    comparison = {
        'persistence': persistence_metrics,
        'linear_trend': trend_metrics,
        'phys_diff': model_metrics
    }
    
    # Print comparison
    logger.info("=== BASELINE COMPARISON ===")
    for method, metric_dict in comparison.items():
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  Track MAE: {metric_dict['track_mae']:.2f}")
        logger.info(f"  Wind MAE: {metric_dict['wind_mae']:.2f}")
        logger.info(f"  Pressure MAE: {metric_dict['pres_mae']:.2f}")
    
    return comparison

def plot_sample_predictions(predictions: Dict[str, torch.Tensor],
                          targets: Dict[str, torch.Tensor],
                          save_dir: str,
                          num_samples: int = 5):
    """Plot sample predictions vs targets"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy
    pred_coords = predictions['coords'].numpy()
    pred_winds = predictions['winds'].numpy()
    pred_pres = predictions['pres'].numpy()
    
    target_coords = targets['coords'].numpy()
    target_winds = targets['winds'].numpy()
    target_pres = targets['pres'].numpy()
    
    # Plot random samples
    num_samples = min(num_samples, pred_coords.shape[0])
    sample_indices = np.random.choice(pred_coords.shape[0], num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Track plot
        axes[0, 0].plot(target_coords[idx, :, 1], target_coords[idx, :, 0], 'b-o', label='Ground Truth', markersize=4)
        axes[0, 0].plot(pred_coords[idx, :, 1], pred_coords[idx, :, 0], 'r--s', label='Prediction', markersize=4)
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].set_title('TC Track')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Wind speed
        time_steps = np.arange(len(target_winds[idx]))
        axes[0, 1].plot(time_steps, target_winds[idx], 'b-o', label='Ground Truth', markersize=4)
        axes[0, 1].plot(time_steps, pred_winds[idx], 'r--s', label='Prediction', markersize=4)
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Wind Speed (kt)')
        axes[0, 1].set_title('Wind Speed Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Pressure
        axes[1, 0].plot(time_steps, target_pres[idx], 'b-o', label='Ground Truth', markersize=4)
        axes[1, 0].plot(time_steps, pred_pres[idx], 'r--s', label='Prediction', markersize=4)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Pressure (hPa)')
        axes[1, 0].set_title('Pressure Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Intensity (Wind vs Pressure)
        axes[1, 1].scatter(target_pres[idx], target_winds[idx], c='blue', alpha=0.7, label='Ground Truth')
        axes[1, 1].scatter(pred_pres[idx], pred_winds[idx], c='red', alpha=0.7, marker='s', label='Prediction')
        axes[1, 1].set_xlabel('Pressure (hPa)')
        axes[1, 1].set_ylabel('Wind Speed (kt)')
        axes[1, 1].set_title('Wind-Pressure Relationship')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_prediction_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_error_analysis(predictions: Dict[str, torch.Tensor],
                       targets: Dict[str, torch.Tensor],
                       save_dir: str):
    """Plot detailed error analysis"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy
    pred_coords = predictions['coords'].numpy()
    pred_winds = predictions['winds'].numpy()
    pred_pres = predictions['pres'].numpy()
    
    target_coords = targets['coords'].numpy()
    target_winds = targets['winds'].numpy()
    target_pres = targets['pres'].numpy()
    
    # Compute errors
    track_errors = np.sqrt(np.sum((pred_coords - target_coords) ** 2, axis=-1))  # [B, N]
    wind_errors = np.abs(pred_winds - target_winds)
    pres_errors = np.abs(pred_pres - target_pres)
    
    # Error statistics by forecast time
    num_steps = track_errors.shape[1]
    time_steps = np.arange(num_steps)
    
    track_mean = np.mean(track_errors, axis=0)
    track_std = np.std(track_errors, axis=0)
    wind_mean = np.mean(wind_errors, axis=0)
    wind_std = np.std(wind_errors, axis=0)
    pres_mean = np.mean(pres_errors, axis=0)
    pres_std = np.std(pres_errors, axis=0)
    
    # Plot error evolution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Track error
    axes[0].plot(time_steps, track_mean, 'b-', linewidth=2, label='Mean Error')
    axes[0].fill_between(time_steps, track_mean - track_std, track_mean + track_std, 
                        alpha=0.3, color='blue', label='±1 Std')
    axes[0].set_xlabel('Forecast Time Step')
    axes[0].set_ylabel('Track Error (degrees)')
    axes[0].set_title('Track Error Evolution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Wind error
    axes[1].plot(time_steps, wind_mean, 'r-', linewidth=2, label='Mean Error')
    axes[1].fill_between(time_steps, wind_mean - wind_std, wind_mean + wind_std,
                        alpha=0.3, color='red', label='±1 Std')
    axes[1].set_xlabel('Forecast Time Step')
    axes[1].set_ylabel('Wind Speed Error (kt)')
    axes[1].set_title('Wind Speed Error Evolution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Pressure error
    axes[2].plot(time_steps, pres_mean, 'g-', linewidth=2, label='Mean Error')
    axes[2].fill_between(time_steps, pres_mean - pres_std, pres_mean + pres_std,
                        alpha=0.3, color='green', label='±1 Std')
    axes[2].set_xlabel('Forecast Time Step')
    axes[2].set_ylabel('Pressure Error (hPa)')
    axes[2].set_title('Pressure Error Evolution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Error distribution histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(track_errors.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Track Error (degrees)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Track Error Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(wind_errors.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_xlabel('Wind Speed Error (kt)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Wind Speed Error Distribution')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(pres_errors.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[2].set_xlabel('Pressure Error (hPa)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Pressure Error Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_results(predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                metadata: List[Dict],
                comparison: Dict[str, Dict],
                save_dir: str):
    """Save evaluation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save predictions and targets
    torch.save({
        'predictions': predictions,
        'targets': targets,
        'metadata': metadata
    }, os.path.join(save_dir, 'predictions.pth'))
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison).T
    comparison_df.to_csv(os.path.join(save_dir, 'baseline_comparison.csv'))
    
    print(f"Results saved to {save_dir}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Physics-constrained DDPM for TC prediction')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU device ID to use (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting Physics-constrained DDPM evaluation")
    
    # Device setup with GPU ID selection
    if args.no_cuda:
        device = torch.device('cpu')
        logger.info("Using CPU (forced by --no_cuda)")
    else:
        # Use command line argument if provided, otherwise use config
        gpu_id = args.gpu_id if args.gpu_id is not None else config['hardware'].get('gpu_id', 0)
        
        if gpu_id == -1:
            device = torch.device('cpu')
            logger.info("Using CPU (gpu_id=-1)")
        elif torch.cuda.is_available():
            if gpu_id >= torch.cuda.device_count():
                logger.warning(f"GPU {gpu_id} not available, using GPU 0")
                gpu_id = 0
            device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            # Set current CUDA device
            torch.cuda.set_device(gpu_id)
        else:
            device = torch.device('cpu')
            logger.info("CUDA not available, using CPU")
    
    # Create dataloaders with consistent normalization
    logger.info("Loading datasets...")
    _, test_loader, _, test_dataset = create_dataloaders(config)
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, config, device)
    
    # Initialize metrics
    metrics = TCMetrics(config)
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions, targets, metadata = generate_predictions(model, test_loader, device, logger)
    
    logger.info(f"Generated predictions for {predictions['coords'].shape[0]} samples")
    
    # Compute metrics - now both predictions and targets are in original scale
    logger.info("Computing metrics...")
    eval_metrics = metrics.compute_all_metrics(predictions, targets, None)
    metrics.print_metrics(eval_metrics, split='test')
    
    # Compare with baselines (simplified - using last historical data as input)
    logger.info("Comparing with baselines...")
    # Note: For full baseline comparison, we'd need to extract historical data from test_loader
    # This is a simplified version
    
    # Create visualizations
    logger.info("Creating visualizations...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_sample_predictions(predictions, targets, 
                           os.path.join(args.output_dir, 'sample_predictions'))
    
    plot_error_analysis(predictions, targets,
                       os.path.join(args.output_dir, 'error_analysis'))
    
    # Save results
    logger.info("Saving results...")
    save_results(predictions, targets, metadata, {'phys_diff': eval_metrics}, args.output_dir)
    
    logger.info("Evaluation completed!")

if __name__ == '__main__':
    main()