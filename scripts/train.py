#!/usr/bin/env python3
"""
Training script with dataset and coordinate normalization.
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
from tqdm import tqdm
import numpy as np
from typing import Dict

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import modules
from dataset.dataset import create_dataloaders, load_config
from models.denoising_network import PhysDiffModel
from utils.losses import CombinedLoss
from utils.metrics import TCMetricsSimple as TCMetrics

class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience=2, min_delta=0.001, monitor='val_loss', restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        """Check if we should stop training."""
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            # No improvement
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

def setup_logging(config: Dict):
    """Setup logging configuration"""
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, logger, config):
    """Training epoch function with uncertainty-weighted multi-task learning"""
    model.train()
    
    total_loss = 0.0
    total_diffusion_loss = 0.0
    total_coord_loss = 0.0
    total_msw_loss = 0.0
    total_mlsp_loss = 0.0
    num_batches = 0
    
    # Check if uncertainty weighting is enabled
    use_uncertainty_weighting = config['training']['loss_config'].get('use_uncertainty_weighting', False)
    
    # Mixed precision training
    use_amp = config['hardware']['mixed_precision'] and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} Training')

    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        optimizer.zero_grad()
        
        try:
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(
                    future_coords=batch['future_coords'],
                    future_winds=batch['future_winds'],
                    future_pres=batch['future_pres'],
                    hist_coords=batch['hist_coords'],
                    hist_winds=batch['hist_winds'], 
                    hist_pres=batch['hist_pres'],
                    hist_env=batch['hist_env'],
                    future_env=batch.get('future_env')  # Use .get() for optional future_env
                )
                
                predictions = model.output_decoder(outputs['z_0'])
                
                targets = {
                    'coords': batch['future_coords'],
                    'winds': batch['future_winds'],
                    'pres': batch['future_pres']
                }
                
                losses = criterion(
                    diffusion_loss=outputs['diffusion_loss'],
                    predictions=predictions,
                    targets=targets
                )
                
                loss = losses['total_loss']

            if torch.isnan(loss) or torch.isinf(loss):
                if batch_idx % 50 == 0:
                    logger.warning(f"Skipping NaN/Inf loss in batch {batch_idx}: loss={loss.item()}")
                continue
            
            if use_uncertainty_weighting and writer and batch_idx % 100 == 0:
                try:
                    uncertainty_weights = criterion.get_uncertainty_weights()
                    for name, weight in uncertainty_weights.items():
                        writer.add_scalar(f'Uncertainty/{name}', weight,
                                        epoch * len(dataloader) + batch_idx)
                except Exception as e:
                    if batch_idx % 500 == 0:
                        logger.warning(f"Uncertainty weight logging failed: {e}")
            
            def check_gradients_detailed(log_details=False):
                """Check gradients for NaN/Inf values."""
                total_norm = 0.0
                nan_grad_params = []
                large_grad_params = []

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        total_norm += param_norm ** 2

                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            nan_grad_params.append(name)
                        elif param_norm > 10.0 and log_details:
                            large_grad_params.append((name, param_norm))

                total_norm = total_norm ** 0.5

                return {
                    'total_norm': total_norm,
                    'has_nan': len(nan_grad_params) > 0,
                    'nan_params': nan_grad_params,
                    'large_grads': large_grad_params
                }

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                has_nan = False
                if torch.any(torch.stack([torch.any(torch.isnan(p.grad)) for p in model.parameters() if p.grad is not None])):
                    has_nan = True

                if has_nan:
                    grad_info = check_gradients_detailed(log_details=True)
                    if batch_idx % 20 == 0:
                        logger.warning(f"Skipping batch {batch_idx}: NaN/Inf in {grad_info['nan_params'][:3]}")
                    scaler.update()
                    continue

                if batch_idx > 0 and batch_idx % 500 == 0:
                    grad_info = check_gradients_detailed(log_details=False)
                    logger.info(f"Batch {batch_idx}: gradient norm = {grad_info['total_norm']:.4f}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                has_nan = False
                if torch.any(torch.stack([torch.any(torch.isnan(p.grad)) for p in model.parameters() if p.grad is not None])):
                    has_nan = True

                if has_nan:
                    grad_info = check_gradients_detailed(log_details=True)
                    if batch_idx % 20 == 0:
                        logger.warning(f"Skipping batch {batch_idx}: NaN/Inf in {grad_info['nan_params'][:3]}")
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['gradient_clip'])
                if batch_idx > 0 and batch_idx % 500 == 0:
                    logger.info(f"Batch {batch_idx}: gradient norm = {grad_norm:.4f}")

                optimizer.step()
                
            total_loss += loss.item()
            total_diffusion_loss += losses.get('diffusion_loss', torch.tensor(0)).item()
            
            # Track reconstruction losses if present
            if 'coord_loss' in losses:
                total_coord_loss += losses['coord_loss'].item()
            if 'msw_loss' in losses:
                total_msw_loss += losses['msw_loss'].item()
            if 'mlsp_loss' in losses:
                total_mlsp_loss += losses['mlsp_loss'].item()
                
            num_batches += 1
            
            # Update progress bar
            progress_dict = {
                'Loss': f'{loss.item():.4f}',
                'Diff': f'{losses.get("diffusion_loss", torch.tensor(0)).item():.4f}'
            }
            
            # Add reconstruction losses if enabled
            if 'coord_loss' in losses:
                progress_dict['Coord'] = f'{losses["coord_loss"].item():.4f}'
            if 'msw_loss' in losses:
                progress_dict['MSW'] = f'{losses["msw_loss"].item():.4f}'
            if 'mlsp_loss' in losses:
                progress_dict['MLSP'] = f'{losses["mlsp_loss"].item():.4f}'
                
            pbar.set_postfix(progress_dict)

            if writer and batch_idx > 0 and batch_idx % 100 == 0:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/DiffusionLoss', losses.get('diffusion_loss', torch.tensor(0)).item(), global_step)

                if batch_idx % 500 == 0:
                    for name, pred in predictions.items():
                        writer.add_scalar(f'Pred_Range/{name}_min', pred.min().item(), global_step)
                        writer.add_scalar(f'Pred_Range/{name}_max', pred.max().item(), global_step)
                
        except Exception as e:
            logger.error(f"Training error in batch {batch_idx}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Reset scaler state to prevent inconsistency
            if scaler is not None:
                try:
                    scaler.update()  # Reset scaler state
                except:
                    pass
            continue
    
    # Average losses
    if num_batches == 0:
        logger.warning("No valid batches processed!")
        return {
            'total_loss': float('inf'),
            'diffusion_loss': float('inf'),
            'coord_loss': float('inf'),
            'msw_loss': float('inf'),
            'mlsp_loss': float('inf')
        }
    
    results = {
        'total_loss': total_loss / num_batches,
        'diffusion_loss': total_diffusion_loss / num_batches
    }
    
    # Add reconstruction losses if present
    if total_coord_loss > 0:
        results['coord_loss'] = total_coord_loss / num_batches
    if total_msw_loss > 0:
        results['msw_loss'] = total_msw_loss / num_batches
    if total_mlsp_loss > 0:
        results['mlsp_loss'] = total_mlsp_loss / num_batches
    
    # Add uncertainty weights if enabled
    if use_uncertainty_weighting:
        try:
            uncertainty_weights = criterion.get_uncertainty_weights()
            results.update(uncertainty_weights)
        except:
            pass
    
    return results

def evaluate(model, dataloader, criterion, metrics, device, dataset, logger, config):
    """Evaluation function using real physical quantities for metrics"""
    model.eval()
    
    total_loss = 0.0
    all_predictions_physical = []  # Store denormalized physical predictions
    all_targets_physical = []      # Store original physical targets
    num_batches = 0
    
    use_amp = config['hardware']['mixed_precision'] and device.type == 'cuda'
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating (Physical Quantities Only)')
        
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            try:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(
                        future_coords=batch['future_coords'],
                        future_winds=batch['future_winds'],
                        future_pres=batch['future_pres'],
                        hist_coords=batch['hist_coords'],
                        hist_winds=batch['hist_winds'],
                        hist_pres=batch['hist_pres'],
                        hist_env=batch['hist_env'],
                        future_env=batch.get('future_env')
                    )
                    
                    predictions_normalized = model.output_decoder(outputs['z_0'])
                    
                    targets_normalized = {
                        'coords': batch['future_coords'],
                        'winds': batch['future_winds'],
                        'pres': batch['future_pres']
                    }
                    
                    # Compute loss using normalized values (this is correct for training)
                    losses = criterion(
                        diffusion_loss=outputs['diffusion_loss'],
                        predictions=predictions_normalized,
                        targets=targets_normalized
                    )
                
                if not torch.isnan(losses['total_loss']):
                    total_loss += losses['total_loss'].item()
                    num_batches += 1

                batch_size = predictions_normalized['coords'].shape[0]

                for i in range(batch_size):
                    reference_point = batch['reference_point'][i].cpu().numpy()

                    pred_coords_norm = predictions_normalized['coords'][i].cpu().numpy()
                    pred_winds_norm = predictions_normalized['winds'][i].cpu().numpy()
                    pred_pres_norm = predictions_normalized['pres'][i].cpu().numpy()

                    pred_coords_phys = dataset._denormalize_coordinates(pred_coords_norm, reference_point)
                    pred_winds_phys, pred_pres_phys = dataset._denormalize_intensity(pred_winds_norm, pred_pres_norm)

                    all_predictions_physical.append({
                        'coords': pred_coords_phys,
                        'winds': pred_winds_phys,
                        'pres': pred_pres_phys
                    })

                    target_coords_phys = batch['future_coords_orig'][i].cpu().numpy()
                    target_winds_phys = batch['future_winds_orig'][i].cpu().numpy()
                    target_pres_phys = batch['future_pres_orig'][i].cpu().numpy()

                    all_targets_physical.append({
                        'coords': target_coords_phys,
                        'winds': target_winds_phys,
                        'pres': target_pres_phys
                    })
                
            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                continue
    
    if all_predictions_physical:
        combined_pred_coords = np.stack([p['coords'] for p in all_predictions_physical])
        combined_pred_winds = np.stack([p['winds'] for p in all_predictions_physical])
        combined_pred_pres = np.stack([p['pres'] for p in all_predictions_physical])

        combined_target_coords = np.stack([t['coords'] for t in all_targets_physical])
        combined_target_winds = np.stack([t['winds'] for t in all_targets_physical])
        combined_target_pres = np.stack([t['pres'] for t in all_targets_physical])

        predictions_physical = {
            'coords': torch.from_numpy(combined_pred_coords).float(),
            'winds': torch.from_numpy(combined_pred_winds).float(),
            'pres': torch.from_numpy(combined_pred_pres).float()
        }

        targets_physical = {
            'coords': torch.from_numpy(combined_target_coords).float(),
            'winds': torch.from_numpy(combined_target_winds).float(),
            'pres': torch.from_numpy(combined_target_pres).float()
        }

        eval_metrics = metrics.compute_all_metrics(predictions_physical, targets_physical, None)
        eval_metrics['total_loss'] = total_loss / max(num_batches, 1)

        try:
            pred_array = np.concatenate([
                combined_pred_coords,
                combined_pred_pres[..., np.newaxis],
                combined_pred_winds[..., np.newaxis]
            ], axis=-1)

            target_array = np.concatenate([
                combined_target_coords,
                combined_target_pres[..., np.newaxis],
                combined_target_winds[..., np.newaxis]
            ], axis=-1)

            tc_results = metrics.evaluate_tc_predictions(pred_array, target_array)
            eval_metrics.update(tc_results)

        except Exception as e:
            logger.error(f"Professional TC evaluation failed: {e}")

    else:
        eval_metrics = {'total_loss': float('inf')}
        logger.warning("No valid predictions generated during evaluation!")
    
    return eval_metrics

def main():
    parser = argparse.ArgumentParser(description='Train with coordinate normalization')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU device ID to use (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting training with relative coordinate normalization")
    
    # Device setup with GPU ID selection
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        # Use command line argument if provided, otherwise use config
        gpu_id = args.gpu_id if args.gpu_id is not None else config['hardware'].get('gpu_id', 0)

        if gpu_id == -1:
            device = torch.device('cpu')
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

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Create data loaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(config)
    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # Create model
    model = PhysDiffModel(config).to(device)
    
    # Loss and metrics
    criterion = CombinedLoss(config)
    metrics = TCMetrics(config)
    
    # Check if uncertainty weighting is enabled
    use_uncertainty_weighting = config['training']['loss_config'].get('use_uncertainty_weighting', False)
    
    if use_uncertainty_weighting:
        logger.info("Using uncertainty-weighted multi-task learning")
        # Get all learnable parameters including uncertainty weights
        uncertainty_params = list(criterion.uncertainty_loss.parameters())
        logger.info(f"Uncertainty parameters: {len(uncertainty_params)}")
    else:
        logger.info("Using simple loss combination")
    
    # Optimizer - include uncertainty parameters if enabled
    if use_uncertainty_weighting:
        # Optimize both model parameters and uncertainty weights
        all_params = list(model.parameters()) + list(criterion.uncertainty_loss.parameters())
        optimizer = optim.AdamW(
            all_params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training']['min_lr']
    )
    
    # Tensorboard
    writer = None
    if config['logging']['tensorboard']:
        writer = SummaryWriter(log_dir=os.path.join(config['logging']['log_dir'], 'tensorboard'))
    
    # Early stopping
    early_stopping = None
    if 'early_stopping' in config['training']:
        early_stopping_config = config['training']['early_stopping']
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 2),
            min_delta=early_stopping_config.get('min_delta', 0.001),
            monitor=early_stopping_config.get('monitor', 'val_loss'),
            restore_best_weights=early_stopping_config.get('restore_best_weights', True)
        )
        logger.info(f"Early stopping enabled: patience={early_stopping.patience}, "
                   f"min_delta={early_stopping.min_delta}")
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(config['training']['num_epochs']):
        logger.info(f"Epoch {epoch}/{config['training']['num_epochs']}")

        # Training
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, logger, config
        )

        logger.info(f"Train Loss: {train_losses['total_loss']:.4f}, "
                   f"Diffusion: {train_losses['diffusion_loss']:.4f}")

        # Log training metrics to tensorboard
        if writer:
            writer.add_scalar('Epoch/Train_Loss', train_losses['total_loss'], epoch)
            writer.add_scalar('Epoch/Train_DiffusionLoss', train_losses['diffusion_loss'], epoch)

        # Validation - run every epoch
        val_metrics = evaluate(
            model, val_loader, criterion, metrics, device, val_dataset, logger, config
        )

        # Print validation metrics
        metrics.print_metrics(val_metrics, epoch, 'validation')

        # Log validation metrics to tensorboard
        if writer:
            writer.add_scalar('Epoch/Val_Loss', val_metrics['total_loss'], epoch)

        # Test evaluation - run every 5 epochs or every epoch if early stopping is enabled
        test_freq = 1 if early_stopping is not None else 5
        if epoch % test_freq == 0:
            test_metrics = evaluate(
                model, test_loader, criterion, metrics, device, test_dataset, logger, config
            )

            # Print test metrics
            metrics.print_metrics(test_metrics, epoch, 'test')

            # Log test metrics to tensorboard
            if writer:
                writer.add_scalar('Epoch/Test_Loss', test_metrics['total_loss'], epoch)

        # Save best model based on validation loss
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            logger.info(f"New best model with validation loss: {best_val_loss:.4f}")

            checkpoint_path = os.path.join(config['training']['checkpoint_dir'], 'best_model.pth')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)

        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_metrics['total_loss'], model):
                logger.info(f"Early stopping triggered at epoch {epoch} after {early_stopping.patience} epochs without improvement")
                logger.info(f"Best validation loss: {early_stopping.best_loss:.4f}")
                if early_stopping.restore_best_weights:
                    logger.info("Restored best model weights")
                break

        # Update learning rate
        scheduler.step()

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {current_lr:.6f}")
        if writer:
            writer.add_scalar('Epoch/LearningRate', current_lr, epoch)
    
    logger.info("Training completed!")

    # Final evaluation on test set with professional metrics display
    logger.info("Performing final evaluation on test set...")
    print("\n" + "="*80)
    print("                           FINAL EVALUATION")
    print("="*80)

    final_metrics = evaluate(model, test_loader, criterion, metrics, device, test_dataset, logger, config)
    
    logger.info("Training and evaluation completed!")
    
    if writer:
        writer.close()

if __name__ == '__main__':
    main()