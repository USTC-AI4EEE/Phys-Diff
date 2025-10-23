import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np
import math

class TCLoss(nn.Module):
    """
    Combined loss function for TC intensity prediction
    Includes coordinate loss (Euclidean distance), wind loss (L2), and pressure loss (L2)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Support both old and new config formats
        if 'loss_weights' in config['training']:
            # Old format
            self.loss_weights = config['training']['loss_weights']
            self.coord_weight = self.loss_weights['coord']
            self.msw_weight = self.loss_weights['msw'] 
            self.mlsp_weight = self.loss_weights['mlsp']
        else:
            # New format - use default weights when using uncertainty weighting
            self.coord_weight = 1.0
            self.msw_weight = 1.0
            self.mlsp_weight = 1.0
        
    def euclidean_distance_loss(self, pred_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distance loss for coordinates
        
        Args:
            pred_coords: [B, N, 2] - predicted coordinates
            target_coords: [B, N, 2] - target coordinates
            
        Returns:
            loss: scalar loss value
        """
        # Compute Euclidean distance for each point
        distances = torch.sqrt(torch.sum((pred_coords - target_coords) ** 2, dim=-1))  # [B, N]
        
        # Average over all points and batches
        loss = torch.mean(distances)
        
        return loss
    
    def l2_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 loss
        
        Args:
            pred: [B, N] - predictions
            target: [B, N] - targets
            
        Returns:
            loss: scalar loss value
        """
        return F.mse_loss(pred, target)
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute combined TC loss
        
        Args:
            predictions: Dict with 'coords', 'winds', 'pres' keys
            targets: Dict with 'coords', 'winds', 'pres' keys
            
        Returns:
            losses: Dict containing individual and total losses
        """
        # Individual losses
        coord_loss = self.euclidean_distance_loss(predictions['coords'], targets['coords'])
        wind_loss = self.l2_loss(predictions['winds'], targets['winds'])
        pres_loss = self.l2_loss(predictions['pres'], targets['pres'])
        
        # Weighted total loss
        total_loss = (self.coord_weight * coord_loss + 
                     self.msw_weight * wind_loss + 
                     self.mlsp_weight * pres_loss)
        
        return {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'wind_loss': wind_loss,
            'pres_loss': pres_loss
        }

class PhysicsConstraintLoss(nn.Module):
    """
    Additional physics-based constraint losses
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
    def intensity_consistency_loss(self, winds: torch.Tensor, pressures: torch.Tensor) -> torch.Tensor:
        """
        Enforce consistency between wind speed and pressure based on empirical TC relationships
        """
        # Simplified wind-pressure relationship (can be made more sophisticated)
        # Higher winds should generally correlate with lower pressure
        
        # Normalize to [0, 1] for correlation calculation
        winds_norm = (winds - winds.min()) / (winds.max() - winds.min() + 1e-8)
        pres_norm = (pressures - pressures.min()) / (pressures.max() - pressures.min() + 1e-8)
        
        # We want negative correlation: as wind increases, pressure should decrease
        correlation = torch.corrcoef(torch.stack([winds_norm.flatten(), pres_norm.flatten()]))[0, 1]
        
        # Loss is higher when correlation is positive (inconsistent with physics)
        loss = torch.clamp(correlation, min=0.0)
        
        return loss
    
    def trajectory_smoothness_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encourage smooth trajectory changes (physical constraint)
        """
        if coords.shape[1] < 2:
            return torch.tensor(0.0, device=coords.device)
        
        # Compute differences between consecutive time steps
        velocity = coords[:, 1:] - coords[:, :-1]  # [B, N-1, 2]
        
        # Compute acceleration (second derivative)
        if velocity.shape[1] < 2:
            return torch.tensor(0.0, device=coords.device)
            
        acceleration = velocity[:, 1:] - velocity[:, :-1]  # [B, N-2, 2]
        
        # L2 penalty on large accelerations
        smoothness_loss = torch.mean(torch.sum(acceleration ** 2, dim=-1))
        
        return smoothness_loss
    
    def forward(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute physics constraint losses
        
        Args:
            predictions: Dict with 'coords', 'winds', 'pres' keys
            
        Returns:
            losses: Dict with constraint losses
        """
        intensity_loss = self.intensity_consistency_loss(predictions['winds'], predictions['pres'])
        smoothness_loss = self.trajectory_smoothness_loss(predictions['coords'])
        
        return {
            'intensity_consistency': intensity_loss,
            'trajectory_smoothness': smoothness_loss
        }

class CombinedLoss(nn.Module):
    """
    Combined loss with uncertainty-based adaptive weighting.
    Includes: diffusion loss + reconstruction losses (coord, wind, pressure)
    Uses learnable uncertainty parameters to automatically balance multiple tasks.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.tc_loss = TCLoss(config)
        
        # Initialize uncertainty-weighted loss
        self.uncertainty_loss = UncertaintyWeightedLoss(config)
        
        # Get loss configuration
        self.loss_config = config['training']['loss_config']
        self.use_uncertainty_weighting = self.loss_config.get('use_uncertainty_weighting', False)
        self.reconstruction_enabled = self.loss_config.get('reconstruction_loss', {}).get('enabled', True)
        
        # Loss function types
        recon_config = self.loss_config.get('reconstruction_loss', {})
        self.coord_loss_type = recon_config.get('coord_loss_type', 'euclidean')
        self.intensity_loss_type = recon_config.get('intensity_loss_type', 'mse')
        
    def get_uncertainty_weights(self) -> Dict[str, float]:
        """Get current uncertainty-based weights for logging."""
        return self.uncertainty_loss.get_uncertainty_weights()
    
    def compute_coordinate_loss(self, pred_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        """Compute coordinate loss with different types."""
        if self.coord_loss_type == 'euclidean':
            return self.tc_loss.euclidean_distance_loss(pred_coords, target_coords)
        elif self.coord_loss_type == 'huber':
            return F.huber_loss(pred_coords, target_coords, reduction='mean', delta=1.0)
        elif self.coord_loss_type == 'smooth_l1':
            return F.smooth_l1_loss(pred_coords, target_coords, reduction='mean')
        else:
            return self.tc_loss.euclidean_distance_loss(pred_coords, target_coords)
    
    def compute_intensity_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute intensity loss (wind/pressure) with different types."""
        if self.intensity_loss_type == 'mse':
            return F.mse_loss(pred, target)
        elif self.intensity_loss_type == 'mae':
            return F.l1_loss(pred, target)
        elif self.intensity_loss_type == 'huber':
            return F.huber_loss(pred, target, reduction='mean', delta=1.0)
        else:
            return F.mse_loss(pred, target)
        
    def forward(self,
                diffusion_loss: torch.Tensor,
                predictions: Dict[str, torch.Tensor] = None,
                targets: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss with uncertainty-based adaptive weighting.
        
        Args:
            diffusion_loss: DDPM diffusion loss
            predictions: Dict with 'coords', 'winds', 'pres' keys (optional)
            targets: Dict with 'coords', 'winds', 'pres' keys (optional)
            
        Returns:
            losses: Dict containing all losses and uncertainty weights
        """
        
        if self.reconstruction_enabled and predictions is not None and targets is not None:
            # Compute individual reconstruction losses
            coord_loss = self.compute_coordinate_loss(predictions['coords'], targets['coords'])
            msw_loss = self.compute_intensity_loss(predictions['winds'], targets['winds'])
            mlsp_loss = self.compute_intensity_loss(predictions['pres'], targets['pres'])
            
            if self.use_uncertainty_weighting:
                # Use uncertainty-weighted multi-task loss
                total_loss, loss_dict = self.uncertainty_loss.compute_weighted_loss(
                    diffusion_loss, coord_loss, msw_loss, mlsp_loss
                )
                
                # Add uncertainty weights to output
                uncertainty_weights = self.get_uncertainty_weights()
                loss_dict.update(uncertainty_weights)
                
                return loss_dict
            else:
                # Use simple combination without uncertainty weighting
                total_loss = diffusion_loss + coord_loss + msw_loss + mlsp_loss
                
                return {
                    'total_loss': total_loss,
                    'diffusion_loss': diffusion_loss,
                    'coord_loss': coord_loss,
                    'msw_loss': msw_loss,
                    'mlsp_loss': mlsp_loss,
                    'total_reconstruction_loss': coord_loss + msw_loss + mlsp_loss
                }
        else:
            # Only diffusion loss (reconstruction disabled or data not provided)
            return {
                'total_loss': diffusion_loss,
                'diffusion_loss': diffusion_loss
            }

class UncertaintyWeightedLoss(nn.Module):
    """
    Implementation of Multi-Task Learning Using Uncertainty to Weigh Losses.
    
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    by Alex Kendall, Yarin Gal, Roberto Cipolla (CVPR 2018)
    
    The method learns task-dependent uncertainty parameters to automatically balance multiple losses.
    For regression tasks, the weighted loss is: L = (1/(2*sigma^2)) * loss + (1/2) * log(sigma^2)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        loss_config = config['training']['loss_config']
        
        # Check if uncertainty weighting is enabled
        self.use_uncertainty_weighting = loss_config.get('use_uncertainty_weighting', False)
        
        if self.use_uncertainty_weighting:
            # Initialize learnable log-variance parameters
            # log_var = log(sigma^2), where sigma^2 is the task-dependent uncertainty
            uncertainty_weights = loss_config['uncertainty_weights']
            
            # Learnable parameters for each task (log-variance)
            self.log_var_diffusion = nn.Parameter(
                torch.tensor(uncertainty_weights['diffusion_log_var'], dtype=torch.float32)
            )
            self.log_var_coord = nn.Parameter(
                torch.tensor(uncertainty_weights['coord_log_var'], dtype=torch.float32)
            )
            self.log_var_msw = nn.Parameter(
                torch.tensor(uncertainty_weights['msw_log_var'], dtype=torch.float32)
            )
            self.log_var_mlsp = nn.Parameter(
                torch.tensor(uncertainty_weights['mlsp_log_var'], dtype=torch.float32)
            )
        else:
            # Use fixed weights if uncertainty weighting is disabled
            self.log_var_diffusion = torch.tensor(0.0)  # sigma^2 = 1.0
            self.log_var_coord = torch.tensor(0.0)
            self.log_var_msw = torch.tensor(0.0)
            self.log_var_mlsp = torch.tensor(0.0)
    
    def get_uncertainty_weights(self) -> Dict[str, float]:
        """Get current uncertainty-based weights for logging"""
        if self.use_uncertainty_weighting:
            return {
                'diffusion_weight': (1.0 / (2.0 * torch.exp(self.log_var_diffusion))).item(),
                'coord_weight': (1.0 / (2.0 * torch.exp(self.log_var_coord))).item(),
                'msw_weight': (1.0 / (2.0 * torch.exp(self.log_var_msw))).item(),
                'mlsp_weight': (1.0 / (2.0 * torch.exp(self.log_var_mlsp))).item(),
                'diffusion_log_var': self.log_var_diffusion.item(),
                'coord_log_var': self.log_var_coord.item(),
                'msw_log_var': self.log_var_msw.item(),
                'mlsp_log_var': self.log_var_mlsp.item()
            }
        else:
            return {
                'diffusion_weight': 1.0,
                'coord_weight': 1.0,
                'msw_weight': 1.0,
                'mlsp_weight': 1.0
            }
    
    def compute_weighted_loss(self, 
                            diffusion_loss: torch.Tensor,
                            coord_loss: torch.Tensor, 
                            msw_loss: torch.Tensor,
                            mlsp_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute uncertainty-weighted multi-task loss.
        
        For regression tasks, the loss is:
        L_i = (1/(2*sigma_i^2)) * loss_i + (1/2) * log(sigma_i^2)
        
        Args:
            diffusion_loss: DDPM diffusion loss
            coord_loss: Coordinate reconstruction loss 
            msw_loss: Maximum sustained wind loss
            mlsp_loss: Minimum sea level pressure loss
            
        Returns:
            total_loss: Combined uncertainty-weighted loss
            loss_dict: Dictionary with individual losses and weights
        """
        
        if self.use_uncertainty_weighting:
            # Compute uncertainty-weighted losses using learnable parameters
            # L = (1/(2*sigma^2)) * loss + (1/2) * log(sigma^2)
            
            # Prevent numerical instability by clamping log_var
            log_var_diffusion = torch.clamp(self.log_var_diffusion, min=-10, max=10)
            log_var_coord = torch.clamp(self.log_var_coord, min=-10, max=10)
            log_var_msw = torch.clamp(self.log_var_msw, min=-10, max=10) 
            log_var_mlsp = torch.clamp(self.log_var_mlsp, min=-10, max=10)
            
            # Weighted diffusion loss
            weighted_diffusion_loss = (
                0.5 * torch.exp(-log_var_diffusion) * diffusion_loss + 
                0.5 * log_var_diffusion
            )
            
            # Weighted coordinate loss
            weighted_coord_loss = (
                0.5 * torch.exp(-log_var_coord) * coord_loss + 
                0.5 * log_var_coord
            )
            
            # Weighted MSW loss  
            weighted_msw_loss = (
                0.5 * torch.exp(-log_var_msw) * msw_loss + 
                0.5 * log_var_msw
            )
            
            # Weighted MLSP loss
            weighted_mlsp_loss = (
                0.5 * torch.exp(-log_var_mlsp) * mlsp_loss + 
                0.5 * log_var_mlsp
            )
            
        else:
            # Use equal weights if uncertainty weighting is disabled
            weighted_diffusion_loss = diffusion_loss
            weighted_coord_loss = coord_loss 
            weighted_msw_loss = msw_loss
            weighted_mlsp_loss = mlsp_loss
        
        # Total loss
        total_loss = (weighted_diffusion_loss + weighted_coord_loss + 
                     weighted_msw_loss + weighted_mlsp_loss)
        
        # Return detailed loss information
        loss_dict = {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'coord_loss': coord_loss,
            'msw_loss': msw_loss,
            'mlsp_loss': mlsp_loss,
            'weighted_diffusion_loss': weighted_diffusion_loss,
            'weighted_coord_loss': weighted_coord_loss,
            'weighted_msw_loss': weighted_msw_loss,
            'weighted_mlsp_loss': weighted_mlsp_loss
        }
        
        return total_loss, loss_dict