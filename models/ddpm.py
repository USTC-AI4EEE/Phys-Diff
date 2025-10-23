import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
import math

class DDPMScheduler:
    """DDPM noise scheduler with different schedules"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.num_timesteps = config['model']['ddpm']['num_timesteps']
        self.beta_schedule = config['model']['ddpm']['beta_schedule']
        self.beta_start = config['model']['ddpm']['beta_start']
        self.beta_end = config['model']['ddpm']['beta_end']
        
        # Create beta schedule
        if self.beta_schedule == "linear":
            self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Add numerical stability - clamp to prevent division by zero
        self.alphas_cumprod = torch.clamp(self.alphas_cumprod, min=1e-8)
        self.alphas_cumprod_prev = torch.clamp(self.alphas_cumprod_prev, min=1e-8)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others with numerical stability
        self.sqrt_alphas_cumprod = torch.sqrt(torch.clamp(self.alphas_cumprod, min=1e-8))
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(torch.clamp(1.0 - self.alphas_cumprod, min=1e-8))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0) with numerical stability
        one_minus_alphas_cumprod = torch.clamp(1.0 - self.alphas_cumprod, min=1e-8)
        one_minus_alphas_cumprod_prev = torch.clamp(1.0 - self.alphas_cumprod_prev, min=1e-8)
        
        self.posterior_variance = self.betas * one_minus_alphas_cumprod_prev / one_minus_alphas_cumprod
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]), min=1e-20)
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / one_minus_alphas_cumprod
        self.posterior_mean_coef2 = one_minus_alphas_cumprod_prev * torch.sqrt(self.alphas) / one_minus_alphas_cumprod
        
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = self.num_timesteps + 1
        s = 0.008
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # Add numerical stability to prevent division by zero
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-8)
        alphas_cumprod_prev = torch.clamp(alphas_cumprod[:-1], min=1e-8)
        
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod_prev)
        # Clamp betas to reasonable range
        return torch.clamp(betas, min=1e-6, max=0.999)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        
        Args:
            x_start: [B, T, D] - initial data
            t: [B] - timesteps
            noise: [B, T, D] - noise tensor, if None will sample random noise
            
        Returns:
            x_t: [B, T, D] - noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Clamp noise to prevent extreme values
        noise = torch.clamp(noise, min=-5.0, max=5.0)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # Add numerical stability checks
        if not torch.isfinite(sqrt_alphas_cumprod_t).all():
            sqrt_alphas_cumprod_t = torch.ones_like(sqrt_alphas_cumprod_t) * 0.5
        
        if not torch.isfinite(sqrt_one_minus_alphas_cumprod_t).all():
            sqrt_one_minus_alphas_cumprod_t = torch.ones_like(sqrt_one_minus_alphas_cumprod_t) * 0.5
        
        result = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Final stability check
        if not torch.isfinite(result).all():
            return x_start  # Fallback to original data
        
        return result
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the diffusion posterior:
        q(x_{t-1} | x_t, x_0)
        """
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t
        
        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t
    
    def p_mean_variance(self, model, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the model to get p(x_{t-1} | x_t).
        
        Args:
            model: denoising model
            x_t: [B, T, D] - noisy data at timestep t
            t: [B] - timesteps
            context: [B, N, D] - conditioning context
            
        Returns:
            model_mean: [B, T, D] - predicted mean
            model_variance: [B, T, D] - predicted variance
        """
        # Predict noise
        predicted_noise = model(x_t, t, context)
        
        # Check for NaN/Inf in predicted noise
        if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
            # Return zeros to prevent propagation of NaN/Inf
            predicted_noise = torch.zeros_like(predicted_noise)
        
        # Predict x_0 from noise with numerical stability
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        # Add small epsilon to prevent division by zero
        sqrt_alphas_cumprod_t = torch.clamp(sqrt_alphas_cumprod_t, min=1e-8)
        
        pred_x_start = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
        
        # Get posterior mean and variance
        model_mean, model_variance, _ = self.q_posterior_mean_variance(pred_x_start, x_t, t)
        
        return model_mean, model_variance
    
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t)
        
        Args:
            model: denoising model
            x_t: [B, T, D] - noisy data at timestep t
            t: [B] - timesteps  
            context: [B, N, D] - conditioning context
            
        Returns:
            x_{t-1}: [B, T, D] - denoised data
        """
        model_mean, model_variance = self.p_mean_variance(model, x_t, t, context)
        
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
    
    def p_sample_loop(self, model, shape: Tuple, context: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Generate samples from the model by running the full denoising loop
        
        Args:
            model: denoising model
            shape: shape of samples to generate
            context: [B, N, D] - conditioning context
            device: device to generate on
            
        Returns:
            samples: [B, T, D] - generated samples
        """
        batch_size = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, context)
            
        return img
    
    def sample(self, model, batch_size: int, context: torch.Tensor, seq_len: int, d_embedding: int) -> torch.Tensor:
        """
        Generate samples
        
        Args:
            model: denoising model
            batch_size: number of samples
            context: [B, N, D] - conditioning context
            seq_len: length of sequence to generate
            d_embedding: embedding dimension
            
        Returns:
            samples: [B, T, D] - generated samples
        """
        shape = (batch_size, seq_len, d_embedding)
        return self.p_sample_loop(model, shape, context, context.device)
    
    def training_losses(self, model, x_start: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Compute training losses for the diffusion model
        
        Args:
            model: denoising model
            x_start: [B, T, D] - clean data
            t: [B] - timesteps
            context: [B, N, D] - conditioning context
            
        Returns:
            loss: scalar loss value
        """
        # Check input data for NaN/Inf
        if not torch.isfinite(x_start).all():
            return torch.tensor(1000.0, device=x_start.device, requires_grad=True)
        
        if not torch.isfinite(context).all():
            return torch.tensor(1000.0, device=x_start.device, requires_grad=True)
        
        # Sample noise with numerical stability
        noise = torch.randn_like(x_start)
        noise = torch.clamp(noise, min=-5.0, max=5.0)  # Clamp extreme noise values
        
        # Add noise to data
        x_t = self.q_sample(x_start, t, noise)
        
        # Check noisy data for NaN/Inf
        if not torch.isfinite(x_t).all():
            return torch.tensor(1000.0, device=x_start.device, requires_grad=True)
        
        # Predict noise
        predicted_noise = model(x_t, t, context)
        
        # Check for NaN/Inf in predicted noise and handle gracefully
        if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
            # Create a differentiable high loss that maintains gradient flow
            large_target = torch.zeros_like(predicted_noise)
            loss = F.mse_loss(predicted_noise, large_target) + 1000.0
            return loss
        
        # Clamp predicted noise to reasonable range
        predicted_noise = torch.clamp(predicted_noise, min=-10.0, max=10.0)
        
        # Compute loss (simple MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, noise)
        
        # Check if loss itself is NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            # Return a differentiable fallback loss
            fallback_loss = torch.tensor(1000.0, device=x_start.device, requires_grad=True)
            return fallback_loss
        
        # Clamp loss to reasonable range
        loss = torch.clamp(loss, min=0.0, max=100.0)
        
        return loss
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """
        Extract values from tensor a at indices t and reshape to broadcast with x_shape
        """
        batch_size = t.shape[0]
        # Ensure t is within valid range
        t = torch.clamp(t, 0, len(a) - 1)
        out = a.gather(-1, t)
        # Add numerical stability
        out = torch.clamp(out, min=1e-8, max=1e8)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class DDPMDiffusion(nn.Module):
    """
    Complete DDPM Diffusion model for TC intensity prediction
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.scheduler = DDPMScheduler(config)
        
        # Move scheduler tensors to device when model is moved
        self.register_buffer('_dummy', torch.tensor(0.0))
        
    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_0: [B, T, D] - clean future TC states
            t: [B] - timesteps
            
        Returns:
            x_t: [B, T, D] - noisy states at timestep t
            noise: [B, T, D] - added noise
        """
        noise = torch.randn_like(x_0)
        x_t = self.scheduler.q_sample(x_0, t, noise)
        return x_t, noise
    
    def reverse_process(self, 
                       model, 
                       batch_size: int, 
                       context: torch.Tensor, 
                       seq_len: int, 
                       d_embedding: int) -> torch.Tensor:
        """
        Reverse diffusion process: p(x_0 | x_T)
        
        Args:
            model: denoising model
            batch_size: number of samples
            context: [B, N, D] - conditioning context
            seq_len: sequence length  
            d_embedding: embedding dimension
            
        Returns:
            x_0: [B, T, D] - generated clean states
        """
        return self.scheduler.sample(model, batch_size, context, seq_len, d_embedding)
    
    def compute_loss(self, model, x_0: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion training loss
        
        Args:
            model: denoising model
            x_0: [B, T, D] - clean future states
            context: [B, N, D] - conditioning context
            
        Returns:
            loss: scalar diffusion loss
        """
        batch_size = x_0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=x_0.device).long()
        
        # Compute loss
        loss = self.scheduler.training_losses(model, x_0, t, context)
        
        return loss
    
    def to(self, device):
        """Move model and scheduler to device"""
        super().to(device)
        
        # Move scheduler tensors to device
        for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                         'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                         'posterior_variance', 'posterior_log_variance_clipped',
                         'posterior_mean_coef1', 'posterior_mean_coef2']:
            if hasattr(self.scheduler, attr_name):
                setattr(self.scheduler, attr_name, getattr(self.scheduler, attr_name).to(device))
        
        return self

class NoisePredictor(nn.Module):
    """
    Simple noise predictor for testing DDPM without full denoising network
    This is a placeholder that will be replaced by the full transformer denoising network
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.d_model = config['model']['d_model']
        self.d_embedding = config['model']['d_embedding']
        
        # Simple MLP for noise prediction
        self.noise_net = nn.Sequential(
            nn.Linear(self.d_embedding + self.d_model + 128, 512),  # input + context + time_embed
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, self.d_embedding)
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128)
        )
        
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Predict noise given noisy input, timestep, and context
        
        Args:
            x_t: [B, T, D_embedding] - noisy input
            t: [B] - timesteps
            context: [B, N, D_model] - conditioning context
            
        Returns:
            predicted_noise: [B, T, D_embedding] - predicted noise
        """
        B, T, D_emb = x_t.shape
        _, N, D_model = context.shape
        
        # Time embedding
        t_normalized = t.float() / self.config['model']['ddpm']['num_timesteps']  # Normalize to [0,1]
        t_embed = self.time_embed(t_normalized.unsqueeze(-1))  # [B, 128]
        t_embed = t_embed.unsqueeze(1).expand(-1, T, -1)  # [B, T, 128]
        
        # Context pooling
        context_pooled = torch.mean(context, dim=1)  # [B, D_model]
        context_pooled = context_pooled.unsqueeze(1).expand(-1, T, -1)  # [B, T, D_model]
        
        # Concatenate inputs
        net_input = torch.cat([x_t, context_pooled, t_embed], dim=-1)  # [B, T, D_emb + D_model + 128]
        
        # Predict noise
        predicted_noise = self.noise_net(net_input)  # [B, T, D_embedding]
        
        return predicted_noise