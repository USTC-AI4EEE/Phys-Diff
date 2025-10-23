import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math
import numpy as np

from networks.tc_encoder import DiffusionEmbedding
from networks.piga import PIGATransformerDecoderLayer

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for processing multi-source context information
    Processes concatenated tokens from: [time_token, hist_tokens, era5_tokens, fengwu_tokens]
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.d_model = config['model']['d_model']
        self.num_layers = config['model']['encoder']['num_layers']
        self.num_heads = config['model']['encoder']['num_heads']
        self.d_ff = config['model']['encoder']['d_ff']
        self.dropout = config['model']['encoder']['dropout']
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, input_sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process multi-source input sequence
        
        Args:
            input_sequence: [B, N, D_model] - concatenated tokens from all sources
            mask: attention mask if needed
            
        Returns:
            memory: [B, N, D_model] - encoded context memory
        """
        # Apply transformer encoder
        memory = self.transformer_encoder(input_sequence, src_key_padding_mask=mask)
        
        # Final layer normalization
        memory = self.layer_norm(memory)
        
        return memory

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder with embedded PIGA modules for Physics-Inspired denoising

    PIGA (Physics-Inspired Gated Attention) is always enabled in this implementation.
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        self.d_model = config['model']['d_model']
        self.num_layers = config['model']['decoder']['num_layers']

        # Stack of PIGA-enhanced decoder layers (always enabled)
        self.layers = nn.ModuleList([
            PIGATransformerDecoderLayer(config) for _ in range(self.num_layers)
        ])

        # Final layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode with Physics-Inspired attention

        Args:
            tgt: [B, T, D_model] - target sequence (diffusion states + time embedding)
            memory: [B, N, D_model] - encoder memory
            tgt_mask: causal mask for target sequence

        Returns:
            output: [B, T, D_model] - decoded features
        """
        output = tgt

        # Apply each PIGA-enhanced decoder layer
        for layer in self.layers:
            output = layer(output, memory, tgt_mask)

        # Final layer normalization
        output = self.layer_norm(output)

        return output

class DenoisingNetwork(nn.Module):
    """
    Complete Transformer Encoder-Decoder denoising network with PIGA modules
    
    Architecture:
    1. Tokenization of all input sources
    2. Transformer Encoder for context encoding
    3. Transformer Decoder with PIGA for Physics-Inspired denoising
    4. Output projection to noise prediction
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.d_model = config['model']['d_model']
        self.d_embedding = config['model']['d_embedding']
        
        # Diffusion time embedding
        self.time_embedding = DiffusionEmbedding(self.d_model)
        
        # Input projections for different sources
        self.state_proj = nn.Conv1d(self.d_embedding, self.d_model, kernel_size=1)  # For z_t
        self.hist_proj = nn.Linear(self.d_model, self.d_model)  # For historical TC tokens
        self.env_proj = nn.Linear(self.d_model, self.d_model)   # For environmental tokens
        
        # Transformer Encoder for context
        self.encoder = TransformerEncoder(config)
        
        # Transformer Decoder with PIGA
        self.decoder = TransformerDecoder(config)
        
        # Output projection to predict noise
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(config['model']['decoder']['dropout']),
            nn.Linear(self.d_model, self.d_embedding)
        )
        
        # Initialize weights properly
        self._init_weights()
        
        # Positional encoding for sequences
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=5000)
    
    def _init_weights(self):
        """Initialize model weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def _tokenize_inputs(self,
                        z_t: torch.Tensor,
                        t: torch.Tensor,
                        hist_tokens: torch.Tensor,
                        era5_tokens: torch.Tensor,
                        fengwu_tokens: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize all input sources into unified token sequences

        Args:
            z_t: [B, T, D_embedding] - noisy diffusion states
            t: [B] - timesteps
            hist_tokens: [B, M, D_model] - historical TC tokens
            era5_tokens: [B, M, num_patches, D_model] - ERA5 environmental tokens
            fengwu_tokens: [B, N, num_patches, D_model] - FengWu environmental tokens (optional)

        Returns:
            encoder_input: [B, N_total, D_model] - concatenated input tokens for encoder
            decoder_input: [B, T, D_model] - input tokens for decoder
        """
        B, T, D_emb = z_t.shape
        B, M, D_model = hist_tokens.shape
        B, M_era, num_patches_era, D_model = era5_tokens.shape

        # Check if FengWu tokens are provided
        use_fengwu_tokens = (fengwu_tokens is not None)

        if use_fengwu_tokens:
            B, N_feng, num_patches_feng, D_model = fengwu_tokens.shape

        # 1. Time token embedding
        time_embed = self.time_embedding(t)  # [B, D_model]
        time_token = time_embed.unsqueeze(1)  # [B, 1, D_model]

        # 2. State tokens (z_t) - project to d_model
        z_t_transposed = z_t.transpose(1, 2)  # [B, D_embedding, T]
        state_tokens = self.state_proj(z_t_transposed).transpose(1, 2)  # [B, T, D_model]

        # 3. Historical TC tokens - already in correct dimension
        hist_tokens_proj = self.hist_proj(hist_tokens)  # [B, M, D_model]

        # 4. ERA5 environmental tokens - flatten patches
        era5_tokens_flat = era5_tokens.view(B, M_era * num_patches_era, D_model)  # [B, M*patches, D_model]
        era5_tokens_proj = self.env_proj(era5_tokens_flat)

        # 5. FengWu environmental tokens - flatten patches (optional)
        if use_fengwu_tokens:
            fengwu_tokens_flat = fengwu_tokens.view(B, N_feng * num_patches_feng, D_model)  # [B, N*patches, D_model]
            fengwu_tokens_proj = self.env_proj(fengwu_tokens_flat)

            # Concatenate all tokens for encoder input (with FengWu)
            encoder_input = torch.cat([
                time_token,           # [B, 1, D_model]
                hist_tokens_proj,     # [B, M, D_model]
                era5_tokens_proj,     # [B, M*patches, D_model]
                fengwu_tokens_proj    # [B, N*patches, D_model]
            ], dim=1)  # [B, N_total, D_model]
        else:
            # Concatenate tokens without FengWu
            encoder_input = torch.cat([
                time_token,           # [B, 1, D_model]
                hist_tokens_proj,     # [B, M, D_model]
                era5_tokens_proj,     # [B, M*patches, D_model]
            ], dim=1)  # [B, N_total_no_fengwu, D_model]

        # Decoder input: state tokens + time embedding
        time_embed_expanded = time_embed.unsqueeze(1).expand(-1, T, -1)  # [B, T, D_model]
        decoder_input = state_tokens + time_embed_expanded  # [B, T, D_model]

        return encoder_input, decoder_input
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for decoder self-attention"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def forward(self,
               z_t: torch.Tensor,
               t: torch.Tensor,
               hist_tokens: torch.Tensor,
               era5_tokens: torch.Tensor, 
               fengwu_tokens: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of denoising network
        
        Args:
            z_t: [B, T, D_embedding] - noisy diffusion states
            t: [B] - diffusion timesteps
            hist_tokens: [B, M, D_model] - historical TC sequence tokens
            era5_tokens: [B, M, num_patches, D_model] - ERA5 environmental tokens
            fengwu_tokens: [B, N, num_patches, D_model] - FengWu environmental tokens
            
        Returns:
            predicted_noise: [B, T, D_embedding] - predicted noise
        """
        B, T, D_emb = z_t.shape
        
        # Tokenize all inputs
        encoder_input, decoder_input = self._tokenize_inputs(
            z_t, t, hist_tokens, era5_tokens, fengwu_tokens
        )
        
        # Add positional encoding to decoder input
        decoder_input = self.pos_encoding(decoder_input)
        
        # Encoder: process context information
        memory = self.encoder(encoder_input)  # [B, N_total, D_model]
        
        # Generate causal mask for decoder
        causal_mask = self._generate_causal_mask(T, z_t.device)
        
        # Decoder: Physics-Inspired denoising with PIGA
        decoded_features = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=causal_mask
        )  # [B, T, D_model]
        
        # Project to noise prediction
        predicted_noise = self.output_proj(decoded_features)  # [B, T, D_embedding]
        
        # Clamp output to prevent extreme values
        predicted_noise = torch.clamp(predicted_noise, min=-10.0, max=10.0)
        
        return predicted_noise

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class PhysDiffModel(nn.Module):
    """
    Complete Physics-constrained DDPM model
    Integrates all components: encoders, DDPM, and denoising network
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Import other components
        from networks.tc_encoder import TCTrajectoryEncoder, FutureStateEncoder
        from networks.env_encoder import EnvironmentalEncoder
        from models.ddpm import DDPMDiffusion
        
        # Component modules
        self.tc_encoder = TCTrajectoryEncoder(config)
        self.future_encoder = FutureStateEncoder(config)
        self.env_encoder = EnvironmentalEncoder(config)
        self.ddpm = DDPMDiffusion(config)
        self.denoising_network = DenoisingNetwork(config)
        
        # Output decoder to convert embeddings back to TC parameters
        self.output_decoder = OutputDecoder(config)
        
        # Initialize parameters properly
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters with Xavier/Kaiming initialization"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform for linear layers
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
                # Kaiming normal for conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # Standard initialization for normalization layers
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                # Normal initialization for embeddings
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def to(self, device):
        """Move model and all components to device"""
        super().to(device)
        
        # Explicitly move DDPM to device
        self.ddpm.to(device)
        
        return self
        
    def encode_context(self,
                      hist_coords: torch.Tensor,
                      hist_winds: torch.Tensor,
                      hist_pres: torch.Tensor,
                      hist_env: torch.Tensor,
                      future_env: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode all context information

        Args:
            hist_coords: [B, M, 2] - historical coordinates
            hist_winds: [B, M] - historical wind speeds
            hist_pres: [B, M] - historical pressures
            hist_env: [B, M, 69, 80, 80] - historical ERA5 fields
            future_env: [B, N, 69, 80, 80] - future FengWu fields (optional)

        Returns:
            hist_tokens: [B, M, D_model] - historical TC tokens
            era5_tokens: [B, M, num_patches, D_model] - ERA5 tokens
            fengwu_tokens: [B, N, num_patches, D_model] - FengWu tokens (None if not provided)
        """
        # Encode historical TC sequence
        hist_sequence, _ = self.tc_encoder(hist_coords, hist_winds, hist_pres)  # [B, M, D_model]

        # Encode environmental fields
        if future_env is not None:
            era5_tokens, fengwu_tokens = self.env_encoder(hist_env, future_env)
        else:
            # Only encode historical ERA5, no future environmental data
            era5_tokens, _ = self.env_encoder(hist_env)
            fengwu_tokens = None

        return hist_sequence, era5_tokens, fengwu_tokens
    
    def forward(self,
               future_coords: torch.Tensor,
               future_winds: torch.Tensor,
               future_pres: torch.Tensor,
               hist_coords: torch.Tensor,
               hist_winds: torch.Tensor,
               hist_pres: torch.Tensor, 
               hist_env: torch.Tensor,
               future_env: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass for training
        
        Args:
            future_coords: [B, N, 2] - future coordinates (ground truth)
            future_winds: [B, N] - future wind speeds (ground truth)
            future_pres: [B, N] - future pressures (ground truth)
            hist_coords: [B, M, 2] - historical coordinates
            hist_winds: [B, M] - historical wind speeds
            hist_pres: [B, M] - historical pressures
            hist_env: [B, M, 69, 80, 80] - historical ERA5 fields
            future_env: [B, N, 69, 80, 80] - future FengWu fields
            
        Returns:
            outputs: Dict containing losses and predictions
        """
        # Encode future states to get z_0
        z_0 = self.future_encoder(future_coords, future_winds, future_pres)  # [B, N, D_embedding]
        
        # Encode context
        hist_tokens, era5_tokens, fengwu_tokens = self.encode_context(
            hist_coords, hist_winds, hist_pres, hist_env, future_env
        )
        
        # Compute diffusion loss
        diffusion_loss = self.ddpm.compute_loss(
            model=lambda z_t, t, context: self.denoising_network(
                z_t, t, hist_tokens, era5_tokens, fengwu_tokens
            ),
            x_0=z_0,
            context=hist_tokens  # Use hist_tokens as primary context
        )
        
        return {
            'diffusion_loss': diffusion_loss,
            'z_0': z_0,
            'hist_tokens': hist_tokens
        }
    
    def sample(self,
              hist_coords: torch.Tensor,
              hist_winds: torch.Tensor,
              hist_pres: torch.Tensor,
              hist_env: torch.Tensor,
              future_env: torch.Tensor = None,
              num_steps: int = 8,
              return_denormalized: bool = False,
              dataset = None,
              reference_points: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Generate future TC predictions
        
        Args:
            hist_coords: [B, M, 2] - historical coordinates
            hist_winds: [B, M] - historical wind speeds  
            hist_pres: [B, M] - historical pressures
            hist_env: [B, M, 69, 80, 80] - historical ERA5 fields
            future_env: [B, N, 69, 80, 80] - future FengWu fields
            num_steps: number of future steps to predict
            return_denormalized: whether to denormalize outputs to original scale
            dataset: dataset object for denormalization (required if return_denormalized=True)
            reference_points: [B, 2] reference points for coordinate denormalization
            
        Returns:
            predictions: Dict containing predicted TC parameters
        """
        batch_size = hist_coords.shape[0]
        
        # Encode context
        hist_tokens, era5_tokens, fengwu_tokens = self.encode_context(
            hist_coords, hist_winds, hist_pres, hist_env, future_env
        )
        
        # Sample from diffusion model
        z_0_pred = self.ddpm.reverse_process(
            model=lambda z_t, t, context: self.denoising_network(
                z_t, t, hist_tokens, era5_tokens, fengwu_tokens
            ),
            batch_size=batch_size,
            context=hist_tokens,
            seq_len=num_steps,
            d_embedding=self.config['model']['d_embedding']
        )  # [B, N, D_embedding]
        
        # Decode embeddings to TC parameters (normalized)
        predictions = self.output_decoder(z_0_pred)
        
        # Denormalize if requested
        if return_denormalized and dataset is not None:
            if hasattr(dataset, '_denormalize_intensity'):
                # Denormalize intensity values
                winds_np = predictions['winds'].cpu().numpy()
                pres_np = predictions['pres'].cpu().numpy()
                winds_denorm, pres_denorm = dataset._denormalize_intensity(winds_np, pres_np)
                predictions['winds'] = torch.from_numpy(winds_denorm)
                predictions['pres'] = torch.from_numpy(pres_denorm)
            
            if hasattr(dataset, '_denormalize_coordinates') and reference_points is not None:
                # Denormalize coordinates with reference points
                coords_np = predictions['coords'].cpu().numpy()
                ref_points_np = reference_points.cpu().numpy()
                coords_denorm = []
                for i in range(batch_size):
                    coords_denorm_i = dataset._denormalize_coordinates(coords_np[i], ref_points_np[i])
                    coords_denorm.append(coords_denorm_i)
                coords_denorm = np.array(coords_denorm)
                predictions['coords'] = torch.from_numpy(coords_denorm)
        
        return predictions

class OutputDecoder(nn.Module):
    """Decode embeddings back to TC parameters (coordinates, wind, pressure)"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.d_embedding = config['model']['d_embedding']
        
        # Separate decoders for each TC parameter
        self.coord_decoder = nn.Sequential(
            nn.Linear(self.d_embedding, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # (lat, lon)
        )
        
        self.wind_decoder = nn.Sequential(
            nn.Linear(self.d_embedding, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # wind speed
        )
        
        self.pres_decoder = nn.Sequential(
            nn.Linear(self.d_embedding, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # pressure
        )
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode embeddings to TC parameters
        
        Args:
            z: [B, N, D_embedding] - encoded states
            
        Returns:
            decoded: Dict with 'coords', 'winds', 'pres' keys
        """
        coords = self.coord_decoder(z)  # [B, N, 2]
        winds = self.wind_decoder(z).squeeze(-1)  # [B, N]
        pres = self.pres_decoder(z).squeeze(-1)  # [B, N]
        
        return {
            'coords': coords,
            'winds': winds, 
            'pres': pres
        }