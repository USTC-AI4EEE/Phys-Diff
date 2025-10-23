import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
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
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MLPLayer(nn.Module):
    """Multi-layer perceptron with ReLU activation"""
    
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(hidden_dims) - 1:  # No activation after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class TCTrajectoryEncoder(nn.Module):
    """
    TC Trajectory Encoder that processes historical TC sequences
    Input: historical TC data (coordinates, wind speed, pressure)
    Output: encoded sequence representations
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.d_model = config['model']['d_model']
        self.d_embedding = config['model']['d_embedding']
        
        # GRU parameters
        gru_hidden = config['model']['tc_encoder']['gru_hidden']
        gru_layers = config['model']['tc_encoder']['gru_layers']
        
        # MLP parameters
        coord_mlp_dims = config['model']['tc_encoder']['coord_mlp_dims']
        mlsp_mlp_dims = config['model']['tc_encoder']['mlsp_mlp_dims'] 
        msw_mlp_dims = config['model']['tc_encoder']['msw_mlp_dims']
        
        # Coordinate embedding (lat, lon) -> high-dim vector
        self.coord_mlp = MLPLayer(
            input_dim=coord_mlp_dims[0],  # 2 for (lat, lon)
            hidden_dims=coord_mlp_dims[1:] + [self.d_embedding]
        )
        
        # MLSP embedding (pressure) -> high-dim vector
        self.mlsp_mlp = MLPLayer(
            input_dim=mlsp_mlp_dims[0],  # 1 for pressure
            hidden_dims=mlsp_mlp_dims[1:] + [self.d_embedding]
        )
        
        # MSW embedding (wind speed) -> high-dim vector
        self.msw_mlp = MLPLayer(
            input_dim=msw_mlp_dims[0],  # 1 for wind speed
            hidden_dims=msw_mlp_dims[1:] + [self.d_embedding]
        )
        
        # Self-attention transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_embedding,
            nhead=config['model']['encoder']['num_heads'],
            dim_feedforward=config['model']['encoder']['d_ff'],
            dropout=config['model']['encoder']['dropout'],
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['model']['encoder']['num_layers']
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=3 * self.d_embedding,  # 3 embeddings concatenated
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=config['model']['encoder']['dropout'] if gru_layers > 1 else 0
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_embedding)
        
        # Output projection to model dimension
        self.output_proj = nn.Linear(gru_hidden, self.d_model)
        
    def forward(self, coords: torch.Tensor, winds: torch.Tensor, pres: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of TC trajectory encoder
        
        Args:
            coords: [B, T, 2] - normalized coordinates (lat, lon)
            winds: [B, T] - normalized wind speeds
            pres: [B, T] - normalized pressures
            
        Returns:
            encoded_sequence: [B, T, D_model] - encoded sequence
            global_context: [B, D_model] - global context vector
        """
        batch_size, seq_len = coords.shape[:2]
        
        # Embed each component
        coord_embed = self.coord_mlp(coords)  # [B, T, D_embedding]
        wind_embed = self.msw_mlp(winds.unsqueeze(-1))  # [B, T, D_embedding]
        pres_embed = self.mlsp_mlp(pres.unsqueeze(-1))  # [B, T, D_embedding]
        
        # Stack embeddings for self-attention
        embeddings = torch.stack([coord_embed, wind_embed, pres_embed], dim=2)  # [B, T, 3, D_embedding]
        embeddings = embeddings.view(batch_size * seq_len, 3, self.d_embedding)  # [B*T, 3, D_embedding]
        
        # Apply transformer encoder for self-attention
        attended_embeddings = self.transformer_encoder(embeddings)  # [B*T, 3, D_embedding]
        
        # Reshape back
        attended_embeddings = attended_embeddings.view(batch_size, seq_len, 3, self.d_embedding)
        
        # Apply residual connection and layer norm
        embeddings_orig = embeddings.view(batch_size, seq_len, 3, self.d_embedding)
        attended_embeddings = self.layer_norm(attended_embeddings + embeddings_orig)
        
        # Global average pooling across the 3 embeddings
        global_context = torch.mean(attended_embeddings, dim=2)  # [B, T, D_embedding]
        
        # Concatenate all embeddings for GRU input
        gru_input = attended_embeddings.view(batch_size, seq_len, -1)  # [B, T, 3*D_embedding]
        
        # Apply GRU for temporal modeling
        gru_output, final_hidden = self.gru(gru_input)  # gru_output: [B, T, gru_hidden], final_hidden: [num_layers, B, gru_hidden]
        
        # Project to model dimension
        encoded_sequence = self.output_proj(gru_output)  # [B, T, D_model]
        
        # Use GRU's final hidden state as global context (this contains the complete sequence memory)
        if self.gru.num_layers == 1:
            # Single layer GRU: final_hidden shape is [1, B, gru_hidden]
            global_context_hidden = final_hidden.squeeze(0)  # [B, gru_hidden]
        else:
            # Multi-layer GRU: use the last layer's hidden state
            global_context_hidden = final_hidden[-1]  # [B, gru_hidden]
        
        # Project GRU's final hidden state to model dimension
        global_context_proj = self.output_proj(global_context_hidden)  # [B, D_model]
        
        return encoded_sequence, global_context_proj

class DiffusionEmbedding(nn.Module):
    """Diffusion time step embedding using sinusoidal position encoding + MLP"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Sinusoidal embedding
        self.half_dim = d_model // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        
        # MLP for processing
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] - time steps
            
        Returns:
            time_embed: [B, d_model] - time embeddings
        """
        device = t.device
        half_dim = self.half_dim
        
        emb = torch.exp(torch.arange(half_dim, device=device) * -self.emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # If d_model is odd, pad with zeros
        if self.d_model % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=device)], dim=-1)
            
        return self.mlp(emb)

class FutureStateEncoder(nn.Module):
    """
    Encoder for future TC states in the diffusion process
    Converts future TC states to high-dimensional embeddings
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.d_model = config['model']['d_model']
        self.d_embedding = config['model']['d_embedding']
        
        # Similar structure to TC trajectory encoder but for future states
        coord_mlp_dims = config['model']['tc_encoder']['coord_mlp_dims']
        mlsp_mlp_dims = config['model']['tc_encoder']['mlsp_mlp_dims']
        msw_mlp_dims = config['model']['tc_encoder']['msw_mlp_dims']
        
        self.coord_mlp = MLPLayer(
            input_dim=coord_mlp_dims[0],
            hidden_dims=coord_mlp_dims[1:] + [self.d_embedding]
        )
        
        self.mlsp_mlp = MLPLayer(
            input_dim=mlsp_mlp_dims[0],
            hidden_dims=mlsp_mlp_dims[1:] + [self.d_embedding]
        )
        
        self.msw_mlp = MLPLayer(
            input_dim=msw_mlp_dims[0],
            hidden_dims=msw_mlp_dims[1:] + [self.d_embedding]
        )
        
        # Transformer encoder for self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_embedding,
            nhead=config['model']['encoder']['num_heads'],
            dim_feedforward=config['model']['encoder']['d_ff'],
            dropout=config['model']['encoder']['dropout'],
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['model']['encoder']['num_layers']
        )
        
        self.layer_norm = nn.LayerNorm(self.d_embedding)
    
    def forward(self, coords: torch.Tensor, winds: torch.Tensor, pres: torch.Tensor) -> torch.Tensor:
        """
        Encode future TC states
        
        Args:
            coords: [B, N, 2] - future coordinates  
            winds: [B, N] - future wind speeds
            pres: [B, N] - future pressures
            
        Returns:
            z_0: [B, N, D_embedding] - encoded future states (z_0 in diffusion process)
        """
        batch_size, seq_len = coords.shape[:2]
        
        # Embed each component
        coord_embed = self.coord_mlp(coords)  # [B, N, D_embedding]
        wind_embed = self.msw_mlp(winds.unsqueeze(-1))  # [B, N, D_embedding]
        pres_embed = self.mlsp_mlp(pres.unsqueeze(-1))  # [B, N, D_embedding]
        
        # Stack embeddings for self-attention
        embeddings = torch.stack([coord_embed, wind_embed, pres_embed], dim=2)  # [B, N, 3, D_embedding]
        embeddings = embeddings.view(batch_size * seq_len, 3, self.d_embedding)
        
        # Apply transformer encoder
        attended_embeddings = self.transformer_encoder(embeddings)  # [B*N, 3, D_embedding]
        
        # Reshape and apply residual + norm
        attended_embeddings = attended_embeddings.view(batch_size, seq_len, 3, self.d_embedding)
        embeddings_orig = embeddings.view(batch_size, seq_len, 3, self.d_embedding)
        attended_embeddings = self.layer_norm(attended_embeddings + embeddings_orig)
        
        # Average pooling to get final representation
        z_0 = torch.mean(attended_embeddings, dim=2)  # [B, N, D_embedding]
        
        return z_0