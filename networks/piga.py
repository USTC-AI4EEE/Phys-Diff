import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math

class PIGAModule(nn.Module):
    """
    Physics-Inspired Gated Attention (PIGA) Module
    
    This module implements physical constraints by modeling interactions between
    different physical quantities (coordinates, wind speed, pressure) through
    cross-task attention and gating mechanisms.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.d_model = config['model']['d_model']
        self.d_sub = config['model']['piga']['d_sub']
        self.gate_mlp_dims = config['model']['piga']['gate_mlp_dims']
        
        # Task-specific mappings using 1D convolution for efficiency
        # This maps from [B, D, N] to [B, D_sub, N]
        self.coord_mapping = nn.Conv1d(self.d_model, self.d_sub, kernel_size=1)
        self.msw_mapping = nn.Conv1d(self.d_model, self.d_sub, kernel_size=1)
        self.mslp_mapping = nn.Conv1d(self.d_model, self.d_sub, kernel_size=1)
        
        # Cross-attention components for each task
        # Coordinate task cross-attention
        self.coord_q_proj = nn.Linear(self.d_sub, self.d_sub)
        self.coord_k_proj = nn.Linear(2 * self.d_sub, self.d_sub)  # Other two tasks concatenated
        self.coord_v_proj = nn.Linear(2 * self.d_sub, self.d_sub)
        
        # MSW task cross-attention  
        self.msw_q_proj = nn.Linear(self.d_sub, self.d_sub)
        self.msw_k_proj = nn.Linear(2 * self.d_sub, self.d_sub)
        self.msw_v_proj = nn.Linear(2 * self.d_sub, self.d_sub)
        
        # MSLP task cross-attention
        self.mslp_q_proj = nn.Linear(self.d_sub, self.d_sub)
        self.mslp_k_proj = nn.Linear(2 * self.d_sub, self.d_sub)
        self.mslp_v_proj = nn.Linear(2 * self.d_sub, self.d_sub)
        
        # Gating MLPs for each task
        self.coord_gate_mlp = self._build_gate_mlp()
        self.msw_gate_mlp = self._build_gate_mlp()
        self.mslp_gate_mlp = self._build_gate_mlp()
        
        # Output projection to combine all tasks back to d_model
        self.output_conv = nn.Conv1d(3 * self.d_sub, self.d_model, kernel_size=1)
        
        # Normalization layers
        self.norm_coord = nn.LayerNorm(self.d_sub)
        self.norm_msw = nn.LayerNorm(self.d_sub)
        self.norm_mslp = nn.LayerNorm(self.d_sub)
        
        # Initialize parameters
        self._init_parameters()
    
    def _build_gate_mlp(self) -> nn.Module:
        """Build gating MLP"""
        layers = []
        input_dim = 2 * self.d_sub  # Original feature + attention result
        
        for i, hidden_dim in enumerate(self.gate_mlp_dims):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(self.gate_mlp_dims[i-1], hidden_dim))
            
            if i < len(self.gate_mlp_dims) - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Sigmoid())  # Final sigmoid for gating
        
        return nn.Sequential(*layers)
    
    def _init_parameters(self):
        """Initialize parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _cross_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-attention
        
        Args:
            query: [B, N, D_sub]
            key: [B, N, D_sub] 
            value: [B, N, D_sub]
            
        Returns:
            attention_output: [B, N, D_sub]
        """
        B, N, D = query.shape
        
        # Compute attention scores
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(D)  # [B, N, N]
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attention_output = torch.bmm(attention_weights, value)  # [B, N, D_sub]
        
        return attention_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PIGA module
        
        Args:
            x: [B, N, D_model] - input features from cross-attention layer
            
        Returns:
            output: [B, N, D_model] - Physics-Inspired features
        """
        B, N, D = x.shape
        
        # Transpose for 1D convolution: [B, N, D] -> [B, D, N]
        x_transposed = x.transpose(1, 2)  # [B, D_model, N]
        
        # Task-specific mappings
        f_coord = self.coord_mapping(x_transposed).transpose(1, 2)  # [B, N, D_sub]
        f_msw = self.msw_mapping(x_transposed).transpose(1, 2)      # [B, N, D_sub]
        f_mslp = self.mslp_mapping(x_transposed).transpose(1, 2)    # [B, N, D_sub]
        
        # Cross-task attention and gating updates
        
        # 1. Update coordinate features
        others_coord = torch.cat([f_msw, f_mslp], dim=-1)  # [B, N, 2*D_sub]
        q_coord = self.coord_q_proj(f_coord)               # [B, N, D_sub]
        k_coord = self.coord_k_proj(others_coord)          # [B, N, D_sub]
        v_coord = self.coord_v_proj(others_coord)          # [B, N, D_sub]
        
        attention_coord = self._cross_attention(q_coord, k_coord, v_coord)  # [B, N, D_sub]
        
        # Gating for coordinates
        gate_input_coord = torch.cat([f_coord, attention_coord], dim=-1)  # [B, N, 2*D_sub]
        gate_coord = self.coord_gate_mlp(gate_input_coord)  # [B, N, 1]
        
        f_coord_updated = (1 - gate_coord) * f_coord + gate_coord * attention_coord  # [B, N, D_sub]
        f_coord_updated = self.norm_coord(f_coord_updated)
        
        # 2. Update MSW features
        others_msw = torch.cat([f_coord, f_mslp], dim=-1)  # [B, N, 2*D_sub]
        q_msw = self.msw_q_proj(f_msw)                     # [B, N, D_sub]
        k_msw = self.msw_k_proj(others_msw)                # [B, N, D_sub]
        v_msw = self.msw_v_proj(others_msw)                # [B, N, D_sub]
        
        attention_msw = self._cross_attention(q_msw, k_msw, v_msw)  # [B, N, D_sub]
        
        # Gating for MSW
        gate_input_msw = torch.cat([f_msw, attention_msw], dim=-1)  # [B, N, 2*D_sub]
        gate_msw = self.msw_gate_mlp(gate_input_msw)  # [B, N, 1]
        
        f_msw_updated = (1 - gate_msw) * f_msw + gate_msw * attention_msw  # [B, N, D_sub]
        f_msw_updated = self.norm_msw(f_msw_updated)
        
        # 3. Update MSLP features
        others_mslp = torch.cat([f_coord, f_msw], dim=-1)  # [B, N, 2*D_sub]
        q_mslp = self.mslp_q_proj(f_mslp)                  # [B, N, D_sub]
        k_mslp = self.mslp_k_proj(others_mslp)             # [B, N, D_sub]
        v_mslp = self.mslp_v_proj(others_mslp)             # [B, N, D_sub]
        
        attention_mslp = self._cross_attention(q_mslp, k_mslp, v_mslp)  # [B, N, D_sub]
        
        # Gating for MSLP
        gate_input_mslp = torch.cat([f_mslp, attention_mslp], dim=-1)  # [B, N, 2*D_sub]
        gate_mslp = self.mslp_gate_mlp(gate_input_mslp)  # [B, N, 1]
        
        f_mslp_updated = (1 - gate_mslp) * f_mslp + gate_mslp * attention_mslp  # [B, N, D_sub]
        f_mslp_updated = self.norm_mslp(f_mslp_updated)
        
        # Concatenate updated features
        updated_features = torch.cat([f_coord_updated, f_msw_updated, f_mslp_updated], dim=-1)  # [B, N, 3*D_sub]
        
        # Project back to d_model dimension
        updated_features_transposed = updated_features.transpose(1, 2)  # [B, 3*D_sub, N]
        output = self.output_conv(updated_features_transposed).transpose(1, 2)  # [B, N, D_model]
        
        return output

class PhysicsConstraints(nn.Module):
    """
    Additional physics constraints that can be applied
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
    def geostrophic_constraint(self, coords: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
        """
        Apply geostrophic balance constraint
        This is a simplified implementation - in practice, you'd need proper gradient calculations
        """
        # This is a placeholder for geostrophic balance constraints
        # In practice, you would implement proper atmospheric physics equations
        return torch.zeros_like(coords)
    
    def thermodynamic_constraint(self, wind: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
        """
        Apply thermodynamic constraints between wind and pressure
        """
        # Placeholder for thermodynamic relationships
        # Could implement wind-pressure relationships based on TC theory
        return torch.zeros_like(wind)
    
    def forward(self, coords: torch.Tensor, wind: torch.Tensor, pressure: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply various physics constraints
        
        Returns:
            Dict containing constraint violations that can be used as additional loss terms
        """
        constraints = {}
        
        # Geostrophic constraint
        constraints['geostrophic'] = self.geostrophic_constraint(coords, pressure)
        
        # Thermodynamic constraint  
        constraints['thermodynamic'] = self.thermodynamic_constraint(wind, pressure)
        
        return constraints

class PIGATransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with optional embedded PIGA module
    
    Architecture:
    - With PIGA: Input -> Self-Attention -> Residual+Norm -> Cross-Attention -> Residual+Norm -> PIGA -> FFN -> Residual+Norm -> Output
    - Without PIGA: Input -> Self-Attention -> Residual+Norm -> Cross-Attention -> Residual+Norm -> FFN -> Residual+Norm -> Output
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.d_model = config['model']['d_model']
        self.num_heads = config['model']['decoder']['num_heads']
        self.d_ff = config['model']['decoder']['d_ff']
        self.dropout = config['model']['decoder']['dropout']
        
        # Check if PIGA should be enabled
        self.use_piga = config['model'].get('ablation', {}).get('use_piga', True)
        
        # Self-attention (with causal mask)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # PIGA module (embedded between cross-attention and FFN) - optional
        if self.use_piga:
            self.piga = PIGAModule(config)
        else:
            self.piga = None
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model),
            nn.Dropout(self.dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for self-attention"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def forward(self, 
                tgt: torch.Tensor, 
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tgt: [B, T, D_model] - target sequence (diffusion states)
            memory: [B, N, D_model] - encoder memory (context)
            tgt_mask: causal mask for target sequence
            
        Returns:
            output: [B, T, D_model] - decoded features
        """
        B, T, D = tgt.shape
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(T, tgt.device)
        
        # 1. Self-attention with causal mask
        attn_output, _ = self.self_attn(
            query=tgt,
            key=tgt, 
            value=tgt,
            attn_mask=tgt_mask
        )
        tgt = self.norm1(tgt + attn_output)
        
        # 2. Cross-attention with encoder memory
        cross_attn_output, _ = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory
        )
        h2 = self.norm2(tgt + cross_attn_output)
        
        # 3. PIGA module - Physics-Inspired gated attention (optional)
        if self.use_piga:
            h_piga = self.piga(h2)
            # 4. Feed-forward network with residual connection
            # Note: residual connection uses h2, PIGA output goes to FFN
            output = self.norm3(h2 + self.ffn(h_piga))
        else:
            # Without PIGA: direct FFN on cross-attention output
            output = self.norm3(h2 + self.ffn(h2))
        
        return output