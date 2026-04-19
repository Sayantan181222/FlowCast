"""
Transformer-Based Demand Forecasting Model.

Implements a Transformer Encoder architecture for time series demand prediction.
The model learns temporal dependencies in historical demand patterns to forecast
future transportation demand.

Architecture Overview:
    Input (seq_len × num_features)
        → Linear Projection (num_features → d_model)
        → Positional Encoding
        → N × Transformer Encoder Layers
            ├── Multi-Head Self-Attention (captures temporal dependencies)
            ├── Add & LayerNorm
            ├── Feed-Forward Network (per-position transformation)
            └── Add & LayerNorm
        → Global Average Pooling (aggregate sequence information)
        → Prediction Head (d_model → forecast_horizon)
    Output (forecast_horizon)

The self-attention mechanism enables the model to capture:
- Long-range temporal dependencies (e.g., weekly patterns)
- Variable-importance weighting across time steps
- Complex non-linear temporal dynamics
"""

import torch
import torch.nn as nn
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FORECAST_CONFIG
from module2_forecasting.positional_encoding import PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer with Pre-LayerNorm.

    Implements the standard Transformer encoder block:
    1. Multi-Head Self-Attention with residual connection
    2. Feed-Forward Network with residual connection

    Uses Pre-LayerNorm (LN before attention/FFN) which provides more stable
    training compared to Post-LayerNorm.
    """

    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        """
        Initialize encoder layer.

        Args:
            d_model: Model embedding dimension.
            n_heads: Number of attention heads.
            dim_feedforward: Hidden dimension of the FFN.
            dropout: Dropout probability.
        """
        super(TransformerEncoderLayer, self).__init__()

        # Multi-Head Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-Forward Network
        # Two linear layers with GELU activation (smoother than ReLU)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Layer Normalization (Pre-Norm configuration)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass through the encoder layer.

        Args:
            x: Input tensor [batch_size, seq_len, d_model].
            src_mask: Optional attention mask.
            src_key_padding_mask: Optional padding mask.

        Returns:
            Output tensor [batch_size, seq_len, d_model].
        """
        # Pre-LayerNorm Self-Attention with residual
        normed = self.norm1(x)
        attn_output, attn_weights = self.self_attention(
            normed, normed, normed,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + self.dropout(attn_output)

        # Pre-LayerNorm FFN with residual
        normed = self.norm2(x)
        ffn_output = self.ffn(normed)
        x = x + ffn_output

        return x


class DemandTransformer(nn.Module):
    """
    Transformer Encoder for Transportation Demand Forecasting.

    Given a sequence of historical demand and temporal features, predicts
    future demand values. The model uses:
    - Linear projection to map raw features to d_model embedding space
    - Sinusoidal positional encoding for temporal position awareness
    - Stack of Transformer encoder layers for sequence modeling
    - Global average pooling to aggregate temporal information
    - MLP prediction head, mapping to forecast horizon

    Mathematical formulation:
        Input:  X = [x_{t-L}, x_{t-L+1}, ..., x_{t-1}]  ∈ R^{L × F}
        Output: Y = [ŷ_t, ŷ_{t+1}, ..., ŷ_{t+H-1}]     ∈ R^H

    where L = lookback_window, F = num_features, H = forecast_horizon
    """

    def __init__(self, num_features, d_model=None, n_heads=None,
                 n_encoder_layers=None, dim_feedforward=None,
                 dropout=None, forecast_horizon=None):
        """
        Initialize the Demand Transformer.

        Args:
            num_features: Number of input features per time step.
            d_model: Embedding dimension (default: 64).
            n_heads: Number of attention heads (default: 4).
            n_encoder_layers: Number of encoder layers (default: 4).
            dim_feedforward: FFN hidden dimension (default: 256).
            dropout: Dropout rate (default: 0.1).
            forecast_horizon: Number of future steps to predict (default: 6).
        """
        super(DemandTransformer, self).__init__()

        self.num_features = num_features
        self.d_model = d_model or FORECAST_CONFIG["d_model"]
        self.n_heads = n_heads or FORECAST_CONFIG["n_heads"]
        self.n_layers = n_encoder_layers or FORECAST_CONFIG["n_encoder_layers"]
        self.dim_ff = dim_feedforward or FORECAST_CONFIG["dim_feedforward"]
        self.dropout_rate = dropout or FORECAST_CONFIG["dropout"]
        self.forecast_horizon = forecast_horizon or FORECAST_CONFIG["forecast_horizon"]

        # === Input Projection ===
        # Maps raw features (F dimensions) to d_model embedding space
        self.input_projection = nn.Sequential(
            nn.Linear(num_features, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )

        # === Positional Encoding ===
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_len=500,
            dropout=self.dropout_rate,
        )

        # === Transformer Encoder Stack ===
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dim_feedforward=self.dim_ff,
                dropout=self.dropout_rate,
            )
            for _ in range(self.n_layers)
        ])

        # Final layer norm after encoder stack
        self.final_norm = nn.LayerNorm(self.d_model)

        # === Prediction Head ===
        # MLP that maps pooled encoder output to forecast horizon
        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.5),  # Less dropout near output
            nn.Linear(self.d_model, self.forecast_horizon),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights using Xavier uniform initialization.

        Xavier init scales weights based on fan-in/fan-out, preventing
        vanishing/exploding gradients in deep networks.
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """
        Forward pass: historical sequence → demand forecast.

        Args:
            x: Input tensor [batch_size, seq_len, num_features].
               Contains lookback_window time steps of historical data.

        Returns:
            Predictions [batch_size, forecast_horizon].
        """
        # Step 1: Project input features to d_model dimension
        # [B, L, F] → [B, L, d_model]
        x = self.input_projection(x)

        # Step 2: Add positional encoding
        # [B, L, d_model] → [B, L, d_model]
        x = self.positional_encoding(x)

        # Step 3: Pass through Transformer encoder stack
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # Step 4: Final layer normalization
        x = self.final_norm(x)

        # Step 5: Global Average Pooling across sequence dimension
        # [B, L, d_model] → [B, d_model]
        # This aggregates information from all time steps into a fixed-size vector
        x = x.mean(dim=1)

        # Step 6: Prediction head
        # [B, d_model] → [B, forecast_horizon]
        predictions = self.prediction_head(x)

        return predictions

    def get_attention_weights(self, x):
        """
        Extract attention weight matrices for visualization.

        Args:
            x: Input tensor [batch_size, seq_len, num_features].

        Returns:
            List of attention weight tensors, one per encoder layer.
            Each tensor has shape [batch_size, n_heads, seq_len, seq_len].
        """
        attention_weights = []

        x = self.input_projection(x)
        x = self.positional_encoding(x)

        for encoder_layer in self.encoder_layers:
            normed = encoder_layer.norm1(x)
            _, attn_w = encoder_layer.self_attention(
                normed, normed, normed,
                need_weights=True,
                average_attn_weights=False,
            )
            attention_weights.append(attn_w.detach())
            x = encoder_layer(x)

        return attention_weights

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def summary(self):
        """Print model architecture summary."""
        params = self.count_parameters()
        print(f"\n{'=' * 50}")
        print(f"DemandTransformer Architecture Summary")
        print(f"{'=' * 50}")
        print(f"Input features:     {self.num_features}")
        print(f"Embedding dim:      {self.d_model}")
        print(f"Attention heads:    {self.n_heads}")
        print(f"Encoder layers:     {self.n_layers}")
        print(f"FFN hidden dim:     {self.dim_ff}")
        print(f"Dropout rate:       {self.dropout_rate}")
        print(f"Forecast horizon:   {self.forecast_horizon}")
        print(f"{'─' * 50}")
        print(f"Total parameters:     {params['total']:,}")
        print(f"Trainable parameters: {params['trainable']:,}")
        print(f"{'=' * 50}")
        return params
