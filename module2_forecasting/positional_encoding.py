"""
Sinusoidal Positional Encoding for Transformer Models.

Injects positional information into the input embeddings so the Transformer
can distinguish between different time steps in the input sequence.

Uses the standard sinusoidal encoding from "Attention Is All You Need":
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

This encoding allows the model to attend to relative positions since
PE(pos+k) can be represented as a linear function of PE(pos) for any
fixed offset k.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.

    Generates fixed (non-learned) positional encodings using sine and cosine
    functions of different frequencies. These are added to the input embeddings
    to provide position information to the Transformer model.

    Properties:
    - Deterministic: No learnable parameters
    - Unique: Each position gets a unique encoding
    - Generalizable: Can handle sequences longer than seen during training
    - Bounded: Values stay in [-1, 1] range
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Dimension of the model embeddings.
            max_len: Maximum sequence length to pre-compute encodings for.
            dropout: Dropout rate applied after adding positional encoding.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the division term: 10000^(2i/d_model)
        # Using log space for numerical stability:
        # exp(-log(10000) * 2i / d_model) = 1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but moves with model to GPU)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Tensor with positional encoding added, same shape as input.
        """
        seq_len = x.size(1)
        # Add positional encoding (broadcasting across batch dimension)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

    def get_encoding(self, seq_len):
        """
        Get raw positional encoding for visualization.

        Args:
            seq_len: Sequence length.

        Returns:
            Numpy array of shape [seq_len, d_model].
        """
        return self.pe[0, :seq_len, :].detach().cpu().numpy()
