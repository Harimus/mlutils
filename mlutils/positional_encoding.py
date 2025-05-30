import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Stolen from pytorch tutorial"""

    def __init__(self, d_model: int, embedding: int = 5000):
        super().__init__()

        position = torch.arange(embedding).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(embedding, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[embedding_dim]``
        """
        return self.pe[x]
