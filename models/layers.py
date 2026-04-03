"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        
        if not self.training:
            return x
        
        if self.p == 0.0:
            return x

        keep_prob = 1.0 - self.p
        # sample bernoulli mask : tossing n p-biased coins independently
        mask = torch.bernoulli(torch.full_like(x, keep_prob))

        # Inverted Dropout: keep the expected value same by dividing by 1-p
        return x*mask/keep_prob
    
