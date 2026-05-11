"""
Image and tabular encoders.
"""

import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    """
    3D residual block: two conv layers with a skip connection.
    Input and output have the same channel depth.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class ImageEncoder(nn.Module):
    """
    Lightweight 3D ResNet encoder for volumetric CT inputs.

    Architecture:
        Input  : (B, 1, 64, 64, 64)
        Stage 1: Conv3d stride-2 → 16ch + ResBlock  → (B, 16, 32, 32, 32)
        Stage 2: Conv3d stride-2 → 32ch + ResBlock  → (B, 32, 16, 16, 16)
        Stage 3: Conv3d stride-2 → 64ch + ResBlock  → (B, 64,  8,  8,  8)
        Pool   : AdaptiveAvgPool3d(1)               → (B, 64)
        Proj   : Linear(64, out_dim)                → (B, out_dim)

    Args:
        out_dim : output feature dimension (default 64)
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            ResBlock3D(16),

            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResBlock3D(32),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResBlock3D(64),

            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(x))


class TabularEncoder(nn.Module):
    """
    MLP encoder for structured clinical features.

    Architecture:
        in_dim → 32 → 64 → out_dim
        Each hidden layer: Linear + BatchNorm1d + ReLU + Dropout

    Args:
        in_dim  : number of input clinical features (default 3)
        out_dim : output feature dimension (default 64)
        dropout : dropout probability (default 0.3)
    """

    def __init__(self, in_dim: int = 3, out_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
