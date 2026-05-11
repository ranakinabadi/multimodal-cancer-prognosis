"""
Unimodal baseline models.
"""

import torch
import torch.nn as nn
from .encoders import ImageEncoder, TabularEncoder


class ImageOnlyModel(nn.Module):
    """
    Survival prediction from 3D CT only.
    Tabular input is accepted but ignored, keeping the API
    consistent across all models.
    """

    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.encoder = ImageEncoder(out_dim=feat_dim)
        self.head    = nn.Linear(feat_dim, 1)

    def forward(self, ct: torch.Tensor, _tab: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(ct)).squeeze(-1)


class TabularOnlyModel(nn.Module):
    """
    Survival prediction from clinical tabular features only.
    CT input is accepted but ignored, keeping the API
    consistent across all models.
    """

    def __init__(self, in_dim: int = 3, feat_dim: int = 64):
        super().__init__()
        self.encoder = TabularEncoder(in_dim=in_dim, out_dim=feat_dim)
        self.head    = nn.Linear(feat_dim, 1)

    def forward(self, _ct: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(tab)).squeeze(-1)
