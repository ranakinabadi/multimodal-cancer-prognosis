"""
Fusion architectures: concat and bidirectional cross-modal attention.
"""

import torch
import torch.nn as nn
from .encoders import ImageEncoder, TabularEncoder


class ConcatFusionModel(nn.Module):
    """
    Naive fusion baseline: concatenate image and tabular features,
    pass through an MLP head.

    Architecture:
        f_img  = ImageEncoder(ct)        → (B, 64)
        f_tab  = TabularEncoder(tab)     → (B, 64)
        fused  = cat([f_img, f_tab])     → (B, 128)
        output = MLP(fused)              → (B,)
    """

    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.img_encoder = ImageEncoder(out_dim=feat_dim)
        self.tab_encoder = TabularEncoder(in_dim=3, out_dim=feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, ct: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        f_img = self.img_encoder(ct)
        f_tab = self.tab_encoder(tab)
        fused = torch.cat([f_img, f_tab], dim=1)
        return self.head(fused).squeeze(-1)


class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional cross-modal attention fusion.

    Each modality attends to the other, producing a representation
    that is conditioned on complementary features:

        Direction 1 — imaging queries tabular:
            Q1 = W_q  · f_img
            K1 = W_k  · f_tab
            V1 = W_v  · f_tab
            a1 = softmax(Q1 K1ᵀ / √d)
            f1 = a1 · V1                    (attended tabular context)

        Direction 2 — tabular queries imaging:
            Q2 = W_q2 · f_tab
            K2 = W_k2 · f_img
            V2 = W_v2 · f_img
            a2 = softmax(Q2 K2ᵀ / √d)
            f2 = a2 · V2                    (attended imaging context)

        fused  = cat([f1, f2])              → (B, 2d)
        output = head(fused)                → (B,)

    Args:
        feat_dim : shared feature dimension d (default 64)
    """

    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.img_encoder = ImageEncoder(out_dim=feat_dim)
        self.tab_encoder = TabularEncoder(in_dim=3, out_dim=feat_dim)

        # Direction 1: image → tabular
        self.W_q  = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_k  = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_v  = nn.Linear(feat_dim, feat_dim, bias=False)

        # Direction 2: tabular → image
        self.W_q2 = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_k2 = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_v2 = nn.Linear(feat_dim, feat_dim, bias=False)

        self.scale = feat_dim ** 0.5

        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def _attend(self,
                query_enc: nn.Linear,
                key_enc:   nn.Linear,
                val_enc:   nn.Linear,
                query_feat: torch.Tensor,
                kv_feat:    torch.Tensor) -> torch.Tensor:
        """Single-direction attention. All inputs are (B, d)."""
        Q = query_enc(query_feat).unsqueeze(1)    # (B, 1, d)
        K = key_enc(kv_feat).unsqueeze(1)         # (B, 1, d)
        V = val_enc(kv_feat).unsqueeze(1)         # (B, 1, d)
        a = torch.softmax(
                torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        return torch.bmm(a, V).squeeze(1)         # (B, d)

    def forward(self, ct: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        f_img = self.img_encoder(ct)
        f_tab = self.tab_encoder(tab)

        f1 = self._attend(self.W_q,  self.W_k,  self.W_v,  f_img, f_tab)
        f2 = self._attend(self.W_q2, self.W_k2, self.W_v2, f_tab, f_img)

        fused = torch.cat([f1, f2], dim=1)
        return self.head(fused).squeeze(-1)

    @torch.no_grad()
    def get_attention_weights(self,
                               ct:  torch.Tensor,
                               tab: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Return attention scalars for interpretability analysis.

        Returns:
            {
              'img_to_tab': (B,)  — how much imaging attends to tabular,
              'tab_to_img': (B,)  — how much tabular attends to imaging,
            }
        """
        f_img = self.img_encoder(ct)
        f_tab = self.tab_encoder(tab)

        Q1 = self.W_q(f_img).unsqueeze(1)
        K1 = self.W_k(f_tab).unsqueeze(1)
        a1 = torch.softmax(
                torch.bmm(Q1, K1.transpose(1, 2)) / self.scale, dim=-1)

        Q2 = self.W_q2(f_tab).unsqueeze(1)
        K2 = self.W_k2(f_img).unsqueeze(1)
        a2 = torch.softmax(
                torch.bmm(Q2, K2.transpose(1, 2)) / self.scale, dim=-1)

        return {
            "img_to_tab": a1.squeeze(),
            "tab_to_img": a2.squeeze(),
        }
