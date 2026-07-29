"""DINOGaze model.

A DINOv2 ViT backbone whose last few block features are fused by a small
Transformer and regressed to a 2D gaze (yaw, pitch).  Deep supervision: every
fused layer also has its own auxiliary head during training.
"""

import copy
import os

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Multi-layer feature fusion blocks
# --------------------------------------------------------------------------- #
class _Conv1x1Pool(nn.Module):
    """1x1 conv + BN + global average pool: [B, C, H, W] -> [B, hidden]."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, hidden, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.pool(x).flatten(1)


class _MLEnhance(nn.Module):
    """Project every layer's feature map to a common hidden dimension."""

    def __init__(self, in_dims, hidden):
        super().__init__()
        self.layers = nn.ModuleList([_Conv1x1Pool(d, hidden) for d in in_dims])

    def forward(self, feature_maps):
        tokens = [layer(f) for layer, f in zip(self.layers, feature_maps)]
        stacked = torch.stack(tokens, dim=0)          # [L, B, hidden]
        return stacked, tokens                         # tokens: list of [B, hidden]


class _EncoderLayer(nn.Module):
    """Pre-`pos` self-attention encoder layer (position re-added every layer)."""

    def __init__(self, dim, nhead=8, ff=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim, ff)
        self.linear2 = nn.Linear(ff, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, src, pos):
        q = k = src + pos.unsqueeze(1)                 # [L, B, dim]
        src = src + self.dropout1(self.attn(q, k, value=src)[0])
        src = self.norm1(src)
        src = src + self.dropout2(self.linear2(self.dropout(self.act(self.linear1(src)))))
        return self.norm2(src)


class _FusionTransformer(nn.Module):
    """A CLS token attends over the L layer tokens; returns the fused CLS feature."""

    def __init__(self, dim, length, nhead=8, ff=512, num_layers=6, dropout=0.1):
        super().__init__()
        if dim % nhead != 0:
            raise ValueError(f"trans_dim ({dim}) must be divisible by fusion_heads ({nhead}).")
        layer = _EncoderLayer(dim, nhead, ff, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pos = nn.Embedding(length + 1, dim)       # +1 for the CLS token

    def forward(self, tokens):                          # tokens: [L, B, dim]
        batch = tokens.size(1)
        cls = self.cls.repeat(1, batch, 1)              # [1, B, dim]
        x = torch.cat([cls, tokens], dim=0)             # [L+1, B, dim]
        pos = self.pos(torch.arange(x.size(0), device=x.device))
        for layer in self.layers:
            x = layer(x, pos)
        return self.norm(x)[0]                          # CLS row -> [B, dim]


# --------------------------------------------------------------------------- #
# Backbone loading
# --------------------------------------------------------------------------- #
def _load_dino(name):
    """Load a DINOv2 backbone from a local repo (DINO_REPO_PATH) or torch.hub."""
    repo = os.environ.get("DINO_REPO_PATH")
    if repo and os.path.isdir(repo):
        return torch.hub.load(repo, name, source="local")
    return torch.hub.load("facebookresearch/dinov2", name)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class DINOGaze(nn.Module):
    def __init__(self, name="dinov2_vits14", out_indices=(8, 9, 10, 11),
                 trans_dim=128, fusion_layers=6, fusion_heads=8,
                 fusion_ff=512, fusion_dropout=0.1,
                 freeze_backbone=False, main_weight=2.0):
        super().__init__()
        self.backbone = _load_dino(name)
        self.out_indices = tuple(out_indices)
        self.freeze_backbone = freeze_backbone
        self.main_weight = main_weight

        embed_dim = getattr(self.backbone, "embed_dim", None) \
            or getattr(self.backbone, "num_features", None)
        if embed_dim is None:
            raise RuntimeError(f"Cannot infer embed_dim from backbone {name}")

        for p in self.backbone.parameters():
            p.requires_grad = not freeze_backbone
        if freeze_backbone:
            self.backbone.eval()

        self.mle = _MLEnhance([embed_dim] * len(self.out_indices), trans_dim)
        self.fusion = _FusionTransformer(
            trans_dim,
            length=len(self.out_indices),
            nhead=fusion_heads,
            ff=fusion_ff,
            num_layers=fusion_layers,
            dropout=fusion_dropout,
        )
        self.head = nn.Linear(trans_dim, 2)
        self.aux_heads = nn.ModuleList(
            [nn.Linear(trans_dim, 2) for _ in self.out_indices])
        self.l1 = nn.L1Loss()

    def _features(self, image):
        return list(self.backbone.get_intermediate_layers(
            image, n=self.out_indices, reshape=True))

    def train(self, mode=True):
        """Keep a frozen backbone in eval mode after the parent mode switch."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, image):
        if self.freeze_backbone:
            with torch.no_grad():
                feature_maps = self._features(image)
        else:
            feature_maps = self._features(image)
        tokens, per_layer = self.mle(feature_maps)      # [L, B, d], list[B, d]
        fused = self.fusion(tokens)                      # [B, d]
        gaze = self.head(fused)                          # [B, 2]
        aux = [h(t) for h, t in zip(self.aux_heads, per_layer)]
        return gaze, aux

    def compute_loss(self, image, gaze_label):
        gaze, aux = self.forward(image)
        loss = self.main_weight * self.l1(gaze, gaze_label)
        for a in aux:
            loss = loss + self.l1(a, gaze_label)
        return loss, gaze


if __name__ == "__main__":
    net = DINOGaze().cuda()
    dummy = torch.zeros(2, 3, 224, 224).cuda()
    g, aux = net(dummy)
    print("gaze", g.shape, "| aux heads", len(aux))
