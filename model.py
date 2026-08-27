"""DINOv2 CLS-token baseline for 2D gaze regression."""

import os

import torch
import torch.nn as nn


def _load_dino(name):
    """Load a DINOv2 backbone from a local repo (DINO_REPO_PATH) or torch.hub."""
    repo = os.environ.get("DINO_REPO_PATH")
    if repo and os.path.isdir(repo):
        return torch.hub.load(repo, name, source="local")
    return torch.hub.load("facebookresearch/dinov2", name)


class DINOGaze(nn.Module):
    def __init__(self, name="dinov2_vits14", head_hidden=128,
                 head_dropout=0.1, freeze_backbone=False):
        super().__init__()
        if head_hidden < 1:
            raise ValueError("head_hidden must be at least 1.")
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must be in [0, 1).")

        self.backbone = _load_dino(name)
        self.freeze_backbone = freeze_backbone

        embed_dim = getattr(self.backbone, "embed_dim", None) \
            or getattr(self.backbone, "num_features", None)
        if embed_dim is None:
            raise RuntimeError(f"Cannot infer embed_dim from backbone {name}")

        for p in self.backbone.parameters():
            p.requires_grad = not freeze_backbone
        if freeze_backbone:
            self.backbone.eval()

        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 2),
        )
        self.l1 = nn.L1Loss()

    def _cls_features(self, image):
        features = self.backbone.forward_features(image)
        if not isinstance(features, dict) or "x_norm_clstoken" not in features:
            raise RuntimeError("DINOv2 backbone did not return x_norm_clstoken.")
        return features["x_norm_clstoken"]

    def train(self, mode=True):
        """Keep a frozen backbone in eval mode after the parent mode switch."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, image):
        if self.freeze_backbone:
            with torch.no_grad():
                cls_token = self._cls_features(image)
        else:
            cls_token = self._cls_features(image)
        return self.head(cls_token), []

    def compute_loss(self, image, gaze_label):
        gaze, _ = self.forward(image)
        return self.l1(gaze, gaze_label), gaze


if __name__ == "__main__":
    net = DINOGaze().cuda()
    dummy = torch.zeros(2, 3, 224, 224).cuda()
    g, aux = net(dummy)
    print("gaze", g.shape, "| aux heads", len(aux))
