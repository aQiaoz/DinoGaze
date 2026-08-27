import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from model import DINOGaze


class _FakeDINO(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, self.embed_dim, bias=False)

    def forward_features(self, image):
        pooled = image.mean(dim=(-2, -1))
        return {
            "x_norm_clstoken": self.projection(pooled),
            "x_norm_patchtokens": image.new_empty(image.size(0), 0, self.embed_dim),
        }


class DINOCLSGazeTest(unittest.TestCase):
    def _model(self, **kwargs):
        with patch("model._load_dino", return_value=_FakeDINO()):
            return DINOGaze(head_hidden=4, head_dropout=0.0, **kwargs)

    def test_head_receives_native_dino_cls_token(self):
        model = self._model()
        images = torch.randn(3, 3, 8, 8)
        expected = model.backbone.forward_features(images)["x_norm_clstoken"]
        received = []
        hook = model.head[0].register_forward_pre_hook(
            lambda _module, inputs: received.append(inputs[0].detach().clone()))

        gaze, auxiliary = model(images)
        hook.remove()

        self.assertEqual(tuple(gaze.shape), (3, 2))
        self.assertEqual(auxiliary, [])
        self.assertTrue(torch.equal(received[0], expected))

    def test_single_l1_loss_updates_backbone_and_head(self):
        model = self._model()
        images = torch.randn(2, 3, 8, 8)
        labels = torch.randn(2, 2)

        loss, gaze = model.compute_loss(images, labels)
        loss.backward()

        self.assertEqual(tuple(gaze.shape), (2, 2))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.backbone.projection.weight.grad)
        self.assertIsNotNone(model.head[0].weight.grad)

    def test_frozen_backbone_stays_in_eval_mode(self):
        model = self._model(freeze_backbone=True)
        model.train()

        self.assertFalse(model.backbone.training)
        self.assertTrue(model.head.training)
        self.assertTrue(all(not parameter.requires_grad
                            for parameter in model.backbone.parameters()))


if __name__ == "__main__":
    unittest.main()
