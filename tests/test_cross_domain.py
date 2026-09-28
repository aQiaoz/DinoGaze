import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from cross_domain import aggregate_groups, evaluate_group, prepare_source
from reader import GazeDataset
from utils import load_config


class CrossDomainTest(unittest.TestCase):
    def test_source_val_and_model(self):
        protocol = load_config('config/cross_domain.yaml')
        for source in ('ethxgaze', 'gaze360'):
            config, fold = prepare_source(protocol, 'config/cross_domain.yaml', source, 'uniform224')
            self.assertTrue(fold['val'])
            self.assertNotEqual(fold['train'], fold['val'])
            self.assertEqual(config['data']['image_size'], 224)
            self.assertNotIn('scale', config['model'])

    def test_sample_weighted_vs_group_mean(self):
        groups = [dict(samples=1, mae=1., yaw_mae=1., pitch_mae=1.),
                  dict(samples=3, mae=5., yaw_mae=5., pitch_mae=5.)]
        result = aggregate_groups(groups)
        self.assertEqual(result['mae'], 4.)
        self.assertEqual(result['macro_group_mae'], 3.)

    def test_target_inference_never_gets_labels(self):
        class ImagesOnly(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(images.size(0), 2), None
        batch = (torch.zeros(2, 3, 8, 8), torch.zeros(2, 2), ['a', 'b'])
        result = evaluate_group(ImagesOnly(), [batch], torch.device('cpu'))
        self.assertEqual(result['samples'], 2)
        self.assertAlmostEqual(result['mae'], 0.)

    def test_plain_target_labels_and_eth_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'Label').mkdir()
            (root / 'Image').mkdir()
            Image.new('RGB', (8, 8)).save(root / 'Image' / 'face.png')
            (root / 'Label' / 'test.label').write_text('Face gaze\nface.png 0.2,0.4\n')
            dataset = GazeDataset(root, 'Image', ['test.label'], image_size=4, resize=True)
            image, targets, _ = dataset[0]
            torch.testing.assert_close(targets, torch.tensor([.4, .2]))
            self.assertEqual(tuple(image.shape), (3, 4, 4))
            with self.assertRaises(ValueError):
                GazeDataset(root, 'Image', ['test.label'], image_size=4)[0]


if __name__ == '__main__':
    unittest.main()
