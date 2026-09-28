import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from PIL import Image

import cross_domain
import train


SAT = hasattr(train, 'validate_input_config')


class TinyGaze(torch.nn.Module):
    """Exercise CLI/checkpoint plumbing without downloading DINO or using GPU."""
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.1))
        self.scale_router = None

    def forward(self, images, scale_target=None):
        return self.weight.expand(images.size(0), 2), None

    def build_scale_target(self, *args):
        return None

    def scale_specification(self):
        return None

    def compute_loss(self, images, gaze, *args, **kwargs):
        pred, _ = self(images)
        loss = (pred - gaze).abs().mean()
        return (loss, pred, dict(gaze=loss, main=loss, auxiliary=loss * 0,
                                scale=loss * 0)) if SAT else (loss, pred)


class CrossCLITest(unittest.TestCase):
    def test_train_select_reload_and_two_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'data'
            (data / 'Label').mkdir(parents=True)
            (data / 'Image').mkdir()
            header = 'Face 2DGaze LeftEyeBox RightEyeBox LeftEyeValid RightEyeValid\n'
            for split in ('train', 'val', 'test', 'm', 'd'):
                Image.new('RGB', (448, 448)).save(data / 'Image' / f'{split}.png')
                (data / 'Label' / f'{split}.label').write_text(
                    header + f'{split}.png 0.2,0.1 -1,-1,-1,-1 -1,-1,-1,-1 0 0\n')
            cfg = dict(dataset='ETH-XGaze', data=dict(root=str(data), image_size=448,
                       resize=False), model=dict(name='fake'),
                       train=dict(epochs=1, batch_size=1, num_workers=0, lr=1e-3,
                                  weight_decay=0., eta_min=1e-5, save_top_k=1),
                       logging=dict(visualize_samples=0),
                       folds=[dict(name='split', train=['train.label'],
                                   val=['val.label'], test=['test.label'])])
            if SAT:
                cfg['model'].update(image_size=448, scale=dict(refine_ratio=.15, st_weight=.1))
            (root / 'source.yaml').write_text(yaml.safe_dump(cfg))
            protocol = dict(sources=dict(ethxgaze='source.yaml'), targets={
                name: dict(root=str(data), groups=[label])
                for name, label in [('mpiifacegaze', 'm.label'), ('eyediap', 'd.label')]})
            path = root / 'dg.yaml'
            path.write_text(yaml.safe_dump(protocol))
            argv = ['cross_domain.py', 'train', '--source', 'ethxgaze', '--config', str(path),
                    '--device', 'cpu', '--save-root', str(root / 'save'), '--exp', 'smoke']
            with patch('sys.argv', argv), patch.object(train, 'DINOGaze', TinyGaze), \
                    patch.object(cross_domain, 'DINOGaze', TinyGaze):
                cross_domain.main()
            out = root / 'save' / 'smoke'
            result = json.loads((out / 'cross_domain.json').read_text())
            self.assertEqual(set(result['targets']), {'mpiifacegaze', 'eyediap'})
            self.assertEqual(result['targets']['eyediap']['samples'], 1)
            self.assertTrue(Path(result['checkpoint']).is_file())
            source = json.loads((out / 'source_result.json').read_text())
            self.assertEqual(source['monitor_name'], 'val')


if __name__ == '__main__':
    unittest.main()
