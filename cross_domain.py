"""Source-only DG: train once on E/G and test on all M/D target groups."""

import argparse
import copy
from pathlib import Path

import torch
import yaml

from model import DINOGaze
from reader import build_loader
from train import apply_runtime_config, resolve_device, run_fold
from utils import angular_error_deg, load_config, resolve_folds, write_json


def prepare_source(protocol, protocol_path, source, variant):
    path = Path(protocol_path).resolve().parent / protocol['sources'][source]
    config = copy.deepcopy(load_config(path))
    overrides = protocol.get('source_overrides', {}).get(source, {})
    for section in ('data', 'train'):
        config[section].update(overrides.get(section, {}))
    folds = resolve_folds(config)
    if len(folds) != 1 or not folds[0].get('val'):
        raise ValueError('DG requires one source split with a source validation set.')
    if folds[0]['train'] == folds[0]['val']:
        raise ValueError('Source train and validation entries must differ.')
    if variant != 'uniform224' or config['model'].get('scale'):
        raise ValueError(f'Unknown variant: {variant}')
    config['data'].update(image_size=224, resize=True)
    config['dg'] = {'source': source, 'variant': variant,
                    'selection': 'source_val', 'target_adaptation': False}
    return config, folds[0]


def aggregate_groups(groups):
    count = sum(group['samples'] for group in groups)
    if not count:
        raise ValueError('Target evaluation has no samples.')
    return {
        'samples': count,
        **{key: sum(g[key] * g['samples'] for g in groups) / count
           for key in ('mae', 'yaw_mae', 'pitch_mae')},
        'macro_group_mae': sum(g['mae'] for g in groups) / len(groups),
        'groups': groups,
    }


@torch.inference_mode()
def evaluate_group(model, loader, device):
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    count = 0
    for images, labels, _ in loader:
        # Neither target gaze labels nor target eye boxes enter the model.
        predicted, _ = model(images.to(device, non_blocking=True))
        gaze = labels.to(device, non_blocking=True)
        angular = angular_error_deg(predicted, gaze)
        component = torch.rad2deg((predicted - gaze).abs())
        totals += torch.stack((angular.double().sum(), component[:, 0].double().sum(),
                               component[:, 1].double().sum()))
        count += gaze.size(0)
    if not count:
        raise ValueError('Cannot evaluate an empty target group.')
    if not torch.isfinite(totals).all():
        raise ValueError('Nonfinite target evaluation metrics.')
    mean = (totals / count).tolist()
    return dict(samples=count, mae=mean[0], yaw_mae=mean[1], pitch_mae=mean[2])


def evaluate_targets(checkpoint_path, config, protocol, args, device, output):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('dataset') != config['dataset'] or checkpoint.get('monitor_name') != 'val':
        raise ValueError('DG requires a checkpoint selected on this source dataset validation set.')
    if 'model_config' not in checkpoint:
        raise ValueError('Checkpoint must contain model_config; bare state dict is ambiguous.')
    model_config = copy.deepcopy(checkpoint['model_config'])
    if model_config.get('scale') or model_config.get('image_size', 224) != 224:
        raise ValueError('Expected a main uniform224 checkpoint.')
    model_config.pop('image_size', None)
    model_config.pop('scale', None)
    model = DINOGaze(**model_config).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    result = {
        'protocol': 'source-only DG; source-val checkpoint; no target adaptation',
        'source': args.source, 'variant': args.variant,
        'checkpoint': str(Path(checkpoint_path).resolve()),
        'epoch': checkpoint.get('epoch'), 'source_monitor': checkpoint.get('monitor'),
        'model_config': model_config, 'input_size': 224, 'limit_per_group': args.limit,
        'smoke_only': bool(args.limit), 'targets': {},
    }
    for name, target in protocol['targets'].items():
        if not target.get('groups') or len(set(target['groups'])) != len(target['groups']):
            raise ValueError(f'Empty or duplicate target groups: {name}')
        groups, seen_paths = [], set()
        for label in target['groups']:
            loader = build_loader(target['root'], target.get('image_dir', 'Image'), [label],
                                  config['train']['batch_size'], train=False,
                                  num_workers=config['train'].get('num_workers', 6),
                                  limit=args.limit, image_size=224, resize=True)
            dataset = getattr(loader.dataset, 'dataset', loader.dataset)
            paths = [record[0] for record in dataset.records]
            if len(set(paths)) != len(paths) or seen_paths.intersection(paths):
                raise ValueError(f'Duplicate target image entries in {name}/{label}.')
            seen_paths.update(paths)
            metrics = evaluate_group(model, loader, device)
            groups.append({'label': label, **metrics})
            print(f'{args.source}->{name}/{label}: {metrics["mae"]:.3f} deg', flush=True)
        report = aggregate_groups(groups)
        report['data'] = target
        result['targets'][name] = report
        write_json(output, result)
        print(f'{args.source}->{name}: sample MAE={report["mae"]:.3f}, '
              f'group mean={report["macro_group_mae"]:.3f}, n={report["samples"]}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--config', default='config/cross_domain.yaml')
    parser.add_argument('--source', choices=['ethxgaze', 'gaze360'], required=True)
    parser.set_defaults(variant='uniform224')
    parser.add_argument('--checkpoint', help='Explicit source-val checkpoint for test mode.')
    parser.add_argument('--exp')
    parser.add_argument('--output', help='Target evaluation JSON for test mode.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-root', default=str(Path(__file__).resolve().parent / 'save'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--limit', type=int, default=0, help='Smoke-only cap per split/group.')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative.')
    if args.mode == 'test' and not args.checkpoint:
        parser.error('test requires --checkpoint selected on source val.')
    if args.mode == 'train' and (args.checkpoint or args.output):
        parser.error('--checkpoint/--output are only supported in test mode.')
    protocol = load_config(args.config)
    config, fold = prepare_source(protocol, args.config, args.source, args.variant)
    config = apply_runtime_config(config, args)
    name = args.exp or f'dg_{args.source}_{args.variant}'
    experiment = Path(args.save_root) / name
    device = resolve_device(args.device)
    if args.mode == 'train':
        # Do not silently mix or resume checkpoints from a previous run.
        experiment.mkdir(parents=True, exist_ok=False)
        with (experiment / 'config.yaml').open('w') as file:
            yaml.safe_dump(config, file, sort_keys=False)
        with (experiment / 'protocol.yaml').open('w') as file:
            yaml.safe_dump(protocol, file, sort_keys=False)
        source_result = run_fold(config, fold, args, device, experiment, 0)
        write_json(experiment / 'source_result.json', source_result)
        args.checkpoint = source_result['best_checkpoint']
    output = Path(args.output) if args.output else experiment / 'cross_domain.json'
    if output.exists():
        raise FileExistsError(f'Use a new output path instead of overwriting {output}')
    evaluate_targets(args.checkpoint, config, protocol, args, device, output)


if __name__ == '__main__':
    main()
