"""Evaluate one DINOGaze checkpoint on a YAML-configured split."""

import argparse
from pathlib import Path

import torch

from model import DINOGaze
from reader import build_loader
from train import evaluate, resolve_device
from utils import find_best_checkpoint, load_config, read_json, resolve_folds


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DINOGaze checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Path to a checkpoint file.")
    parser.add_argument("--exp", default=None, help="Experiment name used to infer the checkpoint directory.")
    parser.add_argument("--fold", required=True, help="Fold name from the YAML config.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-root", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    fold = next((item for item in resolve_folds(config) if item["name"] == args.fold), None)
    if fold is None:
        raise ValueError(f"Unknown fold {args.fold!r}")
    if args.split not in fold:
        raise ValueError(f"Fold {args.fold!r} has no {args.split!r} split.")

    train_config = dict(config["train"])
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.workers is not None:
        train_config["num_workers"] = args.workers
    device = resolve_device(args.device)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        project_dir = Path(__file__).resolve().parent
        save_root = Path(args.save_root) if args.save_root else project_dir / "save"
        experiment = args.exp or config.get("experiment")
        if not experiment:
            experiment = f"{config['dataset'].lower()}_{config['model']['name']}"
        fold_dir = save_root / experiment / args.fold
        result_path = fold_dir / "result.json"
        checkpoint_path = None
        if result_path.is_file():
            recorded_path = Path(read_json(result_path)["best_checkpoint"])
            checkpoint_path = recorded_path if recorded_path.is_file() else (
                fold_dir / "checkpoint" / recorded_path.name
            )
            if not checkpoint_path.is_file():
                checkpoint_path = None
        if checkpoint_path is None:
            checkpoint_path = find_best_checkpoint(fold_dir / "checkpoint")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = DINOGaze(**checkpoint.get("model_config", config["model"])).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    loader = build_loader(
        config["data"]["root"],
        config["data"].get("image_dir", "Image"),
        fold[args.split],
        train_config["batch_size"],
        train=False,
        num_workers=train_config.get("num_workers", 6),
        limit=args.limit,
    )
    metrics = evaluate(model, loader, device)
    print(f"checkpoint: {checkpoint_path}")
    print(f"dataset/fold/split: {config['dataset']} / {args.fold} / {args.split}")
    print(f"samples: {metrics['samples']}")
    print(f"angular MAE: {metrics['mae']:.3f} deg")
    print(f"yaw MAE: {metrics['yaw_mae']:.3f} deg")
    print(f"pitch MAE: {metrics['pitch_mae']:.3f} deg")


if __name__ == "__main__":
    main()
