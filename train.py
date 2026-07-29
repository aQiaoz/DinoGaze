"""Train DINOGaze from one dataset YAML configuration."""

import argparse
import copy
import os
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import DINOGaze
from reader import build_loader
from utils import (
    Logger,
    TopKCheckpoints,
    aggregate_experiment,
    angular_error_deg,
    build_scheduler,
    load_config,
    resolve_folds,
    save_prediction_grid,
    set_seed,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINOGaze with a YAML experiment config.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML file.")
    parser.add_argument("--exp", default=None, help="Experiment name under save/. Overrides config.")
    parser.add_argument("--fold", nargs="+", default=None, help="Only run these named folds.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--save-root", default=None, help="Defaults to <project>/save.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size.")
    parser.add_argument("--workers", type=int, default=None, help="Override train.num_workers.")
    parser.add_argument("--limit", type=int, default=0, help="Cap each split for smoke tests.")
    return parser.parse_args()


def resolve_device(requested):
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.", flush=True)
        device = torch.device("cpu")
    if device.type == "cpu":
        os.environ.setdefault("XFORMERS_DISABLED", "1")
    return device


def build_split_loader(data_config, entries, train_config, *, train, seed, limit):
    return build_loader(
        data_config["root"],
        data_config.get("image_dir", "Image"),
        entries,
        train_config["batch_size"],
        train=train,
        num_workers=train_config.get("num_workers", 6),
        seed=seed,
        limit=limit,
    )


def apply_runtime_config(config, args):
    """Return the exact effective config used by this training invocation."""
    config = copy.deepcopy(config)
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.workers,
    }
    for key, value in overrides.items():
        if value is not None:
            config["train"][key] = value
    config["runtime"] = {"seed": args.seed, "limit": args.limit}
    return config


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    errors, yaw_errors, pitch_errors = [], [], []
    for images, targets, _ in tqdm(loader, desc="evaluate", leave=False):
        predictions, _ = model(images.to(device, non_blocking=True))
        targets = targets.to(device, non_blocking=True)
        errors.append(angular_error_deg(predictions, targets).cpu())
        yaw_errors.append(torch.rad2deg((predictions[:, 0] - targets[:, 0]).abs()).cpu())
        pitch_errors.append(torch.rad2deg((predictions[:, 1] - targets[:, 1]).abs()).cpu())

    if not errors:
        raise ValueError("Cannot evaluate an empty loader.")
    return {
        "mae": torch.cat(errors).mean().item(),
        "yaw_mae": torch.cat(yaw_errors).mean().item(),
        "pitch_mae": torch.cat(pitch_errors).mean().item(),
        "samples": int(sum(batch.numel() for batch in errors)),
    }


@torch.inference_mode()
def save_visualization(model, loader, device, output_path, max_items):
    model.eval()
    paths, predictions, targets = [], [], []
    for images, labels, image_paths in loader:
        predicted, _ = model(images.to(device, non_blocking=True))
        remaining = max_items - len(paths)
        paths.extend(image_paths[:remaining])
        predictions.extend(predicted.cpu().numpy()[:remaining])
        targets.extend(labels.numpy()[:remaining])
        if len(paths) >= max_items:
            break
    save_prediction_grid(paths, predictions, targets, output_path, max_items=max_items)


def train_epoch(model, loader, optimizer, scaler, device, amp_enabled):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, targets, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            loss, _ = model.compute_loss(images, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = targets.size(0)
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def checkpoint_payload(model, config, fold_name, epoch, monitor_name, monitor):
    return {
        "model": model.state_dict(),
        "model_config": config["model"],
        "dataset": config["dataset"],
        "fold": fold_name,
        "epoch": epoch,
        "monitor_name": monitor_name,
        "monitor": monitor,
    }


def run_fold(config, fold, args, device, experiment_dir, fold_index):
    train_config = config["train"]

    fold_dir = experiment_dir / fold["name"]
    log_dir = fold_dir / "log"
    checkpoint_dir = fold_dir / "checkpoint"
    logger = Logger(log_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
    writer = SummaryWriter(log_dir / "tensorboard")
    try:
        seed = args.seed + fold_index
        set_seed(seed)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        train_loader = build_split_loader(
            config["data"], fold["train"], train_config, train=True, seed=seed, limit=args.limit)
        test_loader = build_split_loader(
            config["data"], fold["test"], train_config, train=False, seed=seed, limit=args.limit)
        val_entries = fold.get("val")
        val_loader = None if not val_entries else build_split_loader(
            config["data"], val_entries, train_config, train=False, seed=seed, limit=args.limit)
        monitor_name = "val" if val_loader is not None else "test"
        monitor_loader = val_loader or test_loader

        logger(
            f"dataset={config['dataset']} fold={fold['name']} device={device} "
            f"train={len(train_loader.dataset)} {monitor_name}={len(monitor_loader.dataset)} "
            f"test={len(test_loader.dataset)}"
        )
        logger(
            f"epochs={train_config['epochs']} batch={train_config['batch_size']} "
            f"lr={train_config['lr']:.2e} eta_min={train_config['eta_min']:.2e} "
            f"weight_decay={train_config['weight_decay']}"
        )

        model = DINOGaze(**config["model"]).to(device)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=train_config["lr"],
            weight_decay=train_config["weight_decay"],
        )
        scheduler = build_scheduler(
            optimizer,
            train_config["epochs"],
            train_config.get("warmup_epochs", 0),
            train_config["eta_min"],
        )
        amp_enabled = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        checkpoints = TopKCheckpoints(checkpoint_dir, train_config.get("save_top_k", 5))
        visualize_samples = config.get("logging", {}).get("visualize_samples", 16)

        for epoch in range(1, train_config["epochs"] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device, amp_enabled)
            monitor = evaluate(model, monitor_loader, device)
            current_lr = optimizer.param_groups[0]["lr"]   # lr actually used this epoch
            scheduler.step()
            payload = checkpoint_payload(model, config, fold["name"], epoch, monitor_name, monitor)
            _, is_best = checkpoints.update(payload, epoch, monitor["mae"])

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar(f"mae/{monitor_name}", monitor["mae"], epoch)
            writer.add_scalar("lr", current_lr, epoch)
            logger(
                f"epoch={epoch:03d}/{train_config['epochs']} loss={train_loss:.5f} "
                f"{monitor_name}_mae={monitor['mae']:.3f} yaw={monitor['yaw_mae']:.3f} "
                f"pitch={monitor['pitch_mae']:.3f} lr={current_lr:.2e}"
                f"{' best' if is_best else ''}"
            )
            if is_best and visualize_samples:
                save_visualization(
                    model, monitor_loader, device,
                    log_dir / "vis" / f"epoch_{epoch:03d}_{monitor_name}_{monitor['mae']:.3f}.jpg",
                    visualize_samples,
                )

        best_path = checkpoints.best_path
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        test = evaluate(model, test_loader, device)
        logger(f"best={best_path.name} final_test_mae={test['mae']:.3f} "
               f"yaw={test['yaw_mae']:.3f} pitch={test['pitch_mae']:.3f}")
        result = {
            "fold": fold["name"],
            "best_checkpoint": str(best_path),
            "best_epoch": checkpoint["epoch"],
            "monitor_name": monitor_name,
            "monitor_mae": checkpoint["monitor"]["mae"],
            "test": test,
        }
        write_json(fold_dir / "result.json", result)
        return result
    finally:
        writer.close()
        logger.close()


def main():
    args = parse_args()
    config = apply_runtime_config(load_config(args.config), args)
    folds = resolve_folds(config)
    fold_indices = {fold["name"]: index for index, fold in enumerate(folds)}
    if len(fold_indices) != len(folds):
        raise ValueError("Fold names must be unique.")
    if args.fold:
        requested = set(args.fold)
        folds = [fold for fold in folds if fold["name"] in requested]
        missing = requested - {fold["name"] for fold in folds}
        if missing:
            raise ValueError(f"Unknown fold(s): {', '.join(sorted(missing))}")

    device = resolve_device(args.device)
    project_dir = Path(__file__).resolve().parent
    save_root = Path(args.save_root) if args.save_root else project_dir / "save"
    default_exp = f"{config['dataset'].lower()}_{config['model']['name']}"
    experiment_dir = save_root / (args.exp or config.get("experiment") or default_exp)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = experiment_dir / "config.yaml"
    if snapshot_path.is_file():
        saved_config = load_config(snapshot_path)
        if saved_config != config:
            raise ValueError(
                f"{snapshot_path} does not match this run's effective config. "
                "Use a new --exp name to keep experiment artifacts separate."
            )
    else:
        with snapshot_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)

    results = [
        run_fold(config, fold, args, device, experiment_dir, fold_indices[fold["name"]])
        for fold in folds
    ]
    summary = aggregate_experiment(experiment_dir, config)
    write_json(experiment_dir / "summary.json", summary)
    print(f"\nRan {len(results)} fold(s) this session. "
          f"Experiment now has {summary['completed']}/{summary['total_folds']} folds: "
          f"mean test MAE = {summary['mean_test_mae']:.3f} deg"
          if summary["mean_test_mae"] is not None else
          f"\nRan {len(results)} fold(s); no completed folds found to aggregate.")


if __name__ == "__main__":
    main()
