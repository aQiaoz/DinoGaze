"""Shared configuration, geometry, logging, and checkpoint helpers."""

import json
import random
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

CHECKPOINT_PATTERN = re.compile(r"epoch_\d+_mae_([0-9.]+)\.pt$")


def load_config(path):
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def resolve_folds(config):
    """Expand a fixed split list or a compact leave-one-out declaration."""
    spec = config["folds"]
    if isinstance(spec, list):
        folds = spec
    elif isinstance(spec, dict) and spec.get("mode") == "leave_one_out":
        labels = spec.get("labels", [])
        prefix = spec.get("prefix", "")
        folds = [
            {
                "name": Path(held_out).stem,
                "train": [
                    {"label": label, "prefix": prefix}
                    for label in labels if label != held_out
                ],
                "test": [{"label": held_out, "prefix": prefix}],
            }
            for held_out in labels
        ]
    else:
        raise ValueError("folds must be a list or {mode: leave_one_out, labels: [...]}.")

    for fold in folds:
        if not {"name", "train", "test"}.issubset(fold):
            raise ValueError(f"Each fold needs name, train, and test entries: {fold!r}")
    return folds


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gaze_to_vector(gaze):
    """Convert GazeHub ``(yaw, pitch)`` radians to its 3D unit-vector convention."""
    if isinstance(gaze, torch.Tensor):
        yaw, pitch = gaze.unbind(dim=-1)
        return torch.stack(
            (-torch.cos(pitch) * torch.sin(yaw), -torch.sin(pitch),
             -torch.cos(pitch) * torch.cos(yaw)),
            dim=-1,
        )

    gaze = np.asarray(gaze, dtype=np.float32)
    yaw, pitch = gaze[..., 0], gaze[..., 1]
    return np.stack(
        (-np.cos(pitch) * np.sin(yaw), -np.sin(pitch),
         -np.cos(pitch) * np.cos(yaw)),
        axis=-1,
    )


def angular_error_deg(prediction, target):
    """Per-sample angular error in degrees for torch tensors of shape ``[..., 2]``."""
    pred_vec = gaze_to_vector(prediction)
    target_vec = gaze_to_vector(target)
    cosine = (pred_vec * target_vec).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def angular_error(prediction, target):
    """Angular error in degrees for one NumPy-like ``(yaw, pitch)`` pair."""
    pred_vec = gaze_to_vector(prediction)
    target_vec = gaze_to_vector(target)
    cosine = np.clip(np.sum(pred_vec * target_vec, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def gaze_visual(image, gaze, color, scale=75, thickness=2):
    """Draw one GazeHub gaze direction on an OpenCV BGR image."""
    vector = gaze_to_vector(gaze)
    height, width = image.shape[:2]
    start = (width // 2, height // 2)
    end = (int(start[0] + vector[0] * scale), int(start[1] + vector[1] * scale))
    cv2.arrowedLine(image, start, end, color, thickness, tipLength=0.2)
    return image


def save_prediction_grid(paths, predictions, targets, output_path, max_items=16):
    """Save a compact BGR grid with red predictions and green ground truth."""
    tiles = []
    for path, prediction, target in zip(paths[:max_items], predictions[:max_items], targets[:max_items]):
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        gaze_visual(image, target, (0, 220, 0))
        gaze_visual(image, prediction, (0, 0, 230))
        error = angular_error(prediction, target)
        cv2.putText(image, f"{error:.2f} deg", (6, 216), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(image)

    if not tiles:
        return
    columns = 4
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * ((-len(tiles)) % columns))
    rows = [np.concatenate(tiles[index:index + columns], axis=1)
            for index in range(0, len(tiles), columns)]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.concatenate(rows, axis=0))


class Logger:
    """Timestamped text logger that also echoes to the terminal."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, message):
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        print(line, flush=True)
        self.file.write(line + "\n")

    def close(self):
        self.file.close()


def build_scheduler(optimizer, epochs, warmup_epochs, eta_min):
    import math
    from torch.optim.lr_scheduler import LambdaLR

    warmup_epochs = int(warmup_epochs or 0)
    cosine_epochs = max(epochs - warmup_epochs, 1)

    def schedule_for(base_lr):
        min_factor = eta_min / base_lr if base_lr > 0 else 0.0

        def lr_lambda(step):
            if warmup_epochs and step <= warmup_epochs:
                return 0.01 + 0.99 * (step / warmup_epochs)
            progress = min(max(step - warmup_epochs, 0), cosine_epochs) / cosine_epochs
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return lr_lambda

    return LambdaLR(optimizer, [schedule_for(group["lr"]) for group in optimizer.param_groups])


class TopKCheckpoints:
    """Keep only the checkpoints with the lowest monitor error."""

    def __init__(self, directory, k=5):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.k = k
        if self.k < 1:
            raise ValueError("save_top_k must be at least 1.")
        self.entries = []
        for path in self.directory.glob("epoch_*_mae_*.pt"):
            match = CHECKPOINT_PATTERN.match(path.name)
            if match:
                self.entries.append((float(match.group(1)), path))
        self.entries.sort(key=lambda item: item[0])
        for _, stale_path in self.entries[self.k:]:
            stale_path.unlink(missing_ok=True)
        self.entries = self.entries[:self.k]

    def update(self, payload, epoch, error):
        path = self.directory / f"epoch_{epoch:03d}_mae_{error:.3f}.pt"
        torch.save(payload, path)
        self.entries.append((float(error), path))
        self.entries.sort(key=lambda item: item[0])
        for _, stale_path in self.entries[self.k:]:
            stale_path.unlink(missing_ok=True)
        self.entries = self.entries[:self.k]
        return path, self.entries[0][1] == path

    @property
    def best_path(self):
        if not self.entries:
            raise RuntimeError("No checkpoint has been saved.")
        return self.entries[0][1]


def find_best_checkpoint(directory):
    candidates = []
    for path in Path(directory).glob("epoch_*_mae_*.pt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            candidates.append((float(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {directory}")
    return min(candidates, key=lambda item: item[0])[1]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def read_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def aggregate_experiment(experiment_dir, config):
    """Collect every fold's result.json under an experiment into one summary.

    Scans the whole experiment directory (not just the folds run this session),
    so cross-validation folds launched in separate processes still accumulate
    into a single, canonically-ordered summary with a mean over completed folds.
    """
    experiment_dir = Path(experiment_dir)
    folds = resolve_folds(config)
    results = []
    for fold in folds:
        result_path = experiment_dir / fold["name"] / "result.json"
        if result_path.is_file():
            results.append(read_json(result_path))
    maes = [result["test"]["mae"] for result in results]
    return {
        "dataset": config["dataset"],
        "experiment": experiment_dir.name,
        "completed": len(results),
        "total_folds": len(folds),
        "mean_test_mae": (sum(maes) / len(maes)) if maes else None,
        "folds": results,
    }
