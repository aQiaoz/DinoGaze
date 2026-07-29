"""Aggregate an experiment's per-fold results into a table and summary.json.

Useful when cross-validation folds were trained in separate processes: it scans
every ``save/<exp>/<fold>/result.json`` and reports the mean over completed folds.
"""

import argparse
from pathlib import Path

from utils import aggregate_experiment, load_config, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize DINOGaze cross-validation results.")
    parser.add_argument("--config", required=True, help="Dataset YAML used for training.")
    parser.add_argument("--exp", required=True, help="Experiment name under save/.")
    parser.add_argument("--save-root", default=None, help="Defaults to <project>/save.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    save_root = Path(args.save_root) if args.save_root else Path(__file__).resolve().parent / "save"
    experiment_dir = save_root / args.exp

    summary = aggregate_experiment(experiment_dir, config)
    if not summary["folds"]:
        print(f"No result.json found under {experiment_dir}")
        return

    print(f"{'fold':<14}{'MAE':>9}{'yaw':>9}{'pitch':>9}")
    print("-" * 41)
    for fold in summary["folds"]:
        test = fold["test"]
        print(f"{fold['fold']:<14}{test['mae']:>9.3f}{test['yaw_mae']:>9.3f}{test['pitch_mae']:>9.3f}")
    print("-" * 41)
    label = f"mean ({summary['completed']}/{summary['total_folds']})"
    print(f"{label:<14}{summary['mean_test_mae']:>9.3f}")

    write_json(experiment_dir / "summary.json", summary)
    print(f"\nWrote {experiment_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
