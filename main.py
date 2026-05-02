"""Top-level project CLI."""

from __future__ import annotations

import argparse

from src.experiments.benchmark import generate_report_assets
from src.train import config_from_args as train_config_from_args
from src.train import run_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build the root CLI parser."""
    parser = argparse.ArgumentParser(description="ogbl-collab project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run an experiment")
    train_parser.add_argument("--method", choices=("common_neighbors", "mlp", "gcn"), default="common_neighbors")
    train_parser.add_argument("--scale", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--hidden-channels", type=int)
    train_parser.add_argument("--num-layers", type=int)
    train_parser.add_argument("--dropout", type=float)
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument("--no-save", action="store_true")
    train_parser.add_argument("--disable-tie-breaker", action="store_true")
    train_parser.add_argument("--directed", action="store_true")

    subparsers.add_parser("assets", help="Generate summary CSV and plots from saved results")

    return parser


def main() -> None:
    """Dispatch top-level CLI commands."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        config = train_config_from_args(args)
        result = run_experiment(config)
        print(f"method_name={result['method_name']}")
        print(f"dataset_scale={result['dataset_scale']}")
        print(f"status={result['status']}")
        if "result_path" in result:
            print(f"result_path={result['result_path']}")
        print(f"metrics={result.get('metrics', {})}")
    elif args.command == "assets":
        assets = generate_report_assets()
        print(f"result_count={assets['result_count']}")
        print(f"summary_csv={assets['summary_csv']}")
        print(f"plots={assets['plots']}")


if __name__ == "__main__":
    main()
