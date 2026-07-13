from __future__ import annotations

import argparse

from bathymetry_experiments.analysis.scatter import run_scatter
from bathymetry_experiments.training.experiment import ALL_MODELS, train_model, run_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bathymetry_experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train and evaluate one self-contained model.")
    train.add_argument("--model", choices=ALL_MODELS, required=True)
    train.add_argument("--config", type=str, default="configs/agia_napa.yaml")
    train.add_argument("--output-dir", type=str, default=None)
    train.set_defaults(func=_train)

    exp = sub.add_parser("experiment", help="Train and evaluate multiple models.")
    exp.add_argument("--models", nargs="*", choices=ALL_MODELS, default=[])
    exp.add_argument("--config", type=str, default="configs/agia_napa.yaml")
    exp.add_argument("--output-dir", type=str, default=None)
    exp.set_defaults(func=_experiment)

    scatter = sub.add_parser("scatter", help="Create a scatter plot from *_pred.npy and *_gt.npy files.")
    scatter.add_argument("--pred-dir", type=str, required=True)
    scatter.add_argument("--sample-id", type=str, default=None)
    scatter.add_argument("--out", type=str, default=None)
    scatter.add_argument("--title", type=str, default="Bathymetry Prediction Scatter")
    scatter.add_argument("--max-points", dest="max_points", type=int, default=200_000)
    scatter.add_argument("--seed", type=int, default=0)
    scatter.add_argument("--no-exclude-default", dest="no_exclude_default", action="store_true")
    scatter.set_defaults(func=run_scatter)
    return parser


def _train(args) -> None:
    run_dir = train_model(args.model, args.config, args.output_dir)
    print(run_dir)


def _experiment(args) -> None:
    run_dirs = run_all(args.config, args.models, args.output_dir)
    for run_dir in run_dirs:
        print(run_dir)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
