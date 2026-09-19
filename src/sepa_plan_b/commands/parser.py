"""Argument definitions only: safe to use on a machine without training libraries."""

import argparse


def _prepare(subparsers):
    parser = subparsers.add_parser(
        "prepare", help="Prepare class-mapped image manifests"
    )
    for name in ("train-root", "val-root", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--class-list")
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument(
        "--limit-per-class", type=int, default=0, help="Smoke-test image limit"
    )


def _train(subparsers):
    parser = subparsers.add_parser(
        "train", help="Train one SEPA or matched-control encoder"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--method", choices=("sepa", "full"), default="sepa")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after", type=int, help="Stop without shortening the schedule"
    )


def _evaluation(subparsers):
    descriptions = {
        "probe": "Fit a frozen linear classifier and evaluate kNN",
        "spatial": "Run frozen spatial probes and shortcut controls",
        "padding": "Compare padding with a fixed encoder and classifier",
    }
    for name, description in descriptions.items():
        parser = subparsers.add_parser(name, help=description)
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--output")
        parser.add_argument(
            "--limit",
            type=int,
            default={"probe": 0, "spatial": 1000, "padding": 256}[name],
        )
        parser.add_argument("--device", choices=("cpu", "cuda"))
        if name == "probe":
            parser.add_argument("--label-fraction", type=float)
            parser.add_argument(
                "--train-manifest", help="Optional transfer-task training manifest"
            )
            parser.add_argument(
                "--val-manifest", help="Optional transfer-task validation manifest"
            )


def _analysis(subparsers):
    parser = subparsers.add_parser(
        "analyze", help="Summarize compatible Stage 1 probe results"
    )
    parser.add_argument(
        "--results", nargs="+", required=True, help="Paths or glob patterns"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--delta-pp", type=float, default=1.0)


def _matrix(subparsers):
    parser = subparsers.add_parser(
        "matrix", help="Plan a run matrix; training requires --execute"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--seeds", default="0", help="Comma-separated seeds; default: 0"
    )
    parser.add_argument(
        "--ks", default="0,3,6", help="Comma-separated permutation settings"
    )
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", default="matrix_plan.json")


def _experiments(subparsers):
    from .experiments import EXPERIMENTS

    parser = subparsers.add_parser(
        "experiment", help="Find and launch recorded server workflows"
    )
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="List workflows without importing training code")
    for action in ("show", "run"):
        item = actions.add_parser(
            action,
            help="Show workflow details"
            if action == "show"
            else "Launch a workflow explicitly",
        )
        item.add_argument("name", choices=tuple(EXPERIMENTS))
        if action == "run":
            item.add_argument(
                "--dry-run",
                action="store_true",
                help="Print the command without running it",
            )
            item.epilog = "Pass script options after --, e.g. experiment run v1-100 --dry-run -- --workers 2"


def build_parser():
    parser = argparse.ArgumentParser(
        description="SEPA training, evaluation and experiment workflows"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for register in (_prepare, _train, _evaluation, _analysis, _matrix, _experiments):
        register(subparsers)
    return parser
