from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .config import Config
from .data import manifests, prepare, write_json
from .engine import identity, run_directory, train
from .evaluation import analyze, padding_diagnostic, probe, spatial_probe
from .geometry import VALID_K


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent SEPA Plan B experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Create immutable class-mapped image manifests")
    p.add_argument("--train-root", required=True)
    p.add_argument("--val-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--class-list")
    p.add_argument("--classes", type=int, default=100)
    p.add_argument("--limit-per-class", type=int, default=0, help="Smoke only: cap each class")

    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--method", choices=["sepa", "full"], default="sepa")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int, help="Stop at this completed step without shortening the configured schedule")

    p = sub.add_parser("probe")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--label-fraction", type=float)
    p.add_argument("--train-manifest", help="Optional transfer task")
    p.add_argument("--val-manifest", help="Optional transfer task")
    p.add_argument("--device", choices=["cpu", "cuda"])

    for name in ("padding", "spatial"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--output")
        p.add_argument("--limit", type=int, default=256 if name == "padding" else 1000)
        p.add_argument("--device", choices=["cpu", "cuda"])

    p = sub.add_parser("analyze")
    p.add_argument("--results", nargs="+", required=True, help="Paths or glob patterns for probe.json")
    p.add_argument("--output", required=True)
    p.add_argument("--delta-pp", type=float, default=1.0)

    p = sub.add_parser("matrix", help="Plan a Stage 1 matrix; add --execute to actually train/probe")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--ks", default="0,3,6")
    p.add_argument("--include-full", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output", default="matrix_plan.json")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.train_root, args.val_root, args.out, args.class_list, args.classes, args.limit_per_class)
    elif args.command == "train":
        result = train(Config.load(args.config), seed=args.seed, k=args.k, method=args.method,
                       resume=args.resume, stop_after=args.stop_after)
    elif args.command == "probe":
        result = probe(args.checkpoint, output=args.output, limit=args.limit, label_fraction=args.label_fraction,
                       train_manifest=args.train_manifest, val_manifest=args.val_manifest, device=args.device)
    elif args.command in {"padding", "spatial"}:
        fn = padding_diagnostic if args.command == "padding" else spatial_probe
        result = fn(args.checkpoint, output=args.output, limit=args.limit, device=args.device)
    elif args.command == "analyze":
        paths = sorted({p for pattern in args.results for p in glob.glob(pattern)})
        result = analyze(paths, args.delta_pp)
        write_json(args.output, result)
    else:
        config = Config.load(args.config)
        seeds = [int(s) for s in args.seeds.split(",")]
        ks = [int(k) for k in args.ks.split(",")]
        if len(set(seeds)) != len(seeds) or len(set(ks)) != len(ks):
            parser.error("Matrix seeds and k values must be unique")
        if any(s < 0 for s in seeds) or any(k not in VALID_K for k in ks):
            parser.error("Seeds must be nonnegative and k must be 0,2,3,4,5,6")
        if config.model.relation != "none":
            parser.error("Stage 1 matrix requires relation=none")
        if args.include_full and config.geometry.mode == "pad74":
            parser.error("Full-image reference uses a native geometry; plan it separately")
        arms = [("sepa", k) for k in ks] + ([("full", 0)] if args.include_full else [])
        jobs = [{"seed": s, "method": method, "k": k} for method, k in arms for s in seeds]
        result = {"config": config.to_dict(), "jobs": jobs, "n_runs": len(jobs), "executed": False}
        write_json(args.output, result)
        if args.execute:
            outputs = []
            train_data, val_data = manifests(config)
            for job in jobs:
                meta = identity(config, job["seed"], job["k"], job["method"], train_data, val_data)
                exists = (run_directory(config, meta) / "latest.pt").exists()
                training = train(config, **job, resume=args.resume and exists)
                evaluation = probe(training["checkpoint"])
                outputs.append({"train": training, "probe": evaluation})
                write_json(Path(args.output).with_suffix(".progress.json"), outputs)
            result.update(executed=True, results=outputs)
            write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
