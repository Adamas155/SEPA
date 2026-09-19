"""Plan and execute paired runs without mixing argument parsing and training."""

from pathlib import Path


def integer_list(value, label, parser):
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError:
        parser.error(f"{label} must be a comma-separated list of integers")
    if len(set(values)) != len(values):
        parser.error(f"{label} must be unique")
    return values


def plan(args, config, parser):
    from ..geometry import VALID_K

    seeds = integer_list(args.seeds, "Seeds", parser)
    ks = integer_list(args.ks, "k values", parser)
    if any(seed < 0 for seed in seeds) or any(k not in VALID_K for k in ks):
        parser.error("Seeds must be nonnegative and k must be 0,2,3,4,5,6")
    if config.model.relation != "none":
        parser.error("Stage 1 matrix requires relation=none")
    if args.include_full and config.geometry.mode == "pad74":
        parser.error("Full-image reference uses a native geometry; plan it separately")
    arms = [("sepa", k) for k in ks]
    if args.include_full:
        arms.append(("full", 0))
    jobs = [
        {"seed": seed, "method": method, "k": k} for method, k in arms for seed in seeds
    ]
    return {
        "config": config.to_dict(),
        "jobs": jobs,
        "n_runs": len(jobs),
        "executed": False,
    }


def execute(config, jobs, args):
    from ..data import manifests, write_json
    from ..engine import identity, run_directory, train
    from ..evaluation import probe

    train_data, val_data = manifests(config)
    outputs = []
    for job in jobs:
        meta = identity(
            config, job["seed"], job["k"], job["method"], train_data, val_data
        )
        exists = (run_directory(config, meta) / "latest.pt").exists()
        training = train(config, **job, resume=args.resume and exists)
        evaluation = probe(training["checkpoint"])
        outputs.append({"train": training, "probe": evaluation})
        write_json(Path(args.output).with_suffix(".progress.json"), outputs)
    return outputs


def run(args, parser):
    from ..config import Config
    from ..data import write_json

    config = Config.load(args.config)
    result = plan(args, config, parser)
    write_json(args.output, result)
    if args.execute:
        outputs = execute(config, result["jobs"], args)
        result.update(executed=True, results=outputs)
        write_json(args.output, result)
    return result
