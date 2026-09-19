"""Thin command handlers; numerical work stays in the existing runtime modules."""

import glob


def prepare(args, parser):
    from ..data import prepare as prepare_data

    return prepare_data(
        args.train_root,
        args.val_root,
        args.out,
        args.class_list,
        args.classes,
        args.limit_per_class,
    )


def train(args, parser):
    from ..config import Config
    from ..engine import train as train_encoder

    return train_encoder(
        Config.load(args.config),
        seed=args.seed,
        k=args.k,
        method=args.method,
        resume=args.resume,
        stop_after=args.stop_after,
    )


def probe(args, parser):
    from ..evaluation import probe as evaluate

    return evaluate(
        args.checkpoint,
        output=args.output,
        limit=args.limit,
        label_fraction=args.label_fraction,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        device=args.device,
    )


def diagnostic(args, parser):
    from ..evaluation import padding_diagnostic, spatial_probe

    evaluate = padding_diagnostic if args.command == "padding" else spatial_probe
    return evaluate(
        args.checkpoint, output=args.output, limit=args.limit, device=args.device
    )


def analyze(args, parser):
    from ..data import write_json
    from ..evaluation import analyze as summarize

    paths = sorted({path for pattern in args.results for path in glob.glob(pattern)})
    result = summarize(paths, args.delta_pp)
    write_json(args.output, result)
    return result


def matrix(args, parser):
    from .matrix import run

    return run(args, parser)


HANDLERS = {
    "prepare": prepare,
    "train": train,
    "probe": probe,
    "padding": diagnostic,
    "spatial": diagnostic,
    "analyze": analyze,
    "matrix": matrix,
}


def dispatch(args, parser):
    return HANDLERS[args.command](args, parser)
