"""CLI for a separate, frozen-encoder spatial evaluation experiment."""

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "prepare", "smoke", "run"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True, help="New independent output directory")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--release-features", action="store_true",
                        help="Release only this run's feature arrays after each model; retain heads and evidence")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--test-receipt", type=Path, help="Successful unit-test receipt for this exact experiment code")
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from .pipeline import execute, resolved_config
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = resolved_config(args.config, device=args.device)
    if config["extraction"]["device"] == "cuda" and not torch.cuda.is_available() and args.action not in {"audit", "prepare"}:
        parser.error("CUDA requested but unavailable; choose --device cpu explicitly")
    manifest = execute(config, args.output, action=args.action, release_features=args.release_features,
                       test_receipt=args.test_receipt)
    print(json.dumps({"status": manifest["status"], "output": str(args.output.resolve()),
                      "counts": manifest["counts"], "seconds": manifest["seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
