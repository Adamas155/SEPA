"""Real-image integration checks. Does not estimate representation quality."""
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sepa_plan_b.config import Config, Geometry
from sepa_plan_b.data import write_json
from sepa_plan_b.engine import train, load_checkpoint, source_hash
from sepa_plan_b.evaluation import probe, padding_diagnostic, spatial_probe, analyze
import torch


def equal_state(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/smoke.toml"))
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = Config.load(args.config)
    if config.model.name != "tiny" or config.training.steps > 10:
        parser.error("Use a tiny smoke config with at most ten steps")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path(args.output_root).resolve() if args.output_root else Path(config.training.output_root).parent / f"validation-{stamp}"
    if root.exists():
        parser.error("Validation output must be a new directory")
    config = replace(config, training=replace(config.training, output_root=str(root / "native")))
    result = {"purpose": "Engineering smoke only; not scientific performance evidence", "source_hash": source_hash(),
              "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              "runs": {}}

    def run(name, cfg, k=3, method="sepa", evaluate=False):
        print(f"\n{name}", flush=True)
        trained = train(cfg, k=k, method=method)
        result["runs"][name] = {"train": trained}
        if evaluate:
            result["runs"][name]["probe"] = probe(trained["checkpoint"])
        write_json(root / "summary.json", result)
        return trained

    native = {}
    for name, k, method in (("k0", 0, "sepa"), ("k3", 3, "sepa"), ("k6", 6, "sepa"), ("full", 0, "full")):
        native[name] = run(name, config, k, method, evaluate=True)
    result["stage1_analysis"] = analyze([result["runs"][name]["probe"]["output"] for name in native])
    assert result["stage1_analysis"]["status"] == "INSUFFICIENT"
    result["low_shot"] = probe(native["k3"]["checkpoint"], label_fraction=0.1)
    result["spatial"] = spatial_probe(native["k3"]["checkpoint"], limit=0)

    resumed_cfg = replace(config, training=replace(config.training, output_root=str(root / "resumed")))
    interrupted = train(resumed_cfg, k=3, stop_after=2)
    resumed = train(resumed_cfg, k=3, resume=True)
    a, b = load_checkpoint(native["k3"]["checkpoint"]), load_checkpoint(resumed["checkpoint"])
    result["resume_model_bitwise_equal"] = equal_state(a["model"], b["model"])
    result["resume_optimizer_bitwise_equal"] = equal_state(a["optimizer"], b["optimizer"])
    assert result["resume_model_bitwise_equal"] and result["resume_optimizer_bitwise_equal"]
    del a, b

    for mode in ("reflect", "replicate", "constant"):
        cfg = replace(config, geometry=Geometry("pad74", mode),
                      training=replace(config.training, output_root=str(root / "padded")))
        trained = run(f"pad74-{mode}", cfg, evaluate=True)
        result["runs"][f"pad74-{mode}"]["padding"] = padding_diagnostic(trained["checkpoint"], limit=0)
    for relation in ("undirected", "directed", "aggregation"):
        cfg = replace(config, model=replace(config.model, relation=relation, relation_weight=0.5),
                      training=replace(config.training, output_root=str(root / "relations")))
        run(relation, cfg)

    # Exercise the actual 12-layer, 384-dimensional ViT-S/16 and both attention paths.
    # BF16 is checked here when supported; the tiny checks above use FP32.
    precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp32"
    cfg = replace(config, model=replace(config.model, name="vit_small", predictor_dim=192,
                                       predictor_depth=2, predictor_heads=6),
                  training=replace(config.training, steps=2, warmup_steps=0, batch_size=2,
                                   precision=precision, output_root=str(root / "vit_small")),
                  probe=replace(config.probe, batch_size=2))
    run("vit_small_sepa", cfg, evaluate=True)
    run("vit_small_full", cfg, k=0, method="full", evaluate=True)
    result["passed"] = True
    write_json(root / "summary.json", result)
    print(json.dumps({"summary": str(root / "summary.json"), "passed": True}, indent=2), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
