"""Only frozen extraction and lightweight probes; no pretraining entry points."""

import copy
import gc
from pathlib import Path
import shutil
import shlex
import sys
import time

import numpy as np
import torch

from . import PROJECT_ROOT
from .common import PREPROCESS, code_identity, digest, file_hash, read_json, write_json
from .data import TileDataset, decoded_groups, load_sources, make_splits
from .encoders import load_encoder, state_digest
from .features import (PixelFeatures, cache_identity, extract_cached, save_design,
                       save_fitted)
from .metrics import (adjacency_metrics, cosine_pair_scores, retrieval_metrics,
                      select_thresholds)
from .probe import fit_probe, predict_scores
from .tasks import make_design, ordered_pairs

LIMITATIONS = [
    "Only content-only independent local readout is measured; all nine tiles are visible, so this is not missing+shuffle recovery.",
    "I-JEPA uses the same fixed top-left 5x5 PE window for every tile; this changes the input distribution of the globally pretrained encoder.",
    "All learned heads have at least three probe seeds, but all pretraining seeds are zero. Head-fit variation is not pretraining variation.",
    "Adjacency and retrieval share the same trained head and are complementary readouts, not independent mechanism evidence.",
    "Border masking also removes genuine visual content and introduces distribution shift; a drop is not proof of a border-only shortcut.",
    "Beating border-only features does not prove the encoder ignores borders.",
    "Cosine scores have no intrinsic direction discrimination.",
    "The requested linear concatenation head is additive; fixed-anchor candidate ranking cannot model pairwise interactions. Optional fixed-width MLP results are separate.",
    "No confidence interval is computed. No pair is treated as an independent image, and no cross-pretraining-seed significance is inferred.",
]


def resolved_config(path, *, device=None):
    config = read_json(path)
    if config.get("protocol_version") != 1:
        raise ValueError("Unknown spatial readout protocol")
    config = copy.deepcopy(config)
    seeds = config["probe_seeds"]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("At least three distinct nonnegative probe seeds are required")
    heads = config["heads"]
    if not heads or len(set(heads)) != len(heads) or set(heads) - {"linear", "mlp"} or "linear" not in heads:
        raise ValueError("Linear is required; the only optional head is fixed-width mlp")
    expected = {"ijepa-e100", "v1-k0-e100", "v1-k3-e100", "ijepa-e25", "v2-k0-e25", "v2-k3-e25"}
    if {x["name"] for x in config["checkpoints"]} != expected or len(config["checkpoints"]) != 6:
        raise ValueError("Keep the six predeclared checkpoint entries, even when unavailable")
    def absolute(value):
        if value is None:
            return None
        p = Path(value).expanduser()
        return str((PROJECT_ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    for key in ("train_manifest", "test_manifest", "train_root", "test_root", "existing_split",
                "fallback_existing_split", "duplicate_exclusions"):
        if key in config["data"]:
            config["data"][key] = absolute(config["data"][key])
    config["extraction"]["upstream_dir"] = absolute(config["extraction"]["upstream_dir"])
    for spec in config["checkpoints"]:
        spec["path"] = absolute(spec["path"])
    if device is not None:
        config["extraction"]["device"] = device
    if config["data"]["expected_classes"] != 100 or config["data"]["expected_test_images"] != 5000:
        raise ValueError("Production/smoke sources must be the recorded ImageNet-100 with 5000 validation images")
    config["source_config"] = {"path": str(Path(path).resolve()), "file_hash": file_hash(path)}
    return config


def command_examples(config_path):
    prefix = f"python -m experiments.spatial_readout.run"
    return {
        "unit_tests": "python -m unittest discover -s experiments/spatial_readout/tests -v",
        "audit": f"{prefix} audit --config {config_path} --output experiments/spatial_readout/outputs/audit-new",
        "engineering_smoke": f"{prefix} smoke --config {config_path} --output experiments/spatial_readout/outputs/smoke-new",
        "formal_not_run_by_smoke": f"{prefix} run --config {config_path} --release-features --output experiments/spatial_readout/outputs/full-new",
        "prepare_only": f"{prefix} prepare --config {config_path} --output experiments/spatial_readout/outputs/splits-new",
    }


def resource_plan(counts, config, *, available_models=6, free_bytes=None):
    n = sum(counts.values())
    per_encoder = (n + counts["test"]) * 9 * 384 * 4
    pixel = (n + counts["test"]) * 9 * (54 + 192) * 4
    epochs = config["probe"]["epochs"]
    seeds, heads, rates = len(config["probe_seeds"]), len(config["heads"]), len(config["probe"]["learning_rates"])
    fits = (available_models + 2) * seeds * heads * rates
    return {
        "image_counts": counts, "encoder_models": available_models,
        "full_fp32_encoder_cache_bytes_each": per_encoder,
        "all_encoder_and_pixel_cache_bytes": available_models * per_encoder + pixel,
        "sequential_peak_feature_bytes_approx": max(per_encoder, pixel),
        "reserve_bytes_recommended": 2 * 1024**3,
        "free_bytes_at_planning": free_bytes,
        "clean_encoder_tiles_per_model": n * 9,
        "border_masked_encoder_tiles_per_model": counts["test"] * 9,
        "test_pairs_per_model_condition": counts["test"] * 72,
        "test_retrieval_queries_per_model_condition": counts["test"] * 24,
        "maximum_head_fit_recipes": fits,
        "maximum_pair_training_presentations": fits * epochs * counts["train"] * 72,
        "max_epochs_per_recipe": epochs,
        "precision": "FP32 features, FP32 head; no encoder optimizer",
        "time_limit": "Full runtime is not inferred from tiny smoke throughput; extraction and dev sorting dominate at full scale.",
        "storage_policy": "--release-features deletes only newly generated feature .npy caches after each model; receipts, fitted heads, logits and metrics remain",
    }


def prepare_data(config, output, *, smoke=False):
    dc = config["data"]
    train, test, sources = load_sources(
        dc["train_manifest"], dc["test_manifest"], train_root=dc.get("train_root"),
        test_root=dc.get("test_root"), expected_classes=dc["expected_classes"],
        expected_test_images=dc["expected_test_images"],
    )
    for role, source in (("train", train), ("test", test)):
        if source["fingerprint"] != dc[f"expected_{role}_fingerprint"]:
            raise ValueError(f"{role} source differs from the recorded ImageNet-100 manifest")
        if len(source["records"]) != dc[f"expected_{role}_images"]:
            raise ValueError(f"{role} source image count differs from the recorded dataset")
    existing = next((p for p in (dc.get("existing_split"), dc.get("fallback_existing_split"))
                     if p and Path(p).is_file()), None)
    groups = None
    dedup = {"method": "existing verified manifest file-content identities",
             "limitations": "Engineering subset only: decoded-RGB and perceptual duplicate scan not performed."}
    if not smoke:
        print("prepare: verifying decoded image-content groups (no network downloads)", flush=True)
        groups, dedup = decoded_groups((train, test), output / "content_groups.json", workers=dc["dedup_workers"])
    splits = make_splits(
        train, test, seed=dc["split_seed"], dev_fraction=dc["dev_fraction"],
        content_groups=groups, existing_split=existing,
        smoke_limits=tuple(config["smoke"]["image_limits"]) if smoke else None,
    )
    splits["sources"] = sources
    splits["dedup_provenance"] = dedup
    splits["existing_split"] = {"path": existing, "file_hash": file_hash(existing)} if existing else None
    excluded = dc.get("duplicate_exclusions")
    splits["prior_duplicate_exclusions"] = (
        {"path": excluded, "file_hash": file_hash(excluded), "record": read_json(excluded)}
        if excluded and Path(excluded).is_file() else
        {"available": False, "fallback": "original manifest grouping plus the declared content verification"})
    if not smoke and len(splits["manifests"]["test"]["records"]) != 5000:
        raise ValueError("Formal test must retain all 5000 official validation records")
    write_json(output / "split_manifest.json", splits)
    designs, design_hashes = {}, {}
    for role, manifest in splits["manifests"].items():
        keys = [digest([role, r["sha256"], r["path"], r["source_index"]]) for r in manifest["records"]]
        designs[role] = make_design(keys, seed=dc["design_seed"])
        design_hashes[role] = save_design(output / f"design-{role}.npz", designs[role])
    write_json(output / "design_manifest.json", {
        "seed": dc["design_seed"], "hashes": design_hashes,
        "pairs_per_image": 72, "positives_per_direction": 6,
        "queries_per_image": 24, "candidates_per_query": 8,
        "randomization": "tile containers, candidates and queries, independently of encoder and probe seed",
    })
    return splits, designs, design_hashes


def audit_checkpoints(config, output):
    entries = []
    for spec in config["checkpoints"]:
        print(f"audit: {spec['name']}", flush=True)
        try:
            model = load_encoder(spec, device="cpu", upstream_dir=config["extraction"]["upstream_dir"])
            entries.append({**model.manifest, "status": "verified"})
            del model
        except (FileNotFoundError, ValueError) as error:
            entries.append({**spec, "status": "missing" if isinstance(error, FileNotFoundError) else "invalid",
                            "error": str(error), "checkpoint_verified": False})
        gc.collect()
        write_json(output / "checkpoint_manifest.json", entries)
    return entries


def validate_local_readout(model, dataset, device):
    pixels = dataset[0].to(device)
    permutation = torch.tensor([5, 0, 8, 2, 4, 1, 7, 3, 6], device=device)
    with torch.inference_mode():
        original = model(pixels)
        reordered = model(pixels[permutation])[torch.argsort(permutation)]
        if not torch.allclose(original, reordered, atol=1e-6, rtol=1e-5):
            raise AssertionError("Tile container indices changed content features")
        # Coordinates are deliberately label-side metadata, never function arguments.
        records = [{"pixels": pixels[0], "row": row, "col": col, "tile_id": row * 3 + col}
                   for row in range(3) for col in range(3)]
        repeated = model(torch.stack([r["pixels"] for r in records]))
        if not torch.allclose(repeated, repeated[:1].expand_as(repeated), atol=1e-6, rtol=1e-5):
            raise AssertionError("Identical tile pixels yielded position-dependent features")
    return {"container_reordering": True, "coordinate_metadata_not_forwarded": True,
            "max_reorder_absolute_error": float((original - reordered).abs().max()),
            "encoder_frozen": all(not p.requires_grad for p in model.parameters()),
            "encoder_eval": not model.training}


def flatten_metrics(scores, design, thresholds, metadata):
    adjacency = adjacency_metrics(scores, design["labels"], thresholds)
    retrieval = retrieval_metrics(scores, design)
    rows = []
    for task, values, directions, metrics in (
        ("adjacency", adjacency, ("V", "H", "macro"), ("ap", "auroc", "balanced_accuracy")),
        ("retrieval", retrieval, ("down", "up", "right", "left", "macro"), ("recall_at_1", "mrr")),
    ):
        for direction in directions:
            for metric in metrics:
                rows.append({**metadata, "task": task, "direction": direction,
                             "metric": metric, "value": float(values[direction][metric])})
    return rows


def check_head_reordering(fitted, features, device):
    order = np.array([5, 0, 8, 2, 4, 1, 7, 3, 6])
    original = predict_scores(fitted, np.asarray(features[:1]), device=device)
    moved = predict_scores(fitted, np.asarray(features[:1])[:, order], device=device)
    pairs = ordered_pairs()
    lookup = {tuple(pair): i for i, pair in enumerate(pairs)}
    expected = original[:, [lookup[(int(order[a]), int(order[b]))] for a, b in pairs]]
    if not np.allclose(moved, expected, atol=1e-6, rtol=1e-5):
        raise AssertionError("Tile container order changed a pair prediction")


def evaluate_one(model, name, group, config, output, splits, designs, design_hashes,
                 *, smoke=False, release_features=False):
    device = config["extraction"]["device"]
    directory = output / name
    directory.mkdir()
    family = model.manifest["family"]
    dimension = 384 if family != "pixel_control" else model.dimension
    settings = config["extraction"]
    receipts = {}
    def features(role, condition="clean"):
        manifest = splits["manifests"][role]
        dataset = TileDataset(manifest, designs[role]["tile_orders"], condition=condition)
        identity = cache_identity(model_manifest=model.manifest, split_manifest=manifest,
                                  design_hash=design_hashes[role], condition=condition)
        array, receipt = extract_cached(
            model, dataset, directory / f"{role}-{condition}.npy", identity,
            dimension=dimension, device=device, batch_images=settings["batch_images"],
            tile_batch_size=settings["tile_batch_size"], workers=settings["workers"])
        receipts[f"{role}-{condition}"] = receipt
        return array
    checks = validate_local_readout(
        model, TileDataset(splits["manifests"]["train"], designs["train"]["tile_orders"]), device)
    before = state_digest(model)
    train, dev = features("train"), features("dev")
    fitted_heads = []
    for head in config["heads"]:
        pc = {**config["probe"], "head_kind": head}
        if smoke:
            pc.update(epochs=config["smoke"]["epochs"], patience=config["smoke"]["patience"])
        for seed in config["probe_seeds"]:
            print(f"fit: {name} {head} probe_seed={seed}", flush=True)
            fitted = fit_probe(train, designs["train"]["labels"], dev, designs["dev"]["labels"],
                               seed=seed, config=pc, device=device,
                               progress=lambda r: print(
                                   f"probe: {name} {head} s{seed} epoch={r['epoch']} "
                                   f"dev_AP={r['dev_macro_ap']:.6f} seconds={r['epoch_seconds']:.2f}",
                                   flush=True))
            fitted["training_identity"] = {
                "train_cache": receipts["train-clean"]["file_hash"],
                "dev_cache": receipts["dev-clean"]["file_hash"],
                "train_design": design_hashes["train"], "dev_design": design_hashes["dev"],
                "config": digest(config), "test_used_for_selection": False}
            path = directory / f"head-{head}-s{seed}.pt"
            save_fitted(path, fitted)
            summary = {k: v for k, v in fitted.items() if k != "state_dict"}
            summary.update(head_file=str(path), head_file_hash=file_hash(path))
            write_json(path.with_suffix(".json"), summary)
            fitted_heads.append((head, seed, fitted))
    cosine_thresholds = (
        select_thresholds(cosine_pair_scores(dev), designs["dev"]["labels"])
        if family != "pixel_control" else None)
    # Seal all selections before opening test pixel/feature files.
    write_json(directory / "selection_complete.json", {
        "test_used": False, "probe_seeds": config["probe_seeds"], "config": digest(config),
        "cosine_dev_thresholds": cosine_thresholds,
    })
    if state_digest(model) != before or any(p.grad is not None for p in model.parameters()):
        raise AssertionError("Encoder changed while fitting probe heads")
    checks.update(encoder_unchanged_during_probe=True, encoder_state_before=before,
                  encoder_state_after_probe=state_digest(model), head_selection_sealed_before_test=True)
    del train, dev
    rows = []
    meta = {
        "comparison_group": group, "model": name, "family": family,
        "pretraining_epochs": model.manifest.get("epochs"),
        "pretraining_seed": model.manifest.get("pretraining_seed"),
        "feature_dim": dimension, "n_images": len(splits["manifests"]["test"]["records"]),
        "engineering_only": smoke,
    }
    for condition in ("clean", "border_masked"):
        test = features("test", condition)
        for head, seed, fitted in fitted_heads:
            scores = predict_scores(fitted, test, device=device,
                                    batch_images=config["probe"]["batch_images"])
            check_head_reordering(fitted, test, device)
            metadata = {**meta, "head": head, "probe_seed": seed, "condition": condition,
                        "parameter_count": fitted["parameter_count"]}
            rows.extend(flatten_metrics(scores, designs["test"], fitted["thresholds"], metadata))
            np.save(directory / f"scores-{head}-s{seed}-{condition}.npy", scores, allow_pickle=False)
        if cosine_thresholds is not None:
            scores = cosine_pair_scores(test)
            metadata = {**meta, "head": "cosine", "probe_seed": None, "condition": condition,
                        "parameter_count": 0}
            rows.extend(flatten_metrics(scores, designs["test"], cosine_thresholds, metadata))
        del test
    checks["head_container_reordering"] = True
    checks["encoder_state_after_evaluation"] = state_digest(model)
    if checks["encoder_state_after_evaluation"] != before:
        raise AssertionError("Encoder parameters changed during evaluation")
    if model.manifest.get("checkpoint_verified"):
        if file_hash(model.manifest["path"]) != model.manifest["file_hash"]:
            raise AssertionError("Source checkpoint file changed during evaluation")
        checks["checkpoint_file_unchanged"] = True
    write_json(directory / "checks.json", checks)
    write_json(directory / "metrics.json", rows)
    if release_features:
        gc.collect()
        for cache in directory.glob("*-clean.npy"):
            if cache.name.startswith(("train-", "dev-", "test-")):
                cache.unlink()
        (directory / "test-border_masked.npy").unlink()
        write_json(directory / "feature_release.json", {
            "released": True, "retained": "cache identities, receipts, fitted probes, test logits, metrics",
            "reason": "explicit --release-features; sequential storage bound"})
    return rows, checks


def execute(config, output, *, action, release_features=False, test_receipt=None):
    from .reporting import write_metrics, write_report

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    smoke = action == "smoke"
    manifest = {"status": "running", "engineering_only": smoke, "config": config,
                "code_identity": code_identity(), "preprocess": PREPROCESS,
                "limitations": LIMITATIONS, "checkpoints": [], "checks": {}, "counts": {}}
    if test_receipt is not None:
        receipt = read_json(test_receipt)
        if not receipt.get("successful") or receipt.get("code_identity") != code_identity():
            raise ValueError("Unit-test receipt is unsuccessful or belongs to other experiment code")
        manifest["unit_tests"] = {"path": str(Path(test_receipt).resolve()),
                                  "file_hash": file_hash(test_receipt), **receipt}
    manifest["executed_command"] = shlex.join([sys.executable, "-m", "experiments.spatial_readout.run", *sys.argv[1:]])
    write_json(output / "configuration.json", config)
    rows = []
    try:
        if action == "audit":
            manifest["checkpoints"] = audit_checkpoints(config, output)
            manifest["status"] = "audit_only"
        else:
            splits, designs, design_hashes = prepare_data(config, output, smoke=smoke)
            counts = {role: len(m["records"]) for role, m in splits["manifests"].items()}
            manifest["counts"] = counts
            manifest["split_manifest_hash"] = file_hash(output / "split_manifest.json")
            manifest["design_hashes"] = design_hashes
            # Full-data plan is based on known records, not the engineering subset.
            if smoke:
                full = make_splits(*load_sources(
                    config["data"]["train_manifest"], config["data"]["test_manifest"],
                    train_root=config["data"].get("train_root"), test_root=config["data"].get("test_root"))[:2],
                    seed=config["data"]["split_seed"], dev_fraction=config["data"]["dev_fraction"])
                full_counts = {role: len(m["records"]) for role, m in full["manifests"].items()}
                del full
            else:
                full_counts = counts
            manifest["resource_plan"] = resource_plan(full_counts, config, free_bytes=shutil.disk_usage(output).free)
            manifest["resource_plan"]["counts_may_change_after_decoded_rgb_dedup"] = smoke
            if action == "prepare":
                manifest["status"] = "prepared_only"
            else:
                for spec in config["checkpoints"]:
                    print(f"load: {spec['name']}", flush=True)
                    try:
                        model = load_encoder(spec, config["extraction"]["device"],
                                             upstream_dir=config["extraction"]["upstream_dir"])
                    except (FileNotFoundError, ValueError) as error:
                        manifest["checkpoints"].append({**spec, "status": "missing" if isinstance(error, FileNotFoundError) else "invalid",
                                                        "error": str(error), "checkpoint_verified": False})
                        write_json(output / "checkpoint_manifest.json", manifest["checkpoints"])
                        continue
                    manifest["checkpoints"].append({**model.manifest, "status": "verified"})
                    write_json(output / "checkpoint_manifest.json", manifest["checkpoints"])
                    scores, checks = evaluate_one(
                        model, spec["name"], spec["group"], config, output,
                        splits, designs, design_hashes, smoke=smoke, release_features=release_features)
                    rows.extend(scores)
                    manifest["checks"][spec["name"]] = checks
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                for name in ("color_lowfreq", "border_only"):
                    model = PixelFeatures(name).to(config["extraction"]["device"]).eval()
                    scores, checks = evaluate_one(
                        model, name, "reference", config, output, splits, designs, design_hashes,
                        smoke=smoke, release_features=release_features)
                    rows.extend(scores)
                    manifest["checks"][name] = checks
                manifest["status"] = "engineering_smoke_only" if smoke else "completed"
                manifest["available_comparisons"] = {
                    group: [c["name"] for c in manifest["checkpoints"] if c["group"] == group and c["status"] == "verified"]
                    for group in ("A", "B")}
        manifest["seconds"] = time.monotonic() - start
        write_json(output / "experiment_manifest.json", manifest)
        write_metrics(output, rows)
        examples = command_examples(config["source_config"]["path"])
        pending = ({k: examples[k] for k in ("formal_not_run_by_smoke", "prepare_only")}
                   if smoke else
                   {k: examples[k] for k in ("engineering_smoke", "formal_not_run_by_smoke")}
                   if action == "audit" else
                   {"formal_evaluation": examples["formal_not_run_by_smoke"]}
                   if action == "prepare" else {})
        write_report(output, manifest, rows, commands=pending)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}",
                        seconds=time.monotonic() - start)
        write_json(output / "experiment_manifest.json", manifest)
        write_json(output / "partial_metrics.json", rows)
        raise
