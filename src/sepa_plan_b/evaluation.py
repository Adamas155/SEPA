from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import digest
from .data import ImageTiles, load_manifest, manifests, validate_pair, write_json
from .engine import load_encoder_run
from .geometry import MEAN, STD
from .losses import pair_targets
from .model import initialization
from .numerics import checked_probe_step, feature_statistics, require_finite
from .rng import generator, local_rng


def subset_manifest(manifest, limit=0):
    if limit < 0:
        raise ValueError("Image limit must be nonnegative")
    if not limit or limit >= len(manifest["records"]):
        return manifest
    if limit < len(manifest["classes"]):
        raise ValueError("Image limit must be at least the class count")
    groups = [[r for r in manifest["records"] if r["label"] == c] for c in range(len(manifest["classes"]))]
    selected, offset = [], 0
    while len(selected) < limit:
        for group in groups:
            if offset < len(group) and len(selected) < limit:
                selected.append(group[offset])
        offset += 1
    return {**manifest, "records": selected}


@torch.no_grad()
def extract(model, manifest, config, device, *, tile_features=False, pixel_baselines=False):
    dataset = ImageTiles(manifest, config)
    loader = DataLoader(dataset, batch_size=config.probe.batch_size, num_workers=config.training.num_workers,
                        generator=generator(0, "eval_loader"))
    features, labels, borders, priors = [], [], [], []
    model.eval()
    for batch in loader:
        tiles = batch["canonical"].to(device)
        require_finite("feature extraction", tiles=tiles)
        h = model.encode(tiles)
        feature = (h if tile_features else h.mean(1)).cpu().float()
        require_finite("feature extraction", encoder_output=h, features=feature)
        features.append(feature)
        labels.append(batch["label"])
        if pixel_baselines:
            real = tiles[..., 3:-3, 3:-3] if config.geometry.mode == "pad74" else tiles
            raw = real * real.new_tensor(STD)[None, None, :, None, None] + real.new_tensor(MEAN)[None, None, :, None, None]
            flat = raw.flatten(0, 1)
            strips = [flat[:, :, :8, :].mean(2), flat[:, :, -8:, :].mean(2),
                      flat[:, :, :, :8].mean(3), flat[:, :, :, -8:].mean(3)]
            border = torch.cat([F.adaptive_avg_pool1d(s, 8).flatten(1) for s in strips], 1)
            prior = torch.cat((flat.mean((-1, -2)), F.adaptive_avg_pool2d(flat, 2).flatten(1)), 1)
            require_finite("pixel baseline extraction", border=border, prior=prior)
            borders.append(border.reshape(len(tiles), 9, -1).cpu())
            priors.append(prior.reshape(len(tiles), 9, -1).cpu())
    result = {"features": torch.cat(features), "labels": torch.cat(labels)}
    if pixel_baselines:
        result.update(border=torch.cat(borders), prior=torch.cat(priors))
    return result


def label_subset(labels, fraction, seed):
    selected = []
    for c in labels.unique(sorted=True):
        indices = (labels == c).nonzero().flatten()
        order = torch.randperm(len(indices), generator=generator(seed, "label_subset", int(c)))
        count = max(1, int(np.ceil(len(indices) * fraction)))
        selected.append(indices[order[:count]])
    return torch.cat(selected)


def fit_linear(x, y, classes, config, seed, device):
    """All statistics and optimization use probe TRAIN features only."""
    mean, std = feature_statistics(x, "linear probe training")
    normalized = (x-mean) / std
    require_finite("linear probe training", normalized_features=normalized)
    with initialization(seed, "linear_probe"):
        head = nn.Linear(x.shape[1], classes).to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=config.probe.learning_rate,
                                momentum=0.9, weight_decay=config.probe.weight_decay)
    head.train()
    for epoch in range(config.probe.epochs):
        indices = torch.randperm(len(x), generator=generator(seed, "probe_order", epoch))
        for selected in indices.split(config.probe.batch_size):
            logits = head(normalized[selected].to(device))
            require_finite("linear probe training", logits=logits)
            loss = F.cross_entropy(logits, y[selected].to(device))
            checked_probe_step(loss, head, optimizer, "linear probe training")
    head.eval()
    return head, mean, std


@torch.no_grad()
def linear_metrics(head, mean, std, x, y, device, batch_size=256):
    require_finite("linear probe evaluation", features=x, mean=mean, std=std)
    if (std <= 0).any():
        raise ValueError("Linear probe standard deviations must be positive")
    top1 = top5 = 0
    predictions = []
    for start in range(0, len(x), batch_size):
        normalized = (x[start:start+batch_size] - mean) / std
        require_finite("linear probe evaluation", normalized_features=normalized)
        scores = head(normalized.to(device)).cpu()
        require_finite("linear probe evaluation", logits=scores)
        labels = y[start:start+batch_size]
        top1 += (scores.argmax(-1) == labels).sum().item()
        top5 += (scores.topk(min(5, scores.shape[1]), -1).indices == labels[:, None]).any(-1).sum().item()
        predictions.append(scores.argmax(-1))
    return {"top1": top1/len(x), "top5": top5/len(x)}, torch.cat(predictions)


@torch.no_grad()
def knn_accuracy(train_x, train_y, test_x, test_y, classes, k=20, temperature=0.07, device="cpu"):
    require_finite("kNN", train_features=train_x, test_features=test_x)
    reference = F.normalize(train_x.to(device), dim=-1)
    labels = train_y.to(device)
    correct = 0
    for start in range(0, len(test_x), 128):
        scores = F.normalize(test_x[start:start+128].to(device), dim=-1) @ reference.T
        require_finite("kNN", similarities=scores)
        values, indices = scores.topk(min(k, len(reference)), dim=-1)
        weights = ((values-values.max(-1, keepdim=True).values) / temperature).exp()
        votes = torch.zeros(len(values), classes, device=device)
        votes.scatter_add_(1, labels[indices], weights)
        require_finite("kNN", weights=weights, votes=votes)
        correct += (votes.argmax(-1).cpu() == test_y[start:start+128]).sum().item()
    return correct / len(test_x)


def verified_manifests(config, checkpoint):
    train, val = manifests(config)
    meta = checkpoint["metadata"]
    if train["fingerprint"] != meta["train_fingerprint"] or val["fingerprint"] != meta["val_fingerprint"]:
        raise ValueError("Current data manifests differ from the checkpoint")
    return train, val


def probe(checkpoint_path, *, output=None, limit=0, label_fraction=None, train_manifest=None, val_manifest=None, device=None):
    model, config, checkpoint, target = load_encoder_run(checkpoint_path, device)
    train, val = verified_manifests(config, checkpoint)
    transfer = bool(train_manifest or val_manifest)
    if transfer:
        if not train_manifest or not val_manifest:
            raise ValueError("Transfer probing requires both train and val manifests")
        train, val = load_manifest(train_manifest), load_manifest(val_manifest)
        validate_pair(train, val, len(train["classes"]))
    if label_fraction is not None:
        config = replace(config, probe=replace(config.probe, label_fraction=label_fraction)).validate()
    train, val = subset_manifest(train, limit), subset_manifest(val, limit)
    train_f, val_f = extract(model, train, config, target), extract(model, val, config, target)
    seed = checkpoint["metadata"]["seed"]
    indices = label_subset(train_f["labels"], config.probe.label_fraction, seed)
    x, y = train_f["features"][indices], train_f["labels"][indices]
    classes = len(train["classes"])
    head, mean, std = fit_linear(x, y, classes, config, seed, target)
    scores, _ = linear_metrics(head, mean, std, val_f["features"], val_f["labels"], target)
    scores["knn"] = knn_accuracy(x, y, val_f["features"], val_f["labels"], classes,
                                  config.probe.knn_k, config.probe.knn_temperature, target)
    result = {"schema": 1, "metadata": checkpoint["metadata"], "checkpoint_step": checkpoint["step"],
              "complete_pretraining": checkpoint["step"] == config.training.steps,
              "transfer": transfer, "train_manifest": train["fingerprint"], "val_manifest": val["fingerprint"],
              "evaluation_ids_hash": digest([[r["id"] for r in train["records"]], [r["id"] for r in val["records"]]]),
              "label_fraction": config.probe.label_fraction, "n_train_labels": len(x), "n_val": len(val_f["labels"]),
              "limit": limit, "scores": scores}
    name = "probe.json" if not transfer and config.probe.label_fraction == 1 and not limit else f"probe-{digest(result)[:12]}.json"
    destination = Path(output) if output else Path(checkpoint_path).with_name(name)
    write_json(destination, result)
    torch.save({"head": {k: v.cpu() for k, v in head.state_dict().items()}, "mean": mean, "std": std,
                "metadata": result}, destination.with_suffix(".pt"))
    return {"output": str(destination), **{k: v for k, v in result.items() if k != "metadata"}}


def padding_diagnostic(checkpoint_path, *, output=None, limit=256, device=None):
    model, config, checkpoint, target = load_encoder_run(checkpoint_path, device)
    if config.geometry.mode != "pad74" or model.method != "sepa":
        raise ValueError("Padding diagnostic requires a SEPA pad74 checkpoint")
    train, val = verified_manifests(config, checkpoint)
    train, val = subset_manifest(train, limit), subset_manifest(val, limit)
    reference = config.geometry.padding
    train_f = extract(model, train, config, target)
    ref = extract(model, val, config, target, tile_features=True)
    head, mean, std = fit_linear(train_f["features"], train_f["labels"], len(train["classes"]), config,
                                 checkpoint["metadata"]["seed"], target)
    ref_score, ref_pred = linear_metrics(head, mean, std, ref["features"].mean(1), ref["labels"], target)
    results = {}
    for mode in ("reflect", "replicate", "constant"):
        changed = replace(config, geometry=replace(config.geometry, padding=mode))
        features = ref if mode == reference else extract(model, val, changed, target, tile_features=True)
        score, predictions = linear_metrics(head, mean, std, features["features"].mean(1), features["labels"], target)
        cosine = 1-F.cosine_similarity(ref["features"], features["features"], dim=-1)
        require_finite("padding diagnostic", cosine_distance=cosine)
        results[mode] = {**score, "top1_delta": score["top1"]-ref_score["top1"],
                         "prediction_flip_rate": (predictions != ref_pred).float().mean().item(),
                         "tile_cosine_distance_mean": cosine.mean().item(),
                         "tile_cosine_distance_p95": torch.quantile(cosine.flatten(), 0.95).item()}
    result = {"checkpoint": str(checkpoint_path), "run_id": checkpoint["metadata"]["run_id"],
              "reference_padding": reference, "n_val": len(ref["labels"]), "results": results,
              "interpretation": "Fixed encoder and classifier, identical real 74px content; measures padding sensitivity, not proof of shortcut learning."}
    destination = Path(output) if output else Path(checkpoint_path).with_name("padding_diagnostic.json")
    write_json(destination, result)
    return result


def pair_features(features, i, j, task):
    a, b = features[:, i], features[:, j]
    return torch.cat(((a-b).abs(), a*b), -1) if task == "adjacency" else torch.cat((a, b, a-b, a*b), -1)


def spatial_fit(train_x, test_x, config, seed, task, device):
    # Every image stays entirely in one partition. Sample pairs lazily, not Nx72xD in RAM.
    pairs = torch.tensor([(i, j) for i in range(9) for j in range(9) if i != j])
    adjacent, direction, _ = pair_targets(torch.arange(9)[None])
    labels = (adjacent.long() if task == "adjacency" else direction)[0, pairs[:, 0], pairs[:, 1]]
    require_finite("spatial probe evaluation", features=test_x)
    mean, std = feature_statistics(train_x.flatten(0, 1), "spatial probe training")
    train_x, test_x = (train_x-mean)/std, (test_x-mean)/std
    require_finite("spatial probe", normalized_train=train_x, normalized_test=test_x)
    dim = train_x.shape[-1] * (2 if task == "adjacency" else 4)
    with initialization(seed, f"spatial_{task}"):
        head = nn.Linear(dim, 2 if task == "adjacency" else 8).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.probe.spatial_learning_rate, weight_decay=0.01)
    rng = generator(seed, "spatial_pairs", task)
    b = config.probe.batch_size
    for _ in range(config.probe.spatial_steps):
        image_ids = torch.randint(len(train_x), (b,), generator=rng)
        pair_ids = torch.randint(len(pairs), (b,), generator=rng)
        selected = train_x[image_ids]
        i, j = pairs[pair_ids].T
        a, z = selected[torch.arange(b), i], selected[torch.arange(b), j]
        x = torch.cat(((a-z).abs(), a*z), -1) if task == "adjacency" else torch.cat((a, z, a-z, a*z), -1)
        require_finite("spatial probe training", pair_features=x)
        logits = head(x.to(device))
        require_finite("spatial probe training", logits=logits)
        loss = F.cross_entropy(logits, labels[pair_ids].to(device))
        checked_probe_step(loss, head, optimizer, "spatial probe training")
    head.eval()
    confusion = torch.zeros(2 if task == "adjacency" else 8, 2 if task == "adjacency" else 8, dtype=torch.long)
    with torch.no_grad():
        for p, (i, j) in enumerate(pairs.tolist()):
            for images in test_x.split(b):
                features = pair_features(images, i, j, task)
                require_finite("spatial probe evaluation", pair_features=features)
                logits = head(features.to(device))
                require_finite("spatial probe evaluation", logits=logits)
                predicted = logits.argmax(-1).cpu()
                confusion[labels[p]] += torch.bincount(predicted, minlength=confusion.shape[0])
    recall = confusion.diag() / confusion.sum(1).clamp_min(1)
    return {"accuracy": (confusion.diag().sum()/confusion.sum()).item(),
            "balanced_accuracy": recall.mean().item(), "confusion": confusion.tolist()}


def position_fit(train_x, test_x, config, seed, device):
    """Predict canonical position from one tile on held-out IMAGE identities."""
    require_finite("position probe evaluation", features=test_x)
    train_x, test_x = train_x.flatten(0, 1), test_x.flatten(0, 1)
    train_y = torch.arange(9).repeat(len(train_x)//9)
    test_y = torch.arange(9).repeat(len(test_x)//9)
    mean, std = feature_statistics(train_x, "position probe training")
    train_x = (train_x-mean)/std
    require_finite("position probe training", normalized_features=train_x)
    with initialization(seed, "position_probe"):
        head = nn.Linear(train_x.shape[-1], 9).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.probe.spatial_learning_rate, weight_decay=0.01)
    rng = generator(seed, "position_probe_batches")
    for _ in range(config.probe.spatial_steps):
        indices = torch.randint(len(train_x), (config.probe.batch_size,), generator=rng)
        logits = head(train_x[indices].to(device))
        require_finite("position probe training", logits=logits)
        loss = F.cross_entropy(logits, train_y[indices].to(device))
        checked_probe_step(loss, head, optimizer, "position probe training")
    scores, _ = linear_metrics(head.eval(), mean, std, test_x, test_y, device, config.probe.batch_size)
    return {"accuracy": scores["top1"], "chance": 1/9}


def spatial_probe(checkpoint_path, *, output=None, limit=1000, device=None):
    model, config, checkpoint, target = load_encoder_run(checkpoint_path, device)
    _, val = verified_manifests(config, checkpoint)
    val = subset_manifest(val, limit)
    # Split image identities BEFORE feature/pair construction. Duplicate files stay together.
    ids = sorted({r["id"] for r in val["records"]})
    if len(ids) < 4:
        raise ValueError("Spatial probe requires at least four distinct images")
    local_rng(0, "spatial_image_split").shuffle(ids)
    training_ids = set(ids[:len(ids)//2])
    train = {**val, "records": [r for r in val["records"] if r["id"] in training_ids]}
    test = {**val, "records": [r for r in val["records"] if r["id"] not in training_ids]}
    a = extract(model, train, config, target, tile_features=True, pixel_baselines=True)
    b = extract(model, test, config, target, tile_features=True, pixel_baselines=True)
    result = {"run_id": checkpoint["metadata"]["run_id"], "n_train_images": len(train["records"]),
              "n_test_images": len(test["records"]), "split_hash": digest([sorted(training_ids), sorted(set(ids)-training_ids)]),
              "results": {}}
    for name in ("features", "border", "prior"):
        result["results"][name] = {task: spatial_fit(a[name], b[name], config, checkpoint["metadata"]["seed"], task, target)
                                  for task in ("adjacency", "direction")}
        result["results"][name]["canonical_position"] = position_fit(a[name], b[name], config, checkpoint["metadata"]["seed"], target)
    result["interpretation"] = "Image-disjoint shallow probes; border/prior dimensions differ, so these are shortcut diagnostics, not matched-capacity causal proof."
    destination = Path(output) if output else Path(checkpoint_path).with_name("spatial_probe.json")
    write_json(destination, result)
    return result


def analyze(paths, delta_pp=1.0):
    if not np.isfinite(delta_pp) or delta_pp < 0:
        raise ValueError("delta_pp must be finite and nonnegative")
    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if not results:
        raise ValueError("No probe results found")
    groups, protocol = {}, None
    for r in results:
        if any(not np.isfinite(score) or not 0 <= score <= 1 for score in r["scores"].values()):
            raise ValueError("Probe scores must be finite accuracies in [0,1]")
        if r["transfer"] or r["limit"] or r["label_fraction"] != 1 or not r["complete_pretraining"]:
            raise ValueError("Stage 1 analysis requires complete, full-label, untruncated in-domain probes")
        meta = r["metadata"]
        cfg = meta["config"]
        if cfg["model"]["relation"] != "none":
            raise ValueError("Stage 1 must use relation=none")
        signature = digest([cfg, meta["source_hash"], r["train_manifest"], r["val_manifest"], r["evaluation_ids_hash"]])
        if protocol is not None and signature != protocol:
            raise ValueError("Results have different scientific configurations or data; analyze separately")
        protocol = signature
        arm = "full" if meta["method"] == "full" else f"k{meta['k']}"
        if meta["seed"] in groups.setdefault(arm, {}):
            raise ValueError("Duplicate arm/seed results")
        groups[arm][meta["seed"]] = r["scores"]["top1"]
    rng = np.random.default_rng(0)
    comparisons = {}
    for arm in ("k0", "k6", "full"):
        seeds = sorted(set(groups.get("k3", {})) & set(groups.get(arm, {})))
        if not seeds:
            continue
        differences = np.array([groups["k3"][s]-groups[arm][s] for s in seeds]) * 100
        if len(seeds) >= 2:
            boot = rng.choice(differences, (10000, len(differences)), replace=True).mean(1)
            interval = np.quantile(boot, [0.025, 0.975]).tolist()
        else:
            interval = None
        comparisons[f"k3_minus_{arm}"] = {"paired_seeds": seeds, "mean_pp": float(differences.mean()), "bootstrap_95ci_pp": interval}
    required = [set(groups.get(arm, {})) for arm in ("k0", "k3", "k6", "full")]
    complete = all(len(s) >= 10 for s in required) and all(s == required[0] for s in required)
    achieved = complete and all(comparisons[f"k3_minus_{arm}"]["mean_pp"] > delta_pp for arm in ("k0", "k6"))
    return {"arms": {a: {"seeds": sorted(v), "mean_top1": float(np.mean(list(v.values())))} for a, v in groups.items()},
            "comparisons": comparisons, "delta_pp": delta_pp,
            "status": "INSUFFICIENT" if not complete else "STAGE1_POINT_ESTIMATE_GATE_PASS" if achieved else "STAGE1_POINT_ESTIMATE_GATE_NOT_MET",
            "interpretation": "Exploratory paired bootstrap shown separately from roadmap's point-estimate gate; not proof that spatial corruption generally works or fails."}
