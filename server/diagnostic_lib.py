"""Frozen-encoder diagnostics; deliberately outside the checkpoint source package."""
from pathlib import Path
import hashlib
import math
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sepa_plan_b.config import digest
from sepa_plan_b.data import ImageTiles, write_json
from sepa_plan_b.evaluation import linear_metrics
from sepa_plan_b.model import initialization
from sepa_plan_b.numerics import checked_probe_step, feature_statistics, require_finite
from sepa_plan_b.rng import generator
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_tensor(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def encode_branch(model, tiles, branch):
    if branch == "student":
        return model.encode(tiles)
    if branch != "teacher":
        raise ValueError(f"Unknown encoder branch: {branch}")
    if model.method == "full":
        slots = torch.arange(9, device=tiles.device).expand(len(tiles), -1)
        return model.teacher(tiles, slots)
    return model.teacher(tiles)


def validate_features(payload, metadata, manifest, dimension):
    if payload["metadata"] != metadata:
        raise ValueError("Feature cache identity mismatch")
    records = manifest["records"]
    if payload["ids"] != [r["id"] for r in records]:
        raise ValueError("Feature cache image order mismatch")
    if not torch.equal(payload["labels"], torch.tensor([r["label"] for r in records])):
        raise ValueError("Feature cache labels mismatch")
    if payload["features"].shape != (len(records), dimension) or payload["features"].dtype != torch.float32:
        raise ValueError("Feature cache shape/dtype mismatch")
    require_finite("feature cache", features=payload["features"])


@torch.no_grad()
def cached_extract(path, model, branch, manifest, config, device, metadata, progress):
    path = Path(path)
    receipt = path.with_suffix(".json")
    dimension = config.model.encoder_spec[0]
    if path.exists() and receipt.exists():
        import json
        saved = json.loads(receipt.read_text())
        if saved["metadata"] != metadata or saved["sha256"] != file_sha(path):
            raise ValueError(f"Corrupt or stale feature cache: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_features(payload, metadata, manifest, dimension)
        progress(event="cache_reused", cache=path.name)
        return payload
    if model is None:
        raise ValueError(f"Model required to extract missing cache: {path}")
    model.eval()
    loader = DataLoader(ImageTiles(manifest, config), batch_size=config.probe.batch_size,
                        num_workers=config.training.num_workers, generator=generator(0, "eval_loader"))
    features, labels, ids = [], [], []
    started = time.monotonic()
    for index, batch in enumerate(loader):
        tiles = batch["canonical"].to(device)
        require_finite("diagnostic extraction", tiles=tiles)
        h = encode_branch(model, tiles, branch)
        feature = h.mean(1).cpu().float()
        require_finite("diagnostic extraction", encoder_output=h, features=feature)
        features.append(feature)
        # Do not retain a shared-memory FD for every DataLoader batch.
        labels.append(batch["label"].clone())
        ids.extend(batch["id"])
        if index % 100 == 0 or index + 1 == len(loader):
            progress(event="extract", batch=index+1, batches=len(loader),
                     elapsed_seconds=time.monotonic()-started)
    payload = {"features": torch.cat(features), "labels": torch.cat(labels), "ids": ids, "metadata": metadata}
    validate_features(payload, metadata, manifest, dimension)
    save_tensor(path, payload)
    write_json(receipt, {"metadata": metadata, "sha256": file_sha(path)})
    return payload


def record_audit(records):
    groups = {}
    for i, record in enumerate(records):
        groups.setdefault(record["id"], []).append({"index": i, "label": record["label"], "path": record.get("path")})
    duplicates = {k: v for k, v in groups.items() if len(v) > 1}
    conflicts = {k: v for k, v in duplicates.items() if len({r["label"] for r in v}) > 1}
    return {"records": len(records), "unique_images": len(groups), "duplicate_groups": duplicates,
            "conflicting_groups": conflicts, "conflicting_records": sum(map(len, conflicts.values()))}


def grouped_split(records, seed=0, fraction=0.1, excluded_ids=()):
    """Stratify image-content groups, so repeated images cannot leak across splits."""
    if not 0 < fraction < 1:
        raise ValueError("Split fraction must be between zero and one")
    groups, labels = {}, {}
    excluded_ids = set(excluded_ids)
    for i, record in enumerate(records):
        identity, label = record["id"], record["label"]
        if identity in excluded_ids:
            continue
        if identity in labels and labels[identity] != label:
            raise ValueError("Identical training image has conflicting labels")
        labels[identity] = label
        groups.setdefault(identity, []).append(i)
    if set(labels.values()) != {r["label"] for r in records}:
        raise ValueError("Exclusion removed an entire class")
    dev_ids = set()
    for label in sorted(set(labels.values())):
        candidates = sorted(k for k in groups if labels[k] == label)
        if len(candidates) < 2:
            raise ValueError("Need at least two distinct images per class")
        order = torch.randperm(len(candidates), generator=generator(seed, "diagnostic_dev", label))
        count = min(len(candidates)-1, max(1, math.ceil(len(candidates)*fraction)))
        dev_ids.update(candidates[i] for i in order[:count].tolist())
    train = torch.tensor([i for i, r in enumerate(records) if r["id"] not in dev_ids and r["id"] not in excluded_ids])
    dev = torch.tensor([i for i, r in enumerate(records) if r["id"] in dev_ids])
    return train, dev


def health_indices(records, seed=0, per_class=100, excluded_ids=()):
    excluded_ids = set(excluded_ids)
    unique = {}
    for i, record in enumerate(records):
        if record["id"] not in excluded_ids:
            unique.setdefault(record["id"], i)
    selected = []
    for label in sorted({r["label"] for r in records}):
        indices = sorted(i for i in unique.values() if records[i]["label"] == label)
        order = torch.randperm(len(indices), generator=generator(seed, "diagnostic_health", label))
        selected.extend(indices[i] for i in order[:per_class].tolist())
    return torch.tensor(selected)


def feature_health(x, y, seed=0):
    """Centered spectrum plus raw/centered cosine on identical labeled pairs."""
    require_finite("feature health", features=x)
    x = x.double()
    centered = x-x[:1]
    centered = centered-centered.mean(0)
    covariance = centered.T @ centered / len(x)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    total = eigenvalues.sum().item()
    p = eigenvalues/total if total > 0 else eigenvalues
    effective = (-(p[p > 0]*p[p > 0].log()).sum()).exp().item() if total > 0 else 0.0
    participation = 1/p.square().sum().item() if total > 0 else 0.0
    std = covariance.diag().clamp_min(0).sqrt()
    same_a, same_b, diff_a, diff_b = [], [], [], []
    for c in y.unique(sorted=True):
        ids = (y == c).nonzero().flatten()
        other = (y != c).nonzero().flatten()
        rng = generator(seed, "diagnostic_cosine", int(c))
        ids = ids[torch.randperm(len(ids), generator=rng)]
        if len(ids) > 1:
            same_a.extend(ids.tolist())
            same_b.extend(ids.roll(1).tolist())
        if len(other):
            diff_a.extend(ids.tolist())
            diff_b.extend(other[torch.randint(len(other), (len(ids),), generator=rng)].tolist())
    def cosine_rows(values):
        normalized = F.normalize(values, dim=-1)
        def stats(a, b):
            if not a:
                return None
            scores = (normalized[a]*normalized[b]).sum(-1)
            return {"mean": scores.mean().item(), "p05": scores.quantile(0.05).item(),
                    "p50": scores.quantile(0.5).item(), "p95": scores.quantile(0.95).item(), "n_pairs": len(a)}
        return {"same_class": stats(same_a, same_b), "different_class": stats(diff_a, diff_b)}
    return {"n_images": len(x), "dimension": x.shape[1], "total_centered_variance": total,
            "effective_rank": effective, "participation_rank": participation,
            "explained_variance": {str(k): p[:k].sum().item() for k in (1, 5, 10)},
            "feature_std_mean": std.mean().item(), "feature_std_min": std.min().item(),
            "feature_std_median": std.median().item(), "feature_norm_mean": x.norm(dim=-1).mean().item(),
            "cosine_raw": cosine_rows(x), "cosine_centered": cosine_rows(centered),
            "zero_centered_variance": total == 0,
            "interpretation": "Descriptive train-only measures; compare with matched random encoder. No pass threshold."}


def fit_trajectory(x, y, classes, config, seed, device, *, dev=None, marks=(), progress=lambda **kw: None,
                   snapshot=lambda *args: None):
    """Same optimizer, normalization, initialization and sample order as fit_linear."""
    mean, std = feature_statistics(x, "diagnostic probe training")
    normalized = (x-mean)/std
    require_finite("diagnostic probe", normalized_features=normalized)
    with initialization(seed, "linear_probe"):
        head = nn.Linear(x.shape[1], classes).to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=config.probe.learning_rate,
                                momentum=0.9, weight_decay=config.probe.weight_decay)
    trace = []
    started = time.monotonic()
    for epoch in range(config.probe.epochs):
        head.train()
        order = torch.randperm(len(x), generator=generator(seed, "probe_order", epoch))
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        correct = torch.zeros((), dtype=torch.long, device=device)
        for selected in order.split(config.probe.batch_size):
            logits = head(normalized[selected].to(device))
            require_finite("diagnostic probe training", logits=logits)
            targets = y[selected].to(device)
            loss = F.cross_entropy(logits, targets)
            checked_probe_step(loss, head, optimizer, "diagnostic probe training")
            loss_sum += loss.detach().double()*len(selected)
            correct += (logits.detach().argmax(-1) == targets).sum()
        row = {"epoch": epoch+1, "online_train_loss": loss_sum.item()/len(x),
               "online_train_top1": correct.item()/len(x)}
        head.eval()
        if epoch+1 in marks:
            if dev is not None:
                row["dev_scores"], _ = linear_metrics(head, mean, std, *dev, device)
            snapshot(epoch+1, head, mean, std, row)
        trace.append(row)
        if epoch == 0 or (epoch+1) % 5 == 0 or epoch+1 == config.probe.epochs:
            progress(event="fit", **row, epochs=config.probe.epochs, elapsed_seconds=time.monotonic()-started)
    return head, mean, std, trace


def select_recipe(dev_rows, anchors=("k0_student", "full_student")):
    grouped = {}
    for row in dev_rows:
        key = (row["learning_rate"], row["epoch"])
        scores = grouped.setdefault(key, {})
        if row["anchor"] in scores:
            raise ValueError("Repeated anchor/recipe dev score")
        score = row["dev_scores"]["top1"]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid dev score")
        scores[row["anchor"]] = score
    candidates = []
    for (lr, epoch), values in grouped.items():
        if set(values) != set(anchors):
            raise ValueError("Shared recipe requires all anchors")
        candidates.append({"learning_rate": lr, "epochs": epoch,
                           "mean_dev_top1": sum(values.values())/len(anchors), "anchors": values})
    if not candidates:
        raise ValueError("No valid dev candidates")
    candidates.sort(key=lambda r: (-round(r["mean_dev_top1"], 12), r["epochs"], r["learning_rate"]))
    return {"selected": candidates[0], "candidates": candidates,
            "selection_rule": "Mean internal-dev Top-1 across k0 student and full student; ties: fewer epochs, then smaller LR.",
            "official_validation_used_for_selection": False}
