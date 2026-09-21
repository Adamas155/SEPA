"""Image-level splitting before pairs, with explicit duplicate provenance."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from sepa_plan_b.config import Config
from sepa_plan_b.data import ImageTiles, load_manifest
from diagnostic_lib import grouped_split, record_audit

from .common import digest, file_hash, read_json, write_json


def load_sources(train_path, test_path, *, train_root=None, test_root=None,
                 expected_classes=100, expected_test_images=5000):
    train, test = load_manifest(train_path), load_manifest(test_path)
    if (train["role"], test["role"]) != ("train", "val"):
        raise ValueError("Expected original training and official validation manifests")
    if train["classes"] != test["classes"] or len(train["classes"]) != expected_classes:
        raise ValueError("ImageNet-100 class mappings do not match")
    if len(test["records"]) != expected_test_images:
        raise ValueError(f"Expected {expected_test_images} official test images")
    provenance = {
        "train": {"path": str(Path(train_path).resolve()), "file_hash": file_hash(train_path),
                  "fingerprint": train["fingerprint"], "records": len(train["records"])},
        "test": {"path": str(Path(test_path).resolve()), "file_hash": file_hash(test_path),
                 "fingerprint": test["fingerprint"], "records": len(test["records"])},
    }
    train, test = dict(train), dict(test)
    if train_root is not None:
        train["root"] = str(Path(train_root).resolve())
    if test_root is not None:
        test["root"] = str(Path(test_root).resolve())
    for role, source in (("train", train), ("test", test)):
        if not Path(source["root"]).is_dir():
            raise FileNotFoundError(f"{role} image root unavailable: {source['root']}")
    return train, test, provenance


def decoded_groups(sources, output, *, workers=4):
    """Reuse file identity, then merge exactly identical decoded RGB contents.

    No perceptual/near-duplicate equivalence is claimed. The resulting map is
    tied to both source manifests and verified image bytes.
    """
    identity = {"sources": [x["fingerprint"] for x in sources],
                "method": "manifest file identity + RGB dimensions and decoded pixels v1"}
    output = Path(output)
    if output.exists():
        saved = read_json(output)
        if saved["identity"] != identity:
            raise ValueError("Content-group cache belongs to different manifests")
        expected_keys = {r["sha256"] for source in sources for r in source["records"]}
        if set(saved["groups"]) != expected_keys or saved.get("groups_hash") != digest(saved["groups"]):
            raise ValueError("Content-group cache is incomplete or changed")
        return saved["groups"], saved["provenance"]
    jobs = {}
    for manifest in sources:
        root = Path(manifest["root"]).resolve()
        for record in manifest["records"]:
            path = (root / record["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Image path escapes root")
            jobs.setdefault(record["sha256"], (path, record["sha256"]))

    def one(job):
        path, expected = job
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Image content changed: {path}")
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert("RGB")
            h = hashlib.sha256()
            h.update(str(rgb.size).encode())
            h.update(rgb.tobytes())
        return expected, h.hexdigest()

    if workers < 1:
        raise ValueError("At least one decoding worker is required")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        groups = dict(pool.map(one, jobs.values()))
    provenance = {"method": identity["method"], "unique_file_contents": len(groups),
                  "unique_decoded_contents": len(set(groups.values())),
                  "limitations": "Does not identify perceptually similar, cropped or lossy-recompressed images with differing pixels."}
    write_json(output, {"identity": identity, "groups": groups,
                        "groups_hash": digest(groups), "provenance": provenance})
    return groups, provenance


def make_splits(train, test, *, seed=0, dev_fraction=0.1, content_groups=None,
                existing_split=None, smoke_limits=None):
    """Only manifest metadata is used here; no encoder/probe or test scores."""
    def grouped(records):
        return [{**r, "source_index": i,
                 "id": content_groups[r["sha256"]] if content_groups is not None else r["id"]}
                for i, r in enumerate(records)]

    records, test_records = grouped(train["records"]), grouped(test["records"])
    test_ids = {r["id"] for r in test_records}
    audit = record_audit(records)
    conflict_ids = set(audit["conflicting_groups"])
    excluded_ids = test_ids | conflict_ids
    origin = "new deterministic content-group-stratified split"
    reused = False
    if existing_split is not None:
        old = read_json(existing_split) if isinstance(existing_split, (str, Path)) else existing_split
        a, b = old["train_indices"], old["dev_indices"]
        if any(not isinstance(i, int) or not 0 <= i < len(records) for i in a + b):
            raise ValueError("Existing split has invalid source indices")
        if len(a) != len(set(a)) or len(b) != len(set(b)) or set(a) & set(b):
            raise ValueError("Existing split repeats original images")
        # Verify the index map against original IDs, not newly merged RGB IDs.
        for name, indices in (("train", a), ("dev", b)):
            expected = old.get(name + "_ids_hash")
            if not expected or digest([train["records"][i]["id"] for i in indices]) != expected:
                raise ValueError("Existing split does not identify these source records")
        original_conflicts = record_audit(train["records"])["conflicting_groups"]
        reproduced_a, reproduced_b = grouped_split(
            train["records"], seed=seed, fraction=dev_fraction,
            excluded_ids=original_conflicts,
        )
        recipe_reproduced = a == reproduced_a.tolist() and b == reproduced_b.tolist()
        a = [i for i in a if records[i]["id"] not in excluded_ids]
        b = [i for i in b if records[i]["id"] not in excluded_ids]
        usable = {i for i, r in enumerate(records) if r["id"] not in excluded_ids}
        if recipe_reproduced and set(a) | set(b) == usable and not ({records[i]["id"] for i in a} & {records[i]["id"] for i in b}):
            reused = True
            origin = "existing grouped split reproduced with recorded seed/fraction; test overlap/conflicting content excluded"
        else:
            origin += "; existing map could not preserve the stronger content-group boundary"
    if not reused:
        a, b = grouped_split(records, seed=seed, fraction=dev_fraction, excluded_ids=excluded_ids)
        a, b = a.tolist(), b.tolist()
    manifests = {
        "train": {**train, "records": [records[i] for i in a]},
        "dev": {**train, "records": [records[i] for i in b]},
        "test": {**test, "records": test_records},
    }
    if smoke_limits is not None:
        if len(smoke_limits) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in smoke_limits):
            raise ValueError("Smoke limits must contain exactly three positive integers")
        # Engineering subset only; fixed selection before any head is fitted.
        for role, count in zip(("train", "dev", "test"), smoke_limits):
            candidates = manifests[role]["records"]
            order = sorted(range(len(candidates)), key=lambda i: digest(
                [seed, "engineering-subset", role, candidates[i]["id"], candidates[i]["path"]]))
            manifests[role]["records"] = [candidates[i] for i in sorted(order[:count])]
    for role, manifest in manifests.items():
        manifest["role"] = role
        manifest.pop("fingerprint", None)
        manifest["fingerprint"] = digest(manifest)
        if not manifest["records"]:
            raise ValueError(f"Empty {role} image split")
    validate_disjoint(manifests)
    return {
        "schema": 1, "split_seed": seed, "dev_fraction": dev_fraction, "origin": origin,
        "engineering_only": smoke_limits is not None,
        "excluded_training_records": len(records) - len(a) - len(b),
        "conflicting_content_groups": len(conflict_ids),
        "test_overlap_training_records": sum(r["id"] in test_ids for r in records),
        "dedup_method": "decoded RGB plus original file identities" if content_groups is not None else "existing manifest content IDs",
        "dedup_limitations": "No perceptual near-duplicate detection. Manifest-only engineering mode does not verify decoded-pixel equivalence.",
        "manifests": manifests,
    }


def validate_disjoint(manifests):
    groups = {role: {r["id"] for r in manifest["records"]}
              for role, manifest in manifests.items()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if groups[a] & groups[b]:
            raise ValueError(f"Duplicate original content crosses {a}/{b}")


def mask_border(tiles, width=8):
    if tiles.shape[-2:] != (80, 80) or width != 8:
        raise ValueError("Protocol requires an 8-pixel border on 80x80 tiles")
    masked = tiles.clone()
    masked[..., :width, :] = 0
    masked[..., -width:, :] = 0
    masked[..., :, :width] = 0
    masked[..., :, -width:] = 0
    return masked


class TileDataset(Dataset):
    """Returns pixels only. Position/content IDs remain on the label side."""

    def __init__(self, manifest, tile_orders, *, condition="clean"):
        self.images = ImageTiles(manifest, Config(), training=False)
        self.orders = np.asarray(tile_orders)
        if self.orders.shape != (len(self.images), 9):
            raise ValueError("Tile presentation does not match the image manifest")
        if not np.all(np.sort(self.orders, axis=1) == np.arange(9)):
            raise ValueError("Each tile container must be a permutation")
        if condition not in {"clean", "border_masked"}:
            raise ValueError(condition)
        self.condition = condition

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        tiles = self.images[index]["canonical"][torch.as_tensor(self.orders[index].copy(), dtype=torch.long)]
        return mask_border(tiles) if self.condition == "border_masked" else tiles
