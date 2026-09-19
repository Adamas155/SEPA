from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from .config import Config, digest
from .geometry import sample_layout, split_tiles
from .rng import local_rng, generator

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_manifest(root, classes, role, limit_per_class=0):
    root = Path(root).resolve()
    records = []
    for label, name in enumerate(classes):
        directory = root / name
        if not directory.is_dir():
            raise ValueError(f"Missing class directory: {directory}")
        paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if limit_per_class:
            paths = paths[:limit_per_class]
        if not paths:
            raise ValueError(f"Empty class: {directory}")
        for p in paths:
            data = p.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            records.append({"path": p.relative_to(root).as_posix(), "label": label,
                            "sha256": sha, "id": sha, "bytes": len(data)})
    payload = {"schema": 1, "role": role, "root": str(root), "classes": classes, "records": records}
    payload["fingerprint"] = digest(payload)
    return payload


def prepare(train_root, val_root, out_dir, class_list=None, expected_classes=100, limit_per_class=0):
    if expected_classes < 2 or limit_per_class < 0:
        raise ValueError("Expected at least two classes and a nonnegative image limit")
    out_dir = Path(out_dir)
    if any((out_dir / n).exists() for n in ("train.json", "val.json")):
        raise FileExistsError("Manifest destination already exists; use a new directory")
    if class_list:
        classes = [s.strip() for s in Path(class_list).read_text(encoding="utf-8-sig").splitlines()
                   if s.strip() and not s.lstrip().startswith("#")]
    else:
        classes = sorted(p.name for p in Path(train_root).iterdir() if p.is_dir())
    if len(classes) != expected_classes or len(set(classes)) != len(classes):
        raise ValueError(f"Expected {expected_classes} distinct classes; found {len(classes)}")
    if any(Path(n).name != n or n in {".", ".."} for n in classes):
        raise ValueError("Class names must be plain directory names")
    train = build_manifest(train_root, classes, "train", limit_per_class)
    val = build_manifest(val_root, classes, "val", limit_per_class)
    validate_pair(train, val, expected_classes)
    write_json(out_dir / "train.json", train)
    write_json(out_dir / "val.json", val)
    return {"train": len(train["records"]), "val": len(val["records"]), "classes": len(classes),
            "directory": str(out_dir.resolve())}


def load_manifest(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    fingerprint = result.get("fingerprint")
    payload = {k: v for k, v in result.items() if k != "fingerprint"}
    if result.get("schema") != 1 or digest(payload) != fingerprint:
        raise ValueError(f"Manifest identity mismatch: {path}")
    if not result["records"]:
        raise ValueError("Empty manifest")
    return result


def validate_pair(train, val, expected_classes):
    if train["role"] != "train" or val["role"] != "val":
        raise ValueError("Manifest roles must be train and val")
    if train["classes"] != val["classes"] or len(train["classes"]) != expected_classes:
        raise ValueError("Train/val class mappings differ or class count is incorrect")
    overlap = {x["sha256"] for x in train["records"]} & {x["sha256"] for x in val["records"]}
    if overlap:
        raise ValueError(f"Train/val contain {len(overlap)} identical image files")
    for manifest in (train, val):
        if set(x["label"] for x in manifest["records"]) != set(range(expected_classes)):
            raise ValueError("Every class must have data and labels must match the class map")


def manifests(config):
    if not config.data.train_manifest or not config.data.val_manifest:
        raise ValueError("Prepare data and set train_manifest and val_manifest in the TOML config")
    train, val = load_manifest(config.data.train_manifest), load_manifest(config.data.val_manifest)
    validate_pair(train, val, config.data.expected_classes)
    return train, val


def augment(image, config, rng, training):
    """All randomness is local to (seed, sample identity, epoch), never worker state."""
    width, height = image.size
    if training:
        area = width * height
        box = None
        for _ in range(10):
            target = rng.uniform(config.data.crop_min, config.data.crop_max) * area
            aspect = math.exp(rng.uniform(math.log(3 / 4), math.log(4 / 3)))
            w, h = round(math.sqrt(target * aspect)), round(math.sqrt(target / aspect))
            if 0 < w <= width and 0 < h <= height:
                left, top = rng.randrange(width - w + 1), rng.randrange(height - h + 1)
                box = (left, top, left + w, top + h)
                break
        if box is None:
            size = min(width, height)
            left, top = (width-size)//2, (height-size)//2
            box = (left, top, left+size, top+size)
    else:
        # Fixed center square field of view; identical across geometry candidates.
        size = min(width, height)
        left, top = (width-size)//2, (height-size)//2
        box = (left, top, left+size, top+size)
    image = image.crop(box).resize((config.geometry.image_size,) * 2, Image.Resampling.BICUBIC)
    if training:
        if rng.random() < config.data.horizontal_flip:
            image = ImageOps.mirror(image)
    array = np.array(image, dtype=np.float32, copy=True) / 255.0
    if training:
        jitter = config.data.color_jitter
        brightness, contrast, saturation = [rng.uniform(1-jitter, 1+jitter) for _ in range(3)]
        array = (array * brightness - 0.5) * contrast + 0.5
        # Pixel-local color transform: no whole-image mean from hidden tiles.
        gray = (array * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(-1, keepdims=True)
        array = np.clip(gray + saturation * (array-gray), 0, 1)
    return torch.from_numpy(array).permute(2, 0, 1)


class ImageTiles(Dataset):
    def __init__(self, manifest, config: Config, seed=0, k=0, training=False):
        self.manifest, self.config, self.seed, self.k, self.training = manifest, config, seed, k, training
        self.root = Path(manifest["root"]).resolve()

    def __len__(self):
        return len(self.manifest["records"])

    def __getitem__(self, item):
        index, epoch = item if isinstance(item, tuple) else (item, 0)
        record = self.manifest["records"][index]
        path = (self.root / record["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Image path escapes manifest root")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Image changed after manifest preparation: {path}")
        with Image.open(io.BytesIO(data)) as im:
            pixels = augment(im.convert("RGB"), self.config,
                             local_rng(self.seed, "augmentation", epoch, record["id"]), self.training)
        canonical = split_tiles(pixels, self.config.geometry)
        result = {"canonical": canonical, "label": record["label"], "id": record["id"]}
        if self.training:
            layout = sample_layout(self.seed, epoch, record["id"], self.k)
            result.update(visible=canonical[layout.canonical_ids], observed_slots=layout.visible_slots,
                          query_slots=layout.query_slots, canonical_ids=layout.canonical_ids,
                          moved_count=layout.moved_count)
        return result


def epoch_batches(size, batch_size, seed, epoch, skip=0):
    order = torch.randperm(size, generator=generator(seed, "data_order", epoch)).tolist()
    for start in range(skip * batch_size, size, batch_size):
        yield [(index, epoch) for index in order[start:start+batch_size]]
