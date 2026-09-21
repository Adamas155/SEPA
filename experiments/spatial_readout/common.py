"""Artifact identities and the fixed evaluation protocol."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

PREPROCESS = {
    "crop": "center square; integer floor offsets",
    "resize": "Pillow bicubic 240x240",
    "grid": [3, 3],
    "tile_size": [80, 80],
    "patch_size": [16, 16],
    "tokens_per_tile": 25,
    "encoder_feature_dim": 384,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "rotation": False,
    "padding": False,
    "mask": "none during clean readout; all nine tiles",
    "occlusion": "outer 8 pixels set to normalized zero (fixed RGB mean)",
}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=json_default) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_identity():
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})
