"""Independent tile extraction, compact pixel controls, and bound feature caches."""

import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import PREPROCESS, code_identity, digest, file_hash, read_json, write_json
from .encoders import state_digest


class PixelFeatures(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in {"color_lowfreq", "border_only"}:
            raise ValueError(kind)
        self.kind = kind
        self.dimension = 54 if kind == "color_lowfreq" else 192
        self.manifest = {
            "family": "pixel_control", "name": kind, "feature_dim": self.dimension,
            "pe_protocol": "none; tile pixels only",
            "protocol": ("RGB mean, population std, adaptive average 4x4 thumbnail"
                         if kind == "color_lowfreq" else
                         "outer 8px only; interior fixed RGB preprocessing mean; adaptive average 8x8"),
        }

    @torch.inference_mode()
    def forward(self, tiles):
        if tiles.ndim != 4 or tiles.shape[1:] != (3, 80, 80):
            raise ValueError("Pixel controls require independent 80x80 RGB tiles")
        if self.kind == "border_only":
            pixels = tiles.clone()
            pixels[:, :, 8:-8, 8:-8] = 0
        else:
            pixels = tiles
        mean = pixels.new_tensor(PREPROCESS["mean"])[None, :, None, None]
        std = pixels.new_tensor(PREPROCESS["std"])[None, :, None, None]
        raw = pixels * std + mean
        if self.kind == "border_only":
            return F.adaptive_avg_pool2d(raw, 8).flatten(1)
        return torch.cat((raw.mean((-1, -2)), raw.std((-1, -2), correction=0),
                          F.adaptive_avg_pool2d(raw, 4).flatten(1)), dim=1)


def cache_identity(*, model_manifest, split_manifest, design_hash, condition):
    if condition not in {"clean", "border_masked"}:
        raise ValueError(condition)
    return {
        "schema": 1, "model": model_manifest, "split_fingerprint": split_manifest["fingerprint"],
        "design_hash": design_hash, "preprocess": PREPROCESS,
        "condition": condition, "dtype": "float32", "code_identity": code_identity(),
        "feature_pooling": "mean of 25 patch tokens" if model_manifest.get("family") != "pixel_control" else model_manifest["protocol"],
    }


def open_cache(path, identity, shape):
    path = Path(path)
    receipt = read_json(path.with_suffix(".json"))
    if receipt["identity"] != identity or receipt["shape"] != list(shape):
        raise ValueError("Feature cache identity mismatch")
    if receipt["file_hash"] != file_hash(path):
        raise ValueError("Feature cache bytes changed")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != tuple(shape) or array.dtype != np.float32:
        raise ValueError("Feature cache layout mismatch")
    for start in range(0, len(array), 1024):
        if not np.isfinite(array[start:start + 1024]).all():
            raise ValueError("Nonfinite feature cache")
    return array, receipt


def extract_cached(model, dataset, path, identity, *, dimension, device="cpu",
                   batch_images=16, tile_batch_size=128, workers=0):
    path = Path(path)
    shape = (len(dataset), 9, dimension)
    if path.exists() or path.with_suffix(".json").exists():
        return open_cache(path, identity, shape)
    if not len(dataset) or min(batch_images, tile_batch_size) < 1 or workers < 0:
        raise ValueError("Invalid extraction sizes")
    path.parent.mkdir(parents=True, exist_ok=True)
    needed = int(np.prod(shape)) * 4
    if shutil.disk_usage(path.parent).free < needed + 128 * 1024**2:
        raise OSError(f"Not enough space for {needed} bytes of new FP32 features")
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Encoder parameters must be frozen before extraction")
    model.eval()
    before = state_digest(model)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Incomplete extraction exists; use a fresh output: {temporary}")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(dataset, batch_size=batch_images, shuffle=False, num_workers=workers,
                        pin_memory=str(device).startswith("cuda"), generator=generator)
    start = time.monotonic()
    offset, last_print = 0, start
    with torch.inference_mode():
        for tiles in loader:
            flat = tiles.flatten(0, 1)
            values = []
            for batch in flat.split(tile_batch_size):
                encoded = model(batch.to(device, non_blocking=True))
                if encoded.shape != (len(batch), dimension) or not torch.isfinite(encoded).all():
                    raise ValueError("Encoder output shape or finite check failed")
                if encoded.requires_grad:
                    raise ValueError("Extraction unexpectedly created gradients")
                values.append(encoded.cpu().float().numpy())
            n = len(tiles)
            array[offset:offset + n] = np.concatenate(values).reshape(n, 9, dimension)
            offset += n
            if time.monotonic() - last_print > 20:
                print(f"extract {path.stem}: {offset}/{len(dataset)} images", flush=True)
                last_print = time.monotonic()
    if offset != len(dataset) or state_digest(model) != before:
        raise RuntimeError("Encoder changed or extraction did not cover every image")
    array.flush()
    del array
    os.replace(temporary, path)
    receipt = {
        "identity": identity, "shape": list(shape), "file_hash": file_hash(path),
        "encoder_state_before": before, "encoder_state_after": state_digest(model),
        "encoder_gradients": False, "seconds": time.monotonic() - start,
        "images": len(dataset), "tiles": len(dataset) * 9,
    }
    write_json(path.with_suffix(".json"), receipt)
    return np.load(path, mmap_mode="r", allow_pickle=False), receipt


def save_design(path, design):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Indices fit in uint8; compression preserves exact labels/order.
    compact = {k: (v.astype(np.uint8) if isinstance(v, np.ndarray) and v.dtype.kind in "iu" else v)
               for k, v in design.items()}
    np.savez_compressed(path, **compact)
    return file_hash(path)


def load_design(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def save_fitted(path, fitted):
    def safe(value):
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [safe(v) for v in value]
        return value
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    torch.save(safe(fitted), path)


def load_fitted(path):
    fitted = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("mean", "std", "thresholds", "pos_weight", "train_positive_counts"):
        if key in fitted and isinstance(fitted[key], torch.Tensor):
            fitted[key] = fitted[key].numpy()
    return fitted
