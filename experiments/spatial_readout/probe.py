"""Small content-only pair probes with train statistics and dev-only selection.

Feature arrays have shape ``[images, 9, feature_dim]`` in the container order
chosen by the split manifest. Labels align with ``tasks.ordered_pairs()``;
neither image metadata nor tile coordinates are accepted by the head. Only
one image batch of concatenated pairs is constructed at a time.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .metrics import adjacency_metrics, select_thresholds
from .tasks import ordered_pairs


DEFAULT_SETTINGS = {
    "head_kind": "linear",
    "learning_rates": [0.01],
    "epochs": 30,
    "patience": 5,
    "batch_images": 32,
    "weight_decay": 0.0,
    "std_floor": 1e-6,
}


class PairHead(nn.Module):
    """A fixed linear or 128-unit shallow head accepting only content pairs."""

    def __init__(self, feature_dim: int, head_kind: str = "linear") -> None:
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
            raise TypeError("feature_dim must be a positive integer")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = feature_dim
        self.input_dim = 2 * feature_dim
        self.head_kind = head_kind
        if head_kind == "linear":
            self.layers = nn.Linear(self.input_dim, 2)
        elif head_kind == "mlp":
            self.layers = nn.Sequential(
                nn.Linear(self.input_dim, 128), nn.GELU(), nn.Linear(128, 2)
            )
        else:
            raise ValueError("head_kind must be 'linear' or 'mlp'")

    def forward(self, content_pairs: torch.Tensor) -> torch.Tensor:
        if not isinstance(content_pairs, torch.Tensor):
            raise TypeError("PairHead accepts only a content tensor, not metadata")
        if content_pairs.ndim != 2 or content_pairs.shape[1] != self.input_dim:
            raise ValueError(f"content_pairs must have shape [batch, {self.input_dim}]")
        return self.layers(content_pairs)


def parameter_count(feature_dim: int, head_kind: str = "linear") -> int:
    """Return exact capacity without creating a model or changing the RNG."""
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
        raise TypeError("feature_dim must be a positive integer")
    if feature_dim < 1:
        raise ValueError("feature_dim must be positive")
    if head_kind == "linear":
        return 2 * (2 * feature_dim) + 2
    if head_kind == "mlp":
        return (2 * feature_dim) * 128 + 128 + 128 * 2 + 2
    raise ValueError("head_kind must be 'linear' or 'mlp'")


def _features_shape(features: np.ndarray, name: str) -> tuple[int, int]:
    if not isinstance(features, np.ndarray):
        raise TypeError(f"{name} must be a numpy feature array or memmap")
    if features.ndim != 3 or features.shape[1] != 9 or features.shape[2] < 1:
        raise ValueError(f"{name} must have shape [images, 9, feature_dim]")
    if features.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one image")
    if features.dtype.kind != "f":
        raise TypeError(f"{name} must contain floating-point content features")
    return int(features.shape[0]), int(features.shape[2])


def _label_counts(labels: np.ndarray, images: int, name: str) -> np.ndarray:
    if not isinstance(labels, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if labels.shape != (images, 72, 2):
        raise ValueError(f"{name} must have shape [{images}, 72, 2]")
    if labels.dtype.kind not in "buif":
        raise TypeError(f"{name} must contain binary labels")
    positives = np.zeros(2, dtype=np.int64)
    for start in range(0, images, 256):
        chunk = labels[start : start + 256]
        if not np.all((chunk == 0) | (chunk == 1)):
            raise ValueError(f"{name} must contain only zero/one labels")
        if not np.all(np.sum(chunk, axis=1) == 6):
            raise ValueError(f"{name} needs six positives per image and direction")
        positives += np.sum(chunk, axis=(0, 1), dtype=np.int64)
    if np.any(positives == 0) or np.any(positives == images * 72):
        raise ValueError(f"{name} needs positive and negative examples per direction")
    return positives


def training_statistics(
    train_features: np.ndarray, *, chunk_images: int = 256, std_floor: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Compute population moments over train tiles only using streamed chunks."""
    images, dim = _features_shape(train_features, "train_features")
    if isinstance(chunk_images, bool) or not isinstance(chunk_images, int):
        raise TypeError("chunk_images must be a positive integer")
    if chunk_images < 1 or not math.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("chunk_images and std_floor must be positive")
    count = 0
    mean = np.zeros(dim, dtype=np.float64)
    m2 = np.zeros(dim, dtype=np.float64)
    for start in range(0, images, chunk_images):
        chunk = np.asarray(
            train_features[start : start + chunk_images], dtype=np.float64
        ).reshape(-1, dim)
        if not np.all(np.isfinite(chunk)):
            raise ValueError("train_features contains non-finite values")
        chunk_count = len(chunk)
        chunk_mean = chunk.mean(axis=0)
        chunk_m2 = np.square(chunk - chunk_mean).sum(axis=0)
        delta = chunk_mean - mean
        combined = count + chunk_count
        mean += delta * (chunk_count / combined)
        m2 += chunk_m2 + np.square(delta) * count * chunk_count / combined
        count = combined
    std = np.maximum(np.sqrt(np.maximum(m2 / count, 0.0)), std_floor)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("training statistics cannot be represented in float32")
    return mean, std


def _settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    settings = {**DEFAULT_SETTINGS, **(config or {})}
    unexpected = set(settings) - set(DEFAULT_SETTINGS)
    if unexpected:
        raise ValueError(f"Unknown probe config keys: {sorted(unexpected)}")
    if settings["head_kind"] not in {"linear", "mlp"}:
        raise ValueError("head_kind must be 'linear' or 'mlp'")
    for key in ("epochs", "patience", "batch_images"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    rates = settings["learning_rates"]
    if not isinstance(rates, (list, tuple)) or not rates:
        raise ValueError("learning_rates must be a non-empty declared list")
    rates = [float(rate) for rate in rates]
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("learning_rates must contain finite positive values")
    if len(set(rates)) != len(rates):
        raise ValueError("learning_rates must not repeat a candidate")
    settings["learning_rates"] = rates
    for key in ("weight_decay", "std_floor"):
        settings[key] = float(settings[key])
        if not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if settings["std_floor"] <= 0:
        raise ValueError("std_floor must be positive")
    return settings


def _new_head(feature_dim: int, head_kind: str, seed: int, device: torch.device):
    # All random initialization is on CPU; the caller's CPU/CUDA RNG is unchanged.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = PairHead(feature_dim, head_kind)
    return head.to(device)


def _batch_pairs(
    features: np.ndarray,
    indices: np.ndarray | slice,
    mean: torch.Tensor,
    std: torch.Tensor,
    pairs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    # Copies only this image batch, including for read-only memmaps. The resulting
    # tensor has no gradient connection to an encoder or a cached tensor graph.
    content = np.array(features[indices], dtype=np.float32, copy=True)
    if not np.all(np.isfinite(content)):
        raise ValueError("features contain non-finite float32 values")
    tiles = (torch.from_numpy(content).to(device) - mean) / std
    paired = torch.cat((tiles[:, pairs[:, 0]], tiles[:, pairs[:, 1]]), dim=-1)
    return paired.reshape(-1, 2 * features.shape[2])


def _score_head(
    head: PairHead,
    features: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    pairs: torch.Tensor,
    device: torch.device,
    batch_images: int,
) -> np.ndarray:
    predictions = np.empty((len(features), 72, 2), dtype=np.float32)
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_images):
            end = min(start + batch_images, len(features))
            inputs = _batch_pairs(features, slice(start, end), mean, std, pairs, device)
            logits = head(inputs)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("probe produced non-finite logits")
            predictions[start:end] = logits.reshape(-1, 72, 2).cpu().numpy()
    return predictions


def fit_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    dev_features: np.ndarray,
    dev_labels: np.ndarray,
    *,
    seed: int,
    config: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    progress=None,
) -> dict[str, Any]:
    """Fit only a lightweight head; this API deliberately has no test inputs.

    BCE positive weights equal train negatives / train positives independently
    for V and H. With all 72 natural pairs, each weight is 66/6 = 11. Epoch and
    learning-rate selection maximize dev macro AP; exact ties choose the first
    declared learning rate, then its earliest best epoch. Dev alone supplies
    the two balanced-accuracy thresholds after selecting the head.
    """
    settings = _settings(config)
    fit_started = time.monotonic()
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    train_images, dim = _features_shape(train_features, "train_features")
    dev_images, dev_dim = _features_shape(dev_features, "dev_features")
    if dim != dev_dim:
        raise ValueError("train and dev features must have the same feature dimension")
    positives = _label_counts(train_labels, train_images, "train_labels")
    _label_counts(dev_labels, dev_images, "dev_labels")
    mean, std = training_statistics(train_features, std_floor=settings["std_floor"])
    pos_weight = ((train_images * 72 - positives) / positives).astype(np.float32)
    device = torch.device(device)
    torch_mean = torch.tensor(mean, device=device)
    torch_std = torch.tensor(std, device=device)
    pairs = torch.tensor(ordered_pairs(), dtype=torch.long, device=device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)
    )
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    batch_images = settings["batch_images"]
    for lr_index, learning_rate in enumerate(settings["learning_rates"]):
        head = _new_head(dim, settings["head_kind"], seed, device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=learning_rate, weight_decay=settings["weight_decay"]
        )
        best_for_rate = -math.inf
        stale_epochs = 0
        # Reset for every LR: a seed supplies the same image order to every arm.
        order_rng = np.random.default_rng(seed)
        for epoch in range(1, settings["epochs"] + 1):
            epoch_started = time.monotonic()
            head.train()
            order = order_rng.permutation(train_images)
            total_loss = 0.0
            for start in range(0, train_images, batch_images):
                indices = order[start : start + batch_images]
                inputs = _batch_pairs(
                    train_features, indices, torch_mean, torch_std, pairs, device
                )
                labels = torch.from_numpy(
                    np.array(train_labels[indices], dtype=np.float32, copy=True)
                ).to(device).reshape(-1, 2)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(inputs), labels)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("probe training produced a non-finite loss")
                loss.backward()
                if any(
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                    for parameter in head.parameters()
                ):
                    raise ValueError("probe training produced a non-finite gradient")
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
            dev_scores = _score_head(
                head, dev_features, torch_mean, torch_std, pairs, device, batch_images
            )
            dev_ap = float(adjacency_metrics(dev_scores, dev_labels)["macro"]["ap"])
            if not math.isfinite(dev_ap):
                raise ValueError("dev macro AP must be finite")
            improved = dev_ap > best_for_rate
            history.append(
                {
                    "learning_rate_index": lr_index,
                    "learning_rate": learning_rate,
                    "epoch": epoch,
                    "train_loss": total_loss / train_images,
                    "dev_macro_ap": dev_ap,
                    "improved_for_learning_rate": improved,
                }
            )
            if progress is not None:
                progress({**history[-1], "epoch_seconds": time.monotonic() - epoch_started})
            if improved:
                best_for_rate = dev_ap
                stale_epochs = 0
            else:
                stale_epochs += 1
            if best is None or dev_ap > best["selected_dev_macro_ap"]:
                best = {
                    "state_dict": {
                        name: value.detach().cpu().clone()
                        for name, value in head.state_dict().items()
                    },
                    "selected_epoch": epoch,
                    "selected_learning_rate": learning_rate,
                    "selected_learning_rate_index": lr_index,
                    "selected_dev_macro_ap": dev_ap,
                }
            if stale_epochs >= settings["patience"]:
                break
    if best is None:
        raise RuntimeError("No probe candidate was trained")
    selected = _new_head(dim, settings["head_kind"], seed, device)
    selected.load_state_dict(best["state_dict"], strict=True)
    dev_scores = _score_head(
        selected, dev_features, torch_mean, torch_std, pairs, device, batch_images
    )
    thresholds = np.asarray(select_thresholds(dev_scores, dev_labels), dtype=np.float64)
    if thresholds.shape != (2,) or not np.all(np.isfinite(thresholds)):
        raise ValueError("Dev thresholds must have shape [2] and be finite")
    return {
        **best,
        "schema_version": 1,
        "seed": seed,
        "head_kind": settings["head_kind"],
        "feature_dim": dim,
        "input_dim": 2 * dim,
        "parameter_count": parameter_count(dim, settings["head_kind"]),
        "mean": mean,
        "std": std,
        "thresholds": thresholds,
        "pos_weight": pos_weight,
        "train_positive_counts": positives,
        "train_pair_count": train_images * 72,
        "train_image_count": train_images,
        "dev_image_count": dev_images,
        "history": history,
        "fit_seconds": time.monotonic() - fit_started,
        "settings": settings,
        "selection_rule": "maximum dev macro AP; ties: declared LR order, earliest epoch",
        "standardization_source": "probe-train tiles only; population standard deviation",
        "threshold_source": "probe-dev balanced accuracy only",
        "probe_seed_scope": "head initialization and train image shuffling only",
    }


def predict_scores(
    fitted: Mapping[str, Any],
    features: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_images: int = 32,
) -> np.ndarray:
    """Apply the frozen selected head and train statistics without adaptation."""
    _, dim = _features_shape(features, "features")
    if dim != fitted["feature_dim"] or 2 * dim != fitted["input_dim"]:
        raise ValueError("Feature dimension does not match the fitted probe")
    if isinstance(batch_images, bool) or not isinstance(batch_images, int) or batch_images < 1:
        raise ValueError("batch_images must be a positive integer")
    mean = np.asarray(fitted["mean"], dtype=np.float32)
    std = np.asarray(fitted["std"], dtype=np.float32)
    if mean.shape != (dim,) or std.shape != (dim,):
        raise ValueError("Fitted train statistics have incorrect dimensions")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("Fitted train statistics must be finite with positive std")
    device = torch.device(device)
    head = _new_head(dim, fitted["head_kind"], int(fitted["seed"]), device)
    head.load_state_dict(fitted["state_dict"], strict=True)
    head.requires_grad_(False)
    return _score_head(
        head,
        features,
        torch.tensor(mean, device=device),
        torch.tensor(std, device=device),
        torch.tensor(ordered_pairs(), dtype=torch.long, device=device),
        device,
        batch_images,
    )
