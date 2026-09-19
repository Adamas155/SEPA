from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from pathlib import Path
import tomllib


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Geometry:
    mode: str = "native80"
    padding: str = "reflect"

    @property
    def image_size(self):
        return 224 if self.mode == "pad74" else 240 if self.mode == "native80" else 192

    @property
    def tile_size(self):
        return 64 if self.mode == "native64" else 80

    @property
    def real_tile_size(self):
        return 74 if self.mode == "pad74" else self.tile_size


@dataclass(frozen=True)
class Model:
    name: str = "vit_small"
    predictor_dim: int = 192
    predictor_depth: int = 2
    predictor_heads: int = 6
    relation: str = "none"
    relation_hidden: int = 128
    relation_weight: float = 0.0

    @property
    def encoder_spec(self):
        return {"vit_small": (384, 12, 6), "vit_base": (768, 12, 12),
                "tiny": (32, 2, 4)}[self.name]


@dataclass(frozen=True)
class Data:
    train_manifest: str = ""
    val_manifest: str = ""
    expected_classes: int = 100
    crop_min: float = 0.3
    crop_max: float = 1.0
    horizontal_flip: float = 0.5
    color_jitter: float = 0.2


@dataclass(frozen=True)
class Training:
    steps: int = 20000
    batch_size: int = 64
    learning_rate: float = 0.0001
    weight_decay: float = 0.04
    warmup_steps: int = 500
    minimum_lr_ratio: float = 0.01
    ema_start: float = 0.996
    clip_grad: float = 1.0
    checkpoint_every: int = 500
    log_every: int = 50
    num_workers: int = 0
    device: str = "auto"
    precision: str = "fp32"
    deterministic: bool = True
    collapse_ratio: float = 0.05
    collapse_min_std: float = 0.0001
    collapse_min_image_std: float = 0.0001
    output_root: str = "runs"


@dataclass(frozen=True)
class Probe:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 0.05
    weight_decay: float = 0.0
    label_fraction: float = 1.0
    knn_k: int = 20
    knn_temperature: float = 0.07
    spatial_steps: int = 300
    spatial_learning_rate: float = 0.01


@dataclass(frozen=True)
class Config:
    geometry: Geometry = field(default_factory=Geometry)
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)
    training: Training = field(default_factory=Training)
    probe: Probe = field(default_factory=Probe)

    def validate(self):
        g, m, d, t, p = self.geometry, self.model, self.data, self.training, self.probe
        if g.mode not in {"native80", "native64", "pad74"} or g.padding not in {"reflect", "replicate", "constant"}:
            raise ValueError("Unknown geometry or padding mode")
        if m.name not in {"tiny", "vit_small", "vit_base"}:
            raise ValueError("model.name must be tiny, vit_small, or vit_base")
        if m.predictor_dim <= 0 or m.predictor_dim % 4 or m.predictor_heads < 1 or m.predictor_dim % m.predictor_heads:
            raise ValueError("predictor_dim must be divisible by 4 and predictor_heads")
        if m.predictor_depth < 1 or m.relation_hidden < 1:
            raise ValueError("Invalid predictor/head size")
        if m.relation not in {"none", "undirected", "directed", "aggregation"}:
            raise ValueError("Unknown relation mode")
        if m.relation_weight < 0 or (m.relation == "none" and m.relation_weight != 0):
            raise ValueError("relation_weight must be nonnegative and zero when relation=none")
        if not 0 < d.crop_min <= d.crop_max <= 1 or not 0 <= d.horizontal_flip <= 1 or not 0 <= d.color_jitter <= 1:
            raise ValueError("Invalid augmentation range")
        if d.expected_classes < 2:
            raise ValueError("expected_classes must be at least 2")
        if min(t.steps, t.batch_size, t.checkpoint_every, t.log_every) < 1 or t.num_workers < 0:
            raise ValueError("Invalid training count")
        if t.learning_rate <= 0 or t.weight_decay < 0 or t.clip_grad <= 0 or not 0 <= t.warmup_steps < t.steps:
            raise ValueError("Invalid optimizer/schedule configuration")
        if not 0 < t.minimum_lr_ratio <= 1 or not 0 <= t.ema_start <= 1 or not 0 <= t.collapse_ratio <= 1:
            raise ValueError("Invalid learning rate/EMA/collapse range")
        if min(t.collapse_min_std, t.collapse_min_image_std) <= 0:
            raise ValueError("Collapse standard deviation thresholds must be positive")
        if t.precision not in {"fp32", "bf16"} or t.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Supported runtime: auto/cpu/cuda, fp32/bf16")
        if min(p.epochs, p.batch_size, p.knn_k, p.spatial_steps) < 1 or not 0 < p.label_fraction <= 1:
            raise ValueError("Invalid probe configuration")
        if min(p.learning_rate, p.knn_temperature, p.spatial_learning_rate) <= 0 or p.weight_decay < 0:
            raise ValueError("Invalid probe optimizer")
        # Reject NaN/infinity anywhere, including values loaded from TOML.
        digest(asdict(self))
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        types = {"geometry": Geometry, "model": Model, "data": Data, "training": Training, "probe": Probe}
        if set(raw) - set(types):
            raise ValueError(f"Unknown config sections: {set(raw) - set(types)}")
        kwargs = {}
        for name, values in raw.items():
            unknown = set(values) - {f.name for f in fields(types[name])}
            if unknown:
                raise ValueError(f"Unknown {name} fields: {unknown}")
            kwargs[name] = types[name](**values)
        return cls(**kwargs).validate()

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        raw.setdefault("training", {})
        raw["training"].setdefault("output_root", "../runs")
        for section, key in (("data", "train_manifest"), ("data", "val_manifest"), ("training", "output_root")):
            value = raw.get(section, {}).get(key)
            if value:
                raw[section][key] = str((path.parent / value).resolve())
        return cls.from_dict(raw)
