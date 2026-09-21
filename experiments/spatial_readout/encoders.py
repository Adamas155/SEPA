"""Strict student-checkpoint adapters for content-only, independent tile readout."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import shutil
import sys
import tarfile
from types import ModuleType

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = "52c1ae95d05f743e000e8f10a1f3a79b10cff048"
UPSTREAM_ARCHIVE_HASH = "00fa10a14d12a772b9479d1ba0d8422eab2c4db053d01e472c7378b6260fcbd2"
# These three original files are the only upstream Python modules executed here.
UPSTREAM_MODULE_HASHES = {
    "src/models/vision_transformer.py": "69be453a6367460fd0c94588af6531c022a2469ff72e0a793c16fd110dd498c0",
    "src/utils/tensors.py": "66cac858005d01e93ad6f481f384a9d75cb1fb8ad2bff0fd7e9a127f384429c0",
    "src/masks/utils.py": "124da04a7513329a14793389d89a98db87498caf7623a10bc772b097c74d97ec",
}
# Recorded by the existing I-JEPA context() and the original 100-epoch protocol.
TRAIN_FINGERPRINT = "9a7ff7775290a65a26289b713781f92a9373db05261a96d77a58c3aebc4b13fd"
TRAIN_EXAMPLES = 126684
FIXED_PE_INDICES = tuple(r * 15 + c for r in range(5) for c in range(5))


class CheckpointError(ValueError):
    """Weights or their recorded provenance do not satisfy the comparison."""


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state_digest(module):
    """Digest all parameters and persistent buffers, including positional tables."""
    value = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        description = [name, str(tensor.dtype), list(tensor.shape)]
        value.update(json.dumps(description, separators=(",", ":")).encode())
        value.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise CheckpointError(f"Missing or malformed {label}")
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise CheckpointError(f"{label} must be an integer >= {minimum}")
    return value


def _identity(metadata, key="run_id"):
    from sepa_plan_b.config import digest

    _mapping(metadata, "checkpoint metadata")
    actual = digest({k: v for k, v in metadata.items() if k != key})
    if metadata.get(key) != actual:
        raise CheckpointError(f"Checkpoint {key} identity is absent or inconsistent")


def _spec(spec):
    _mapping(spec, "checkpoint specification")
    family, group = spec.get("family"), spec.get("group")
    if family not in {"v1", "ijepa", "v2"} or group not in {"A", "B"}:
        raise CheckpointError("Expected family=v1/ijepa/v2 and group=A/B")
    if (family == "v1" and group != "A") or (family == "v2" and group != "B"):
        raise CheckpointError("V1 belongs to the 100-epoch group; V2 to the 25-epoch group")
    epochs = _integer(spec.get("epochs"), "requested training epochs", 1)
    if epochs != {"A": 100, "B": 25}[group]:
        raise CheckpointError("Requested training epochs disagree with comparison group")
    if _integer(spec.get("pretraining_seed"), "pretraining seed") != 0:
        raise CheckpointError("This comparison is restricted to recorded pretraining seed 0")
    if spec.get("branch", "student") != "student":
        raise CheckpointError("Only the student encoder is allowed")
    if family != "ijepa":
        if _integer(spec.get("k"), "requested k") not in {0, 3}:
            raise CheckpointError("Only the matched k0 and k3 arms are allowed")
    elif spec.get("k") is not None:
        raise CheckpointError("An I-JEPA baseline has no SEPA k arm")
    if not isinstance(spec.get("name"), str) or not spec["name"]:
        raise CheckpointError("A checkpoint name is required")
    if not isinstance(spec.get("path"), (str, Path)) or not str(spec["path"]):
        raise CheckpointError("A real checkpoint path is required; no random fallback exists")


def _progress(checkpoint, config, train_fingerprint, spec, *, official):
    step = _integer(checkpoint.get("step"), "checkpoint step", 1)
    batch = config.get("batch_size") if official else config["training"].get("batch_size")
    batch = _integer(batch, "pretraining batch size", 1)
    if train_fingerprint != TRAIN_FINGERPRINT:
        raise CheckpointError(
            "Unknown pretraining data fingerprint: cannot verify epochs from the recorded dataset size"
        )
    derived = math.ceil(TRAIN_EXAMPLES / batch)
    supplied = _integer(spec.get("steps_per_epoch", 1980), "protocol steps per epoch", 1)
    if supplied != derived:
        raise CheckpointError("Steps per epoch disagree with recorded train data and batch size")
    if step != spec["epochs"] * derived:
        raise CheckpointError(
            f"Checkpoint has {step}/{derived} epochs, expected exactly {spec['epochs']}"
        )
    scheduled = config.get("epochs") * derived if official else config["training"].get("steps")
    if type(scheduled) is not int or step > scheduled:
        raise CheckpointError("Checkpoint step exceeds its configured training schedule")
    if official and (checkpoint.get("lr_step") != step or checkpoint.get("wd_step") != step):
        raise CheckpointError("Checkpoint step and saved optimizer schedules disagree")
    return {
        "step": step,
        "epochs": spec["epochs"],
        "steps_per_epoch": derived,
        "epoch_verification": {
            "method": "checkpoint step divided by ceil(recorded train count / checkpoint batch size)",
            "train_count": TRAIN_EXAMPLES,
            "train_fingerprint": train_fingerprint,
            "batch_size": batch,
            "count_source": "existing server/ijepa_baseline.py context() and 100-epoch protocol",
        },
    }


def _finite_state(state):
    _mapping(state, "student encoder state")
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise CheckpointError("Student encoder state must contain tensors")
    if any(not torch.isfinite(v).all().item() for v in state.values()):
        raise CheckpointError("Student encoder contains nonfinite weights")


def prepare_upstream(archive, destination):
    """Extract the already bundled source archive, without downloading anything."""
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    if file_digest(archive) != UPSTREAM_ARCHIVE_HASH:
        raise CheckpointError("The local upstream archive differs from the recorded snapshot")
    if not destination.is_relative_to(EXPERIMENT_ROOT) or destination == EXPERIMENT_ROOT:
        raise ValueError("Extracted sources must remain inside this independent experiment directory")
    if destination.exists():
        raise FileExistsError(destination)
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isdir() or member.isfile()):
                raise CheckpointError("Unsafe member in the upstream archive")
        if sum(member.size for member in members) > 10 * 1024 * 1024:
            raise CheckpointError("Unexpectedly large upstream source archive")
        destination.mkdir(parents=True)
        try:
            for member in members:
                target = destination / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.extractfile(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
        except Exception:
            shutil.rmtree(destination)
            raise
    return destination


def _verify_upstream(directory, pin):
    _mapping(pin, "upstream source provenance")
    directory = Path(directory).resolve()
    if pin.get("commit") != UPSTREAM_COMMIT:
        raise CheckpointError("Unsupported upstream commit; original source is required")
    if pin.get("archive_sha256") != UPSTREAM_ARCHIVE_HASH:
        raise CheckpointError("Checkpoint refers to a different upstream archive")
    expected = _mapping(pin.get("python_files"), "upstream Python-file pins")
    actual = {}
    for path in directory.rglob("*.py"):
        if not path.resolve().is_relative_to(directory):
            raise CheckpointError("Upstream Python file resolves outside its source directory")
        actual[path.relative_to(directory).as_posix()] = file_digest(path)
    if actual != dict(expected):
        raise CheckpointError("Upstream Python files do not match the checkpoint provenance")
    if any(actual.get(name) != value for name, value in UPSTREAM_MODULE_HASHES.items()):
        raise CheckpointError("The required official encoder modules were modified")
    return directory


def _python_module(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@contextmanager
def _upstream_imports(directory):
    # The upstream files contain absolute src.* imports. Restore all aliases on exit
    # so neither this project nor another independently loaded upstream is replaced.
    names = ("src", "src.utils", "src.masks", "src.utils.tensors", "src.masks.utils")
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in names}
    try:
        for name in names[:3]:
            package = ModuleType(name)
            package.__path__ = [str(directory / name.replace(".", "/"))]
            sys.modules[name] = package
        for name in names[3:]:
            sys.modules[name] = _python_module(name, directory / (name.replace(".", "/") + ".py"))
        yield
    finally:
        for name in names:
            if previous[name] is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]


def _official_constructor(directory, pin):
    directory = _verify_upstream(directory, pin)
    with _upstream_imports(directory):
        module = _python_module(
            "_sepa_spatial_official_vit", directory / "src/models/vision_transformer.py"
        )
    return module.vit_small


class FrozenLocalEncoder(nn.Module):
    """Accept pixels only; each batch element is one independent 80-pixel tile."""

    def __init__(self, encoder, family, manifest):
        super().__init__()
        if family not in {"v1", "ijepa", "v2"}:
            raise ValueError(family)
        self.encoder = encoder.float().requires_grad_(False)
        self.family = family
        self.manifest = manifest
        self.register_buffer("_local_pe_indices", torch.tensor(FIXED_PE_INDICES), persistent=False)
        self.train(False)

    def train(self, mode=True):
        # Calling train() on an outer probe container cannot unfreeze this encoder.
        super().train(False)
        self.encoder.requires_grad_(False)
        return self

    @torch.inference_mode()
    def tokens(self, tiles):
        if not isinstance(tiles, torch.Tensor) or tiles.ndim != 4 or tuple(tiles.shape[1:]) != (3, 80, 80):
            raise ValueError("Independent encoder input must be N x 3 x 80 x 80 pixels only")
        if not tiles.is_floating_point():
            raise ValueError("Input tiles must be preprocessed floating-point pixels")
        self.train(False)
        parameter = next(self.encoder.parameters())
        tiles = tiles.to(device=parameter.device, dtype=torch.float32)
        if not torch.isfinite(tiles).all().item():
            raise ValueError("Input tiles contain nonfinite values")
        if self.family == "v1":
            output = self.encoder.tokens(tiles[:, None])[:, 0]
        else:
            if tuple(self.encoder.pos_embed.shape) != (1, 225, 384):
                raise CheckpointError("Expected the original 15 x 15 ViT-S/16 positional table")
            output = self.encoder.patch_embed(tiles)
            if tuple(output.shape[1:]) != (25, 384):
                raise CheckpointError("Expected exactly 25 independent 384-wide patch tokens")
            # Select a 5x5 square, NOT the first 25 contiguous global positions.
            output = output + self.encoder.pos_embed.index_select(1, self._local_pe_indices)
            for block in self.encoder.blocks:
                output = block(output)
            output = self.encoder.norm(output)
        if tuple(output.shape) != (len(tiles), 25, 384):
            raise CheckpointError("The loaded encoder does not produce N x 25 x 384 tokens")
        if not torch.isfinite(output).all().item():
            raise FloatingPointError("Encoder produced nonfinite features")
        return output

    @torch.inference_mode()
    def forward(self, tiles):
        return self.tokens(tiles).mean(dim=1)


def load_encoder(spec, device="cpu", *, upstream_dir=None):
    """Load an existing student strictly; never reconstruct missing weights randomly."""
    from sepa_plan_b.config import Config
    from sepa_plan_b.engine import load_checkpoint, source_hash
    from sepa_plan_b.model import TileEncoder

    _spec(spec)
    path = Path(spec["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Trained checkpoint is unavailable: {path}")
    actual_hash = file_digest(path)
    if spec.get("expected_file_hash") not in (None, actual_hash):
        raise CheckpointError("Checkpoint file hash differs from the expected existing checkpoint")
    family = spec["family"]
    try:
        checkpoint = load_checkpoint(path) if family == "v1" else torch.load(path, map_location="cpu", weights_only=True)
        checkpoint = _mapping(checkpoint, "checkpoint")
        metadata = _mapping(checkpoint.get("metadata"), "checkpoint metadata")
        _identity(metadata)
        if family == "v1":
            if metadata.get("source_hash") != source_hash():
                raise CheckpointError("V1 core source differs from the checkpoint source")
            config = Config.from_dict(_mapping(metadata.get("config"), "V1 config"))
            if config.model.name != "vit_small" or config.geometry.mode != "native80":
                raise CheckpointError("V1 must be the recorded native80 ViT-S/16 encoder")
            if metadata.get("method") != "sepa" or _integer(metadata.get("k"), "checkpoint k") != spec["k"]:
                raise CheckpointError("V1 method or k arm disagrees with its specification")
            seed = metadata.get("seed")
            if _integer(seed, "checkpoint pretraining seed") != spec["pretraining_seed"]:
                raise CheckpointError("Checkpoint pretraining seed differs from the requested seed")
            progress = _progress(checkpoint, config.to_dict(), metadata.get("train_fingerprint"), spec, official=False)
            full_state = _mapping(checkpoint.get("model"), "V1 model state")
            state = {name.removeprefix("encoder."): value for name, value in full_state.items() if name.startswith("encoder.")}
            _finite_state(state)
            with torch.random.fork_rng(devices=[]):
                encoder = TileEncoder(*config.model.encoder_spec, config.geometry.tile_size)
            pe_protocol = {
                "name": "original_shared_tile_internal_pe",
                "table": "encoder.patch_pe",
                "shape": [25, 384],
                "global_tile_position_used": False,
            }
            model_config = config.to_dict()
            provenance = {"core_source_hash": metadata["source_hash"], "core_source_verified": True}
        else:
            if family == "v2":
                shared = _mapping(metadata.get("shared"), "V2 shared protocol")
                _identity(shared, "screen_id")
                baseline = _mapping(shared.get("baseline"), "V2 baseline protocol")
                _identity(baseline)
                config = _mapping(shared.get("baseline_config"), "V2 encoder config")
                v2_config = _mapping(shared.get("v2_config"), "V2 branch config")
                if shared.get("experiment") != "sepa_v2_25epoch_screen" or v2_config.get("screen_epochs") != 25:
                    raise CheckpointError("Unexpected V2 experiment or screen duration")
                if config != baseline.get("config") or metadata.get("k") != spec["k"] or metadata.get("arm") != f"k{spec['k']}":
                    raise CheckpointError("V2 baseline config or k arm is inconsistent")
                if v2_config.get("seed") != config.get("seed"):
                    raise CheckpointError("V2 branch and baseline pretraining seeds disagree")
                model_config = {"baseline_config": dict(config), "v2_config": dict(v2_config)}
            else:
                baseline = metadata
                config = _mapping(metadata.get("config"), "I-JEPA config")
                model_config = dict(config)
            if config.get("model") != "vit_small" or config.get("image_size") != 240 or config.get("patch_size") != 16:
                raise CheckpointError("Only the recorded official ViT-S/16 at 240 pixels is supported")
            if config.get("upstream_commit") != UPSTREAM_COMMIT:
                raise CheckpointError("Checkpoint model source differs from the recorded official encoder")
            seed = config.get("seed")
            if _integer(seed, "checkpoint pretraining seed") != spec["pretraining_seed"]:
                raise CheckpointError("Checkpoint pretraining seed differs from the requested seed")
            progress = _progress(checkpoint, config, baseline.get("train_fingerprint"), spec, official=True)
            state = _mapping(checkpoint.get("encoder"), "official student encoder state")
            _finite_state(state)
            if upstream_dir is None:
                raise FileNotFoundError("Provide the extracted bundled I-JEPA source as upstream_dir; no automatic download is performed")
            constructor = _official_constructor(upstream_dir, baseline.get("upstream"))
            with torch.random.fork_rng(devices=[]):
                encoder = constructor(img_size=[240], patch_size=16)
            pe_protocol = {
                "name": "fixed_top_left_5x5_window_of_original_15x15_pe",
                "table": "encoder.pos_embed",
                "indices": list(FIXED_PE_INDICES),
                "same_window_for_every_tile": True,
                "global_tile_position_used": False,
                "matches_v2_training_local_branch": family == "v2",
                "input_distribution_shift_for_global_ijepa": family == "ijepa",
            }
            provenance = {"upstream_directory": str(Path(upstream_dir).resolve()), "upstream": baseline["upstream"], "upstream_source_verified": True}
        pe_key = "patch_pe" if family == "v1" else "pos_embed"
        if pe_key not in state or not torch.equal(state[pe_key], encoder.state_dict()[pe_key]):
            raise CheckpointError("The fixed positional table differs from the original training encoder")
        encoder.load_state_dict(state, strict=True)
    except CheckpointError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, pickle.UnpicklingError) as error:
        raise CheckpointError(f"Malformed or incompatible {family} student checkpoint: {error}") from error
    encoder = encoder.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    manifest = {
        "name": spec["name"], "family": family, "group": spec["group"],
        "path": str(path), "file_hash": actual_hash, "file_bytes": path.stat().st_size,
        "branch": "student", "pretraining_seed": seed, "k": metadata.get("k"),
        **progress,
        "model_config": model_config, "metadata": dict(metadata),
        "pe_protocol": pe_protocol, "source_provenance": provenance,
        "input_shape": [3, 80, 80], "patch_tokens": 25, "feature_dim": 384,
        "pooling": "mean of all 25 output patch tokens", "precision": "fp32",
        "checkpoint_verified": True, "requires_grad": False,
        "state_hash": state_digest(encoder),
        "interpretation": "spatial readability under content-only independent local readout; not a full missing+shuffle recovery test",
    }
    return FrozenLocalEncoder(encoder, family, manifest).to(device)
