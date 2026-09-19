from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from .config import Config, digest
from .data import ImageTiles, epoch_batches, manifests, write_json
from .diagnostics import position_sensitivity
from .geometry import VALID_K
from .losses import diagnostics, latent_loss, relation_loss
from .model import SEPA
from .rng import generator, seed_for


def source_hash():
    files = sorted(Path(__file__).parent.glob("*.py"))
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def device_for(config):
    requested = config.training.device
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else
                          "cpu" if requested == "auto" else requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if config.training.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 mode requires a CUDA device with bfloat16 support; use fp32 on CPU")
    return device


def setup_runtime(config):
    torch.use_deterministic_algorithms(config.training.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.training.deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def schedule(step, config):
    t = config.training
    if step < t.warmup_steps:
        lr = t.learning_rate * (step+1) / max(1, t.warmup_steps)
    else:
        progress = (step-t.warmup_steps) / max(1, t.steps-t.warmup_steps-1)
        lr = t.learning_rate * (t.minimum_lr_ratio + (1-t.minimum_lr_ratio)*0.5*(1+math.cos(math.pi*progress)))
    progress = step / max(1, t.steps-1)
    ema = 1 - (1-t.ema_start)*0.5*(1+math.cos(math.pi*progress))
    return lr, ema


def identity(config, seed, k, method, train, val):
    payload = {"schema": 1, "config": config.to_dict(), "seed": seed, "k": k, "method": method,
               "train_fingerprint": train["fingerprint"], "val_fingerprint": val["fingerprint"],
               "source_hash": source_hash()}
    payload["run_id"] = digest(payload)
    return payload


def run_directory(config, meta):
    label = f'{config.geometry.mode}-{config.geometry.padding}-{meta["method"]}-k{meta["k"]}-s{meta["seed"]}'
    return Path(config.training.output_root) / f'{label}-{meta["run_id"][:12]}'


def save_checkpoint(path, model, optimizer, step, meta):
    checkpoint = {"schema": 1, "metadata": meta, "model": model.state_dict(),
                  "optimizer": optimizer.state_dict(), "step": step,
                  "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                  "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temp)
    os.replace(temp, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != 1:
        raise ValueError("Unsupported checkpoint format; legacy checkpoints are not accepted")
    meta = checkpoint["metadata"]
    if digest({k: v for k, v in meta.items() if k != "run_id"}) != meta["run_id"]:
        raise ValueError("Checkpoint identity was modified")
    return checkpoint


def load_encoder_run(path, device=None):
    checkpoint = load_checkpoint(path)
    meta = checkpoint["metadata"]
    if meta["source_hash"] != source_hash():
        raise ValueError("Checkpoint was created by a different source version")
    config = Config.from_dict(meta["config"])
    target = torch.device(device) if device else device_for(config)
    setup_runtime(config)
    model = SEPA(config, meta["seed"], meta["method"]).to(target)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config, checkpoint, target


def trim_history(path, completed_step):
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break  # interrupted final append; later records cannot exist
        if row["step"] <= completed_step:
            kept.append(json.dumps(row))
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def train(config: Config, *, seed=0, k=3, method="sepa", resume=False, stop_after=None):
    config.validate()
    if method not in {"sepa", "full"} or not isinstance(seed, int) or seed < 0:
        raise ValueError("Expected a nonnegative integer seed and method=sepa/full")
    if k not in VALID_K or (method == "full" and k != 0):
        raise ValueError("Invalid k; full-image control is always k=0")
    if stop_after is not None and not 0 < stop_after <= config.training.steps:
        raise ValueError("stop_after must be within the configured schedule")
    train_manifest, val_manifest = manifests(config)
    meta = identity(config, seed, k, method, train_manifest, val_manifest)
    directory = run_directory(config, meta)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".running"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Run is locked: {lock}. Check the recorded process before removing a stale lock.") from exc
    with os.fdopen(fd, "w") as handle:
        handle.write(str(os.getpid()))
    try:
        return _train(config, meta, directory, train_manifest, resume, stop_after)
    finally:
        lock.unlink(missing_ok=True)


def _train(config, meta, directory, train_manifest, resume, stop_after):
    t, seed, k, method = config.training, meta["seed"], meta["k"], meta["method"]
    checkpoint_path = directory / "latest.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"Run already exists; pass --resume: {directory}")
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint to resume: {directory}")
    device = device_for(config)
    setup_runtime(config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(seed_for(seed, "training"))
    random.seed(seed_for(seed, "python"))
    model = SEPA(config, seed, method).to(device).train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=t.learning_rate, weight_decay=t.weight_decay)
    step = 0
    if resume:
        state = load_checkpoint(checkpoint_path)
        if state["metadata"] != meta:
            raise ValueError("Resume config, data identity, or source differs from checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        write_json(directory / "run.json", meta)
        save_checkpoint(checkpoint_path, model, optimizer, step, meta)
    history = directory / "history.jsonl"
    trim_history(history, step)
    dataset = ImageTiles(train_manifest, config, seed=seed, k=k, training=True)
    reference = dataset[(0, 0)]
    reference_tiles = reference["canonical"][None].to(device)
    steps_per_epoch = math.ceil(len(dataset) / t.batch_size)
    end = min(stop_after or t.steps, t.steps)
    started = time.time()
    while step < end:
        epoch, skip = divmod(step, steps_per_epoch)
        loader = DataLoader(dataset, batch_sampler=epoch_batches(len(dataset), t.batch_size, seed, epoch, skip),
                            num_workers=t.num_workers, generator=generator(seed, "loader", epoch),
                            pin_memory=device.type == "cuda")
        for batch in loader:
            if step >= end:
                break
            lr, ema = schedule(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items() if torch.is_tensor(value)}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=t.precision == "bf16"):
                predicted, encoded, relations = model(tensors["visible"], tensors["observed_slots"], tensors["query_slots"])
                target = model.targets(tensors["canonical"], tensors["query_slots"])
                semantic = latent_loss(predicted, target)
                auxiliary = (relation_loss(relations, tensors["canonical_ids"], config.model.relation == "directed")
                             if relations is not None else semantic.new_zeros(()))
                total = semantic + config.model.relation_weight * auxiliary
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite loss at step {step}; latest checkpoint retained")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(parameters, t.clip_grad, error_if_nonfinite=True)
            optimizer.step()
            model.update_teacher(ema)
            step += 1
            metrics = diagnostics(predicted, target, encoded, canonical_ids=tensors["canonical_ids"],
                                  query_slots=tensors["query_slots"], sample_ids=batch["id"],
                                  collapse_ratio=t.collapse_ratio, minimum_std=t.collapse_min_std,
                                  minimum_image_std=t.collapse_min_image_std)
            row = {"step": step, "epoch": epoch, "loss": total.item(), "semantic": semantic.item(),
                   "relation": auxiliary.item(), "learning_rate": lr, "ema": ema, "gradient_norm": grad.item(),
                   "moved_mean": tensors["moved_count"].float().mean().item(),
                   "identity_fraction": (tensors["moved_count"] == 0).float().mean().item(),
                   "targets": int(target.shape[0] * 2), **metrics,
                   "batch_ids": batch["id"]}
            if step == 1 or step % t.log_every == 0 or step == end:
                row.update(position_sensitivity(model, reference_tiles), position_reference_id=reference["id"])
            with history.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            if step == 1 or step % t.log_every == 0 or step == end:
                print(f"step={step}/{t.steps} loss={row['loss']:.5f} std_ratio={row['std_ratio']:.4f} k={k}", flush=True)
                if row["collapse_flag"]:
                    print("collapse warning: " + ", ".join(row["collapse_reasons"]), flush=True)
            if step % t.checkpoint_every == 0 or step == end:
                save_checkpoint(checkpoint_path, model, optimizer, step, meta)
    result = {"run_dir": str(directory), "checkpoint": str(checkpoint_path), "step": step,
              "complete": step == t.steps, "elapsed_seconds_this_invocation": time.time()-started,
              "trainable_parameters": sum(p.numel() for p in parameters), "device": str(device),
              "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    write_json(directory / "status.json", result)
    return result
