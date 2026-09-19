"""Auditable single-GPU adapter around a byte-pinned official I-JEPA snapshot.

The upstream encoder, predictor, initialization, masks, transforms and optimizer
are imported unchanged. Only manifest loading, deterministic batch seeds,
checkpointing and the shared SEPA evaluation protocol are implemented here.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/ijepa-52c1ae9"
sys.path[:0] = [str(VENDOR), str(ROOT / "src"), str(ROOT / "server")]

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.helper import init_model, init_opt
from src.masks.multiblock import MaskCollator
from src.masks.utils import apply_masks
from src.transforms import make_transforms
from sepa_plan_b.config import Config, digest
from sepa_plan_b.data import epoch_batches, manifests, write_json
from sepa_plan_b.engine import load_encoder_run, setup_runtime, source_hash, trim_history
from sepa_plan_b.evaluation import knn_accuracy, linear_metrics
from sepa_plan_b.numerics import require_finite
from sepa_plan_b.rng import generator, seed_for
from diagnostic_lib import (cached_extract, feature_health, file_sha, fit_trajectory,
                            health_indices, record_audit, save_tensor)

CORE_SHA = "a659af0796728674296cf87415ad5ebf02b8d946836d142bae894e43d0bec0c4"
TRAIN_SHA = "9a7ff7775290a65a26289b713781f92a9373db05261a96d77a58c3aebc4b13fd"
VAL_SHA = "2fb02ba5781f39d1b6503c4f10b95a5614d5d3ce446c9cf503d76754a0741df4"


def utc():
    return datetime.now(timezone.utc).isoformat()


def context():
    cfg = json.loads((ROOT / "server/ijepa_baseline_config.json").read_text())
    base = Config.load(ROOT / "configs/native80_100ep.toml")
    train, val = manifests(base)
    assert source_hash() == CORE_SHA, "SEPA core changed"
    assert (train["fingerprint"], val["fingerprint"]) == (TRAIN_SHA, VAL_SHA)
    assert (len(train["records"]), len(val["records"])) == (126684, 5000)
    pin = json.loads((VENDOR / "UPSTREAM.json").read_text())
    actual = {str(p.relative_to(VENDOR)): file_sha(p) for p in VENDOR.rglob("*.py")}
    assert pin["commit"] == cfg["upstream_commit"] and actual == pin["python_files"]
    meta = {"config": cfg, "train_fingerprint": TRAIN_SHA, "val_fingerprint": VAL_SHA,
            "core_sha": CORE_SHA, "upstream": pin,
            "adapter_sha": file_sha(__file__), "diagnostic_sha": file_sha(ROOT / "server/diagnostic_lib.py")}
    meta["run_id"] = digest(meta)
    return cfg, base, train, val, meta


class Images(Dataset):
    def __init__(self, manifest, cfg):
        self.manifest, self.cfg = manifest, cfg
        self.root = Path(manifest["root"]).resolve()
        self.transform = make_transforms(crop_size=cfg["image_size"], crop_scale=(0.3, 1.0),
                                        horizontal_flip=False, color_distortion=False, gaussian_blur=False)

    def __len__(self):
        return len(self.manifest["records"])

    def __getitem__(self, item):
        index, epoch, step = item
        rec = self.manifest["records"][index]
        path = (self.root / rec["path"]).resolve()
        assert path.is_relative_to(self.root)
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == rec["sha256"], str(path)
        with Image.open(io.BytesIO(raw)) as im, torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_for(self.cfg["seed"], "ijepa_augmentation", epoch, rec["id"]))
            x = self.transform(im.convert("RGB"))
        return x, rec["label"], step


class Batches:
    def __init__(self, size, cfg, start=0, stop=None):
        self.size, self.cfg, self.start = size, cfg, start
        self.ipe = math.ceil(size / cfg["batch_size"])
        self.stop = self.ipe * cfg["epochs"] if stop is None else stop

    def __len__(self):
        return self.stop - self.start

    def __iter__(self):
        for epoch in range(self.start // self.ipe, self.cfg["epochs"]):
            skip = max(0, self.start - epoch * self.ipe)
            for j, batch in enumerate(epoch_batches(self.size, self.cfg["batch_size"],
                                                    self.cfg["seed"], epoch, skip), skip):
                step = epoch * self.ipe + j
                if step >= self.stop:
                    return
                yield [(i, e, step) for i, e in batch]


class SeededMasks(MaskCollator):
    def __init__(self, cfg):
        super().__init__(input_size=cfg["image_size"], patch_size=cfg["patch_size"], **cfg["mask"])
        self.seed = cfg["seed"]

    def step(self):
        # Do not use upstream's worker-shared, scheduling-dependent counter.
        return self.batch_seed

    def __call__(self, batch):
        steps = {item[2] for item in batch}
        assert len(steps) == 1
        step = steps.pop()
        for attempt in range(100):
            self.batch_seed = seed_for(self.seed, "ijepa_masks", step, attempt)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.batch_seed)
                (images, labels, _), enc, pred = super().__call__(batch)
            # Upstream can relax exclusions after failed sampling attempts.
            if all(not (e[:, :, None] == p[:, None, :]).any().item() for e in enc for p in pred):
                return images, labels, enc, pred, step, attempt
        raise RuntimeError("Cannot sample nonoverlapping context and targets")


def loader(train, cfg, start=0, stop=None):
    return DataLoader(Images(train, cfg), batch_sampler=Batches(len(train["records"]), cfg, start, stop),
                      collate_fn=SeededMasks(cfg), num_workers=cfg["workers"], pin_memory=True,
                      persistent_workers=cfg["workers"] > 0, generator=generator(cfg["seed"], "ijepa_loader"))


def build(cfg, ipe):
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    enc, pred = init_model("cuda", patch_size=cfg["patch_size"], model_name=cfg["model"],
                           crop_size=cfg["image_size"], pred_depth=cfg["predictor_depth"],
                           pred_emb_dim=cfg["predictor_dim"])
    teacher = copy.deepcopy(enc).requires_grad_(False).eval()
    opt, _, lr, wd = init_opt(enc, pred, ipe, cfg["start_lr"], cfg["learning_rate"],
                              cfg["warmup_epochs"], cfg["epochs"], cfg["weight_decay"],
                              cfg["final_weight_decay"], cfg["final_lr"], False, 1.0)
    return enc, pred, teacher, opt, lr, wd


def update(models, batch, cfg, total):
    enc, pred, teacher, opt, lr, wd = models
    x, _, masks_enc, masks_pred, step, retries = batch
    x = x.cuda(non_blocking=True)
    me, mp = [[m.cuda(non_blocking=True) for m in group] for group in (masks_enc, masks_pred)]
    new_lr, new_wd = lr.step(), wd.step()
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["precision"] == "bf16"):
        with torch.no_grad():
            targets = F.layer_norm(teacher(x), (enc.embed_dim,))
            targets = apply_masks(targets, mp)
        predictions = pred(enc(x, me), me, mp)
        assert predictions.shape == targets.shape
        loss = F.smooth_l1_loss(predictions, targets)
    require_finite("I-JEPA prediction", loss=loss, prediction=predictions, target=targets)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), float("inf"))
    require_finite("I-JEPA gradient", gradient_norm=norm)
    opt.step()
    momentum = cfg["ema_start"] + step * (1 - cfg["ema_start"]) / total
    with torch.no_grad():
        for online, target in zip(enc.parameters(), teacher.parameters()):
            target.mul_(momentum).add_(online, alpha=1-momentum)
        t = targets.float().reshape(len(mp), len(x), -1, enc.embed_dim)
        image_std = t.std(dim=1, correction=0).mean().item()
        ratio_loss = F.smooth_l1_loss(t.mean(dim=1, keepdim=True).expand_as(t), t).item()
    return {"step": step+1, "loss": loss.item(), "lr": new_lr, "weight_decay": new_wd,
            "ema": momentum, "grad_norm": norm.item(), "batch_size": len(x),
            "context_patches": me[0].shape[1], "target_patches_per_block": mp[0].shape[1],
            "mask_resamples": retries, "target_image_std": image_std,
            "batch_mean_target_loss": ratio_loss}


def save(path, models, step, meta, seconds):
    enc, pred, teacher, opt, lr, wd = models
    save_tensor(path, {"metadata": meta, "step": step, "encoder": enc.state_dict(),
                       "predictor": pred.state_dict(), "teacher": teacher.state_dict(),
                       "optimizer": opt.state_dict(), "lr_step": lr._step, "wd_step": wd._step,
                       "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                       "training_seconds": seconds})


def restore(path, models, meta):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ck["metadata"] == meta, "Checkpoint protocol/source mismatch"
    enc, pred, teacher, opt, lr, wd = models
    for model, key in [(enc, "encoder"), (pred, "predictor"), (teacher, "teacher")]:
        model.load_state_dict(ck[key], strict=True)
    opt.load_state_dict(ck["optimizer"])
    lr._step, wd._step = ck["lr_step"], ck["wd_step"]
    torch.set_rng_state(ck["torch_rng"])
    torch.cuda.set_rng_state_all(ck["cuda_rng"])
    assert lr._step == wd._step == ck["step"]
    return ck["step"], ck["training_seconds"]


def join_tiles(tiles):
    b, _, c, h, w = tiles.shape
    return tiles.reshape(b, 3, 3, c, h, w).permute(0, 3, 1, 4, 2, 5).reshape(b, c, 3*h, 3*w)


class ProbeAdapter(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, tiles):
        return self.encoder(join_tiles(tiles))


def evaluate(directory, models, base, train, val, meta, progress):
    # Fixed final checkpoint; validation never selects epochs or hyperparameters.
    setup_runtime(base)
    final_sha = file_sha(directory / "latest.pt")
    audit = record_audit(train["records"])
    idx = health_indices(train["records"], excluded_ids=audit["conflicting_groups"])
    results = {}
    for branch, enc in [("student", models[0]), ("teacher", models[2])]:
        identity = {"run_id": meta["run_id"], "checkpoint_sha": final_sha, "branch": branch,
                    "probe_config": base.probe.__dict__, "evaluation": "shared_center_square_fp32"}
        model = ProbeAdapter(enc).eval()
        features = []
        for split, manifest in [("train", train), ("val", val)]:
            def report(**kw):
                progress(phase="feature_extraction", branch=branch, split=split, **kw)
            payload = cached_extract(directory / "evaluation" / f"{branch}-{split}.pt", model,
                                     "student", manifest, base, "cuda", identity, report)
            features.append(payload)
        tr, va = features
        def report_fit(**kw):
            if kw.get("epoch", 0) % 10 == 0:
                progress(phase="linear_probe", branch=branch, **kw)
        head, mean, std, trace = fit_trajectory(tr["features"], tr["labels"], 100, base,
                                                meta["config"]["seed"], "cuda", progress=report_fit)
        scores, predictions = linear_metrics(head, mean, std, va["features"], va["labels"], "cuda")
        scores["knn"] = knn_accuracy(tr["features"], tr["labels"], va["features"], va["labels"],
                                    100, k=20, temperature=0.07, device="cuda")
        health = feature_health(tr["features"][idx], tr["labels"][idx])
        result = {"identity": identity, "scores": scores, "feature_health": health,
                  "n_train_labels": len(tr["labels"]), "n_val": len(va["labels"]), "trace": trace}
        result_path = directory / "evaluation" / f"{branch}-result.json"
        save_tensor(result_path.with_suffix(".pt"), {"identity": identity, "head": head.cpu().state_dict(),
                    "mean": mean, "std": std, "predictions": predictions})
        write_json(result_path, result)
        results[branch] = result
        progress(phase="branch_complete", branch=branch, scores=scores)
    # Complete the already-requested final EMA check for the two key SEPA arms.
    for arm in ("k0", "k3"):
        paths = list((ROOT / "runs-long100").glob(f"native80-reflect-sepa-{arm}-s0-*/milestones/epoch-100-step-198000.pt"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one final SEPA {arm} checkpoint")
        checkpoint = paths[0]
        model, config, ck, device = load_encoder_run(checkpoint, "cuda")
        identity = {"checkpoint_sha": file_sha(checkpoint), "branch": "teacher", "run_id": ck["metadata"]["run_id"],
                    "probe_config": config.probe.__dict__}
        fs = [cached_extract(directory / "evaluation" / f"sepa-{arm}-teacher-{split}.pt", model, "teacher",
                             manifest, config, device, identity,
                             lambda **kw: progress(phase="sepa_teacher_extraction", arm=arm, **kw))
              for split, manifest in [("train", train), ("val", val)]]
        tr, va = fs
        head, mean, std, trace = fit_trajectory(tr["features"], tr["labels"], 100, config, 0, device,
            progress=lambda **kw: progress(phase="sepa_teacher_probe", arm=arm, **kw) if kw.get("epoch", 0) % 10 == 0 else None)
        scores, predictions = linear_metrics(head, mean, std, va["features"], va["labels"], device)
        scores["knn"] = knn_accuracy(tr["features"], tr["labels"], va["features"], va["labels"], 100, device=device)
        result = {"identity": identity, "scores": scores, "trace": trace,
                  "feature_health": feature_health(tr["features"][idx], tr["labels"][idx])}
        path = directory / "evaluation" / f"sepa-{arm}-teacher-result.json"
        save_tensor(path.with_suffix(".pt"), {"identity": identity, "head": head.cpu().state_dict(),
                                             "mean": mean, "std": std, "predictions": predictions})
        write_json(path, result)
        results[f"sepa_{arm}_teacher"] = result
        del model, fs, tr, va, head
    summary = {"status": "complete", "completed_utc": utc(), "metadata": meta,
               "checkpoint_sha": final_sha, "results": {k: v["scores"] for k, v in results.items()}}
    write_json(directory / "summary.json", summary)
    return summary


def smoke(cfg, base, train, meta):
    """Real-image gradients, mask isolation, resumed updates and timing."""
    out = ROOT / "server-logs/ijepa-smoke"
    out.mkdir(parents=True, exist_ok=True)
    ipe = math.ceil(len(train["records"]) / cfg["batch_size"])
    models = build(cfg, ipe)
    batches = list(loader(train, cfg, stop=3))
    enc, pred, teacher = models[:3]
    x, _, me, mp, _, _ = batches[0]
    with torch.no_grad():
        # Changing all hidden pixels must have no effect on the context encoder.
        a = x[:2].cuda()
        visible = torch.zeros((2, 225), dtype=torch.bool, device="cuda")
        visible.scatter_(1, me[0][:2].cuda(), True)
        pixel_mask = visible.reshape(2, 15, 15).repeat_interleave(16, 1).repeat_interleave(16, 2)
        b = torch.where(pixel_mask[:, None], a, torch.randn_like(a))
        assert torch.equal(enc(a, [me[0][:2].cuda()]), enc(b, [me[0][:2].cuda()]))
        target_tokens = torch.arange(2*225*3).reshape(2, 225, 3)
        got = apply_masks(target_tokens, [m[:2] for m in mp])
        want = torch.cat([target_tokens[torch.arange(2)[:, None], m[:2]] for m in mp])
        assert torch.equal(got, want), "Target block/sample ordering"
        # Exactly reconstruct the image represented by the canonical tiles.
        from sepa_plan_b.geometry import split_tiles
        assert torch.equal(join_tiles(split_tiles(x[0], base.geometry, normalize=False)[None])[0], x[0])
    # Per-image and mask reproducibility do not depend on worker scheduling.
    serial_cfg = {**cfg, "workers": 0}
    again = next(iter(loader(train, serial_cfg, stop=1)))
    assert torch.equal(again[0], batches[0][0])
    for group in (2, 3):
        assert all(torch.equal(a, b) for a, b in zip(again[group], batches[0][group]))
    replay = next(iter(loader(train, serial_cfg, start=1, stop=2)))
    assert torch.equal(replay[0], batches[1][0])
    initial = enc.patch_embed.proj.weight.detach().clone()
    rows, times = [], []
    for batch in batches[:2]:
        torch.cuda.synchronize(); started = time.monotonic()
        rows.append(update(models, batch, cfg, ipe*cfg["epochs"]))
        torch.cuda.synchronize(); times.append(time.monotonic()-started)
        if batch[4] == 0:
            save(out / "resume.pt", models, 1, meta, 0)
    assert not torch.equal(initial, enc.patch_embed.proj.weight)
    assert all(p.grad is None for p in teacher.parameters())
    expected = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in models[:3]]
    restore(out / "resume.pt", models, meta)
    replay_row = update(models, replay, cfg, ipe*cfg["epochs"])
    assert replay_row == rows[-1], "Resume metrics differ"
    assert all(torch.equal(v.cpu(), state[k]) for m, state in zip(models[:3], expected)
               for k, v in m.state_dict().items()), "Resumed weights differ"
    rows.append(update(models, batches[2], cfg, ipe*cfg["epochs"]))
    # Exercise the final short batch and the final schedule without training it.
    last = next(iter(Batches(len(train["records"]), cfg, start=ipe-1, stop=ipe)))
    assert len(last) == 28 and last[-1][1:] == (0, ipe-1)
    for step in (1, ipe*cfg["epochs"]):
        models[4]._step = models[5]._step = step-1
        lr, wd = models[4].step(), models[5].step()
        assert math.isfinite(lr) and math.isfinite(wd)
    assert abs(lr-cfg["final_lr"]) < 1e-12 and abs(wd-cfg["final_weight_decay"]) < 1e-12
    receipt = {"status": "passed", "utc": utc(), "metadata": meta,
               "checks": ["hidden_pixel_isolation", "target_order", "tile_reassembly", "worker_independence",
                          "resumed_data", "encoder_updates", "teacher_no_grad", "bitwise_resume",
                          "last_batch_28", "schedule_endpoints", "finite_bf16_gradients"],
               "rows": rows, "step_seconds": times,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
               "trainable_encoder_parameters": sum(p.numel() for p in enc.parameters() if p.requires_grad),
               "trainable_predictor_parameters": sum(p.numel() for p in pred.parameters() if p.requires_grad)}
    write_json(ROOT / "server-logs/ijepa_smoke_receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def run(cfg, base, train, val, meta, args):
    directory = ROOT / "runs-ijepa" / f"official-small240-s0-{meta['run_id'][:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    smoke_receipt = json.loads((ROOT / "server-logs/ijepa_smoke_receipt.json").read_text())
    assert smoke_receipt["status"] == "passed" and smoke_receipt["metadata"] == meta
    status_path = ROOT / "server-logs/ijepa_status.json"
    def progress(**kw):
        row = {"pid": os.getpid(), "updated_utc": utc(), "run_dir": str(directory),
               "run_id": meta["run_id"], "shutdown_on_success": args.shutdown_on_success, **kw}
        write_json(status_path, row)
        print(json.dumps(row), flush=True)
    lock = (ROOT / "server-logs/driver.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ipe = math.ceil(len(train["records"]) / cfg["batch_size"])
    total = ipe*cfg["epochs"]
    started = time.monotonic()
    try:
        progress(phase="initializing", target_step=total)
        write_json(directory / "protocol.json", meta)
        models = build(cfg, ipe)
        checkpoint = directory / "latest.pt"
        start, prior_seconds = 0, 0
        if checkpoint.exists():
            if not args.resume:
                raise FileExistsError("Existing checkpoint requires --resume")
            start, prior_seconds = restore(checkpoint, models, meta)
            trim_history(directory / "history.jsonl", start)
        elif args.resume:
            raise FileNotFoundError(checkpoint)
        last_log = time.monotonic()
        seen = 0
        losses = []
        with (directory / "history.jsonl").open("a", buffering=1) as history:
            for batch in loader(train, cfg, start):
                row = update(models, batch, cfg, total)
                step = row["step"]
                losses.append(row["loss"]); seen += 1
                if step % 25 == 0 or step == start+1 or step % ipe == 0:
                    now = time.monotonic()
                    row.update(epoch=step/ipe, mean_loss=float(np.mean(losses)),
                               recent_seconds_per_step=(now-last_log)/seen,
                               training_seconds=prior_seconds+now-started,
                               max_gpu_bytes=torch.cuda.max_memory_allocated(), utc=utc())
                    history.write(json.dumps(row)+"\n")
                    progress(phase="training", target_step=total, **row)
                    last_log, seen, losses = now, 0, []
                if step % ipe == 0:
                    save(checkpoint, models, step, meta, prior_seconds+time.monotonic()-started)
                    if step//ipe in cfg["milestones"]:
                        dest = directory / "milestones" / f"epoch-{step//ipe:03d}.pt"
                        dest.parent.mkdir(exist_ok=True)
                        if dest.exists():
                            raise FileExistsError(dest)
                        shutil.copy2(checkpoint, dest)
                        write_json(dest.with_suffix(".json"), {"step": step, "sha256": file_sha(dest),
                                                               "run_id": meta["run_id"], "utc": utc()})
        assert models[4]._step == total
        progress(phase="evaluating", step=total, target_step=total)
        summary = evaluate(directory, models, base, train, val, meta, progress)
        summary["training_seconds"] = torch.load(checkpoint, map_location="cpu", weights_only=True)["training_seconds"]
        write_json(directory / "summary.json", summary)
        progress(phase="complete", step=total, target_step=total, results=summary["results"])
        if args.shutdown_on_success:
            write_json(directory / "shutdown.json", {"status": "requested", "utc": utc(),
                                                       "summary_sha": file_sha(directory / "summary.json")})
            os.sync()
            subprocess.run(["/bin/bash", "/usr/bin/shutdown"], check=True)
    except BaseException as exc:
        progress(phase="needs_review", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shutdown-on-success", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    cfg, base, train, val, meta = context()
    assert torch.cuda.is_available()
    assert cfg["precision"] != "bf16" or torch.cuda.is_bf16_supported()
    torch.set_num_threads(cfg["cpu_threads"])
    torch.set_num_interop_threads(1)
    setup_runtime(base)
    if args.smoke:
        smoke(cfg, base, train, meta)
    else:
        run(cfg, base, train, val, meta, args)


if __name__ == "__main__":
    main()
