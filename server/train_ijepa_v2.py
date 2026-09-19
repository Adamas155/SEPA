"""SEPA V2 screen on the byte-pinned I-JEPA implementation.

75% of batches are exactly the baseline global I-JEPA update. Every fourth
batch is a local spatial branch: seven visible native 80px tiles are encoded
independently without slot identity, while the unchanged official predictor
uses observed positions to predict 50 patch targets from two hidden canonical
tiles of a clean global EMA teacher. K0 and k3 are paired arms.

This module has no power-management action.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import fcntl
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

import ijepa_baseline as j
from diagnostic_lib import grouped_split
from src.models.vision_transformer import vit_small
from sepa_plan_b.geometry import sample_layout

torch = j.torch
F = j.F
ROOT = j.ROOT
BASELINE_ID = "3e242d05b38e789016cceb96b7c1d55983842c5d640b508965442a6576b8558f"
ARMS = ("k0", "k3")


def tile_patch_indices(device="cpu"):
    grid = torch.arange(225, device=device).reshape(15, 15)
    result = torch.stack([grid[r*5:(r+1)*5, c*5:(c+1)*5].flatten()
                          for r in range(3) for c in range(3)])
    assert torch.equal(result.flatten().sort().values, torch.arange(225, device=device))
    return result


def layouts(step, batch_size, k, device):
    """Return source pixels, observed predictor positions and target positions."""
    patches = tile_patch_indices()
    source, observed, target, moved, mappings = [], [], [], [], []
    for row in range(batch_size):
        layout = sample_layout(0, step // 1980, f"step-{step}-row-{row}", k)
        assert torch.equal(layout.visible_slots.sort().values, layout.visible_slots)
        assert torch.equal(layout.query_slots.sort().values, layout.query_slots)
        assert len(layout.visible_slots) == 7 and len(layout.query_slots) == 2
        source.append(patches[layout.canonical_ids].flatten())
        observed.append(patches[layout.visible_slots].flatten())
        target.append(patches[layout.query_slots].flatten())
        moved.append(layout.moved_count)
        mappings.append(layout.piece_to_slot)
    return (torch.stack(source).to(device), torch.stack(observed).to(device),
            torch.stack(target).to(device), torch.tensor(moved, device=device),
            torch.stack(mappings).to(device))


def local_encode(encoder, images, source_indices):
    """Encode seven source tiles independently; no canonical/observed slot PE."""
    b = len(images)
    patches = encoder.patch_embed(images)
    gathered = torch.gather(patches, 1, source_indices[..., None].expand(-1, -1, encoder.embed_dim))
    x = gathered.reshape(b, 7, 25, encoder.embed_dim)
    # Same 5x5 local coordinates for every tile. These contain no tile identity.
    local_pe = encoder.pos_embed[:, tile_patch_indices(images.device)[0]].reshape(1, 1, 25, encoder.embed_dim)
    x = (x + local_pe).reshape(b*7, 25, encoder.embed_dim)
    for block in encoder.blocks:
        x = block(x)
    return encoder.norm(x).reshape(b, 7*25, encoder.embed_dim)


def context(arm=None):
    baseline_cfg, base, train, val, parent = j.context()
    assert parent["run_id"] == BASELINE_ID
    v2 = json.loads((ROOT / "server/ijepa_v2_config.json").read_text())
    assert tuple(v2["arms"]) == ARMS and v2["screen_epochs"] == 25
    shared = {"schema": 1, "experiment": "sepa_v2_25epoch_screen",
              "baseline": parent, "baseline_config": baseline_cfg, "v2_config": v2,
              "script_sha": j.file_sha(__file__),
              "test_sha": j.file_sha(ROOT / "server/test_ijepa_v2.py"),
              "config_sha": j.file_sha(ROOT / "server/ijepa_v2_config.json"),
              "data": {"train": train["fingerprint"], "val": val["fingerprint"]},
              "global_branch": "unchanged official I-JEPA forward, masks, target, loss and update",
              "local_branch": {"student": "seven independent native 80px tiles; 25 patch tokens each; repeated local 5x5 PE; no slot identity",
                  "teacher": "clean global 240px EMA teacher",
                  "predictor": "unchanged official 6-layer 384-wide predictor; observed context PE and canonical target PE",
                  "target": "all 25 patch embeddings in each of two hidden noncenter canonical tiles",
                  "loss": "teacher output LayerNorm plus mean Smooth L1, identical to global branch"},
              "screen_evaluation": {"data": "deterministic grouped 90/10 split of training manifest; conflicting duplicate groups excluded",
                  "encoder": "global student", "probe": asdict(base.probe), "imageNet100_val_used": False},
              "shutdown_on_success": False}
    shared["screen_id"] = j.digest(shared)
    if arm is None:
        return baseline_cfg, v2, base, train, val, shared
    if arm not in ARMS:
        raise ValueError(arm)
    meta = {"shared": shared, "arm": arm, "k": int(arm[1:])}
    meta["run_id"] = j.digest(meta)
    directory = ROOT / "runs-ijepa-v2" / f"{arm}-s0-{meta['run_id'][:12]}"
    return baseline_cfg, v2, base, train, val, meta, directory


def build(cfg, ipe):
    return j.build(cfg, ipe)


def update(models, batch, cfg, v2, total, k):
    enc, pred, teacher, opt, lr, wd = models
    x, _, masks_enc, masks_pred, step, retries = batch
    x = x.cuda(non_blocking=True)
    new_lr, new_wd = lr.step(), wd.step()
    opt.zero_grad(set_to_none=True)
    is_local = step % v2["branch_period"] == v2["local_branch_residue"]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["precision"] == "bf16"):
        with torch.no_grad():
            teacher_tokens = F.layer_norm(teacher(x), (enc.embed_dim,))
        if is_local:
            source, observed, target_indices, moved, mapping = layouts(step, len(x), k, x.device)
            encoded = local_encode(enc, x, source)
            predictions = pred(encoded, [observed], [target_indices])
            targets = j.apply_masks(teacher_tokens, [target_indices])
            assert encoded.shape == (len(x), 175, enc.embed_dim)
            assert predictions.shape == targets.shape == (len(x), 50, enc.embed_dim)
            branch = "local"
            context_patches, target_patches, mask_resamples = 175, 50, 0
            moved_mean = moved.float().mean().item()
            identity_fraction = (moved == 0).float().mean().item()
        else:
            me, mp = [[m.cuda(non_blocking=True) for m in group] for group in (masks_enc, masks_pred)]
            targets = j.apply_masks(teacher_tokens, mp)
            encoded = enc(x, me)
            predictions = pred(encoded, me, mp)
            assert predictions.shape == targets.shape
            branch = "global"
            context_patches, target_patches, mask_resamples = me[0].shape[1], mp[0].shape[1], retries
            moved_mean = identity_fraction = 0.0
        loss = F.smooth_l1_loss(predictions, targets)
    j.require_finite("SEPA V2", loss=loss, prediction=predictions, target=targets, encoded=encoded)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), float("inf"))
    j.require_finite("SEPA V2 gradient", gradient_norm=norm)
    opt.step()
    momentum = cfg["ema_start"] + step * (1-cfg["ema_start"]) / total
    with torch.no_grad():
        for online, target_parameter in zip(enc.parameters(), teacher.parameters()):
            target_parameter.mul_(momentum).add_(online, alpha=1-momentum)
        flat = targets.float().reshape(-1, len(x), targets.shape[1], enc.embed_dim)
        image_std = flat.std(dim=1, correction=0).mean().item()
        mean_target_loss = F.smooth_l1_loss(flat.mean(1, keepdim=True).expand_as(flat), flat).item()
    return {"step": step+1, "branch": branch, "loss": loss.item(), "lr": new_lr,
            "weight_decay": new_wd, "ema": momentum, "grad_norm": norm.item(),
            "batch_size": len(x), "context_patches": context_patches,
            "target_patches": target_patches, "mask_resamples": mask_resamples,
            "moved_mean": moved_mean, "identity_fraction": identity_fraction,
            "target_image_std": image_std, "batch_mean_target_loss": mean_target_loss}


def checkpoint_valid(path, models, meta):
    step, seconds = j.restore(path, models, meta)
    return step, seconds


def train_arm(arm, resume):
    cfg, v2, base, train, _, meta, directory = context(arm)
    receipt = json.loads((ROOT / "server-logs/ijepa_v2_smoke.json").read_text())
    assert receipt["status"] == "passed" and receipt["shared"] == meta["shared"]
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "worker.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ipe = math.ceil(len(train["records"]) / cfg["batch_size"])
    full_total = ipe * cfg["epochs"]
    stop = ipe * v2["screen_epochs"]
    status = ROOT / "server-logs" / f"ijepa_v2_{arm}_status.json"
    def progress(**kw):
        row = {"pid": os.getpid(), "utc": j.utc(), "arm": arm, "k": meta["k"],
               "run_id": meta["run_id"], "run_dir": str(directory), "screen_step": stop,
               "full_step": full_total, "shutdown_on_success": False, **kw}
        j.write_json(status, row)
        print(json.dumps(row), flush=True)
    started = time.monotonic()
    try:
        j.write_json(directory / "protocol.json", meta)
        j.write_json(directory / "runtime.json", {"torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "shutdown_on_success": False, "utc": j.utc()})
        models = build(cfg, ipe)
        latest = directory / "latest.pt"
        start, prior = 0, 0.0
        if latest.exists():
            if not resume:
                raise FileExistsError("Existing checkpoint requires --resume")
            start, prior = checkpoint_valid(latest, models, meta)
            j.trim_history(directory / "history.jsonl", start)
        elif resume:
            raise FileNotFoundError(latest)
        assert 0 <= start <= stop
        if start == stop:
            progress(phase="screen_training_complete", step=start, epoch=start/ipe)
            return
        progress(phase="training", step=start, epoch=start/ipe)
        last_time, count, losses = time.monotonic(), 0, {"global": [], "local": []}
        with (directory / "history.jsonl").open("a", buffering=1) as history:
            for batch in j.loader(train, cfg, start=start, stop=stop):
                row = update(models, batch, cfg, v2, full_total, meta["k"])
                step = row["step"]
                losses[row["branch"]].append(row["loss"])
                count += 1
                if step % 25 == 0 or step == start+1 or step % ipe == 0:
                    now = time.monotonic()
                    row.update(epoch=step/ipe, recent_seconds_per_step=(now-last_time)/count,
                        global_mean_loss=(sum(losses["global"])/len(losses["global"]) if losses["global"] else None),
                        local_mean_loss=(sum(losses["local"])/len(losses["local"]) if losses["local"] else None),
                        training_seconds=prior+now-started,
                        max_gpu_bytes=torch.cuda.max_memory_allocated(), utc=j.utc())
                    history.write(json.dumps(row)+"\n")
                    progress(phase="training", **row)
                    last_time, count, losses = now, 0, {"global": [], "local": []}
                if step % ipe == 0:
                    j.save(latest, models, step, meta, prior+time.monotonic()-started)
                    if step//ipe in (10, 25):
                        milestone = directory / "milestones" / f"epoch-{step//ipe:03d}.pt"
                        milestone.parent.mkdir(exist_ok=True)
                        shutil.copy2(latest, milestone)
                        j.write_json(milestone.with_suffix(".json"), {"step": step,
                            "sha256": j.file_sha(milestone), "run_id": meta["run_id"], "utc": j.utc()})
        assert models[4]._step == models[5]._step == stop
        progress(phase="screen_training_complete", step=stop, epoch=v2["screen_epochs"],
                 checkpoint_sha=j.file_sha(latest))
    except BaseException as exc:
        progress(phase="needs_review", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        lock.close()


def load_student(path, state_key="encoder"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    encoder = vit_small(img_size=[240], patch_size=16)
    encoder.load_state_dict(ck[state_key], strict=True)
    return encoder.requires_grad_(False).eval().cuda(), ck


def screen_evaluate():
    cfg, v2, base, train, _, shared = context()
    output = ROOT / "diagnostics" / f"ijepa-v2-screen-{shared['screen_id'][:12]}"
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "evaluation.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = ROOT / "server-logs/ijepa_v2_status.json"
    def progress(**kw):
        row = {"pid": os.getpid(), "utc": j.utc(), "screen_id": shared["screen_id"],
               "output": str(output), "shutdown_on_success": False, **kw}
        j.write_json(status, row)
        print(json.dumps(row), flush=True)
    try:
        candidates = {}
        baseline = list((ROOT / "runs-ijepa").glob("official-small240-s0-*/milestones/epoch-025.pt"))
        if len(baseline) != 1:
            raise RuntimeError(f"Expected one baseline epoch-25 checkpoint, found {len(baseline)}")
        candidates["ijepa-baseline-e25"] = baseline[0]
        for arm in ARMS:
            *_, meta, directory = context(arm)
            path = directory / "milestones/epoch-025.pt"
            receipt = json.loads(path.with_suffix(".json").read_text())
            assert receipt["run_id"] == meta["run_id"] and receipt["sha256"] == j.file_sha(path)
            candidates[f"v2-{arm}-e25"] = path
        audit = j.record_audit(train["records"])
        train_rows, dev_rows = grouped_split(train["records"], seed=0,
            fraction=v2["internal_dev_fraction"], excluded_ids=audit["conflicting_groups"])
        health_rows = j.health_indices(train["records"], excluded_ids=audit["conflicting_groups"])
        results = {}
        for name, path in candidates.items():
            progress(phase="loading", candidate=name)
            encoder, ck = load_student(path)
            checkpoint_sha = j.file_sha(path)
            identity = {"shared": shared, "candidate": name, "checkpoint_sha": checkpoint_sha,
                        "checkpoint_step": ck["step"], "branch": "global_student",
                        "split": "grouped_train_internal_dev"}
            payload = j.cached_extract(output / f"{name}-train-features.pt", j.ProbeAdapter(encoder),
                "student", train, base, "cuda", identity,
                lambda **kw: progress(phase="feature_extraction", candidate=name, **kw))
            x, y = payload["features"], payload["labels"]
            marks = (base.probe.epochs,)
            head, mean, std, trace = j.fit_trajectory(x[train_rows], y[train_rows], 100, base, 0, "cuda",
                dev=(x[dev_rows], y[dev_rows]), marks=marks,
                progress=lambda **kw: progress(phase="linear_probe", candidate=name, **kw))
            dev_scores = trace[-1]["dev_scores"]
            dev_scores["knn"] = j.knn_accuracy(x[train_rows], y[train_rows], x[dev_rows], y[dev_rows],
                                               100, k=20, temperature=0.07, device="cuda")
            result = {"identity": identity, "scores": dev_scores,
                "feature_health": j.feature_health(x[health_rows], y[health_rows]),
                "probe_final": trace[-1], "n_probe_train": len(train_rows), "n_dev": len(dev_rows),
                "n_conflicting_records_excluded": audit["conflicting_records"]}
            stem = output / f"{name}-result.json"
            j.save_tensor(stem.with_suffix(".pt"), {"identity": identity, "head": head.cpu().state_dict(),
                                                    "mean": mean, "std": std})
            j.write_json(stem, result)
            j.write_json(output / f"{name}-receipt.json", {"identity": identity,
                "result_sha": j.file_sha(stem), "head_sha": j.file_sha(stem.with_suffix(".pt"))})
            results[name] = result
            del encoder, payload, x, y, head
            torch.cuda.empty_cache()
        summary = {"status": "complete", "shared": shared,
                   "results": {k: {"scores": v["scores"], "feature_health": v["feature_health"]}
                               for k, v in results.items()},
                   "imageNet100_val_used": False, "utc": j.utc(), "shutdown_on_success": False}
        j.write_json(output / "summary.json", summary)
        progress(phase="complete", results={k: v["scores"] for k, v in results.items()})
    except BaseException as exc:
        progress(phase="needs_review", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        lock.close()


def coordinate(resume):
    cfg, v2, _, _, _, shared = context()
    receipt = json.loads((ROOT / "server-logs/ijepa_v2_smoke.json").read_text())
    assert receipt["status"] == "passed" and receipt["shared"] == shared
    lock = (ROOT / "server-logs/ijepa_v2_driver.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = ROOT / "server-logs/ijepa_v2_status.json"
    try:
        for arm in ARMS:
            args = [sys.executable, "-u", __file__, "--worker", "--arm", arm]
            *_, directory = context(arm)
            if (directory / "latest.pt").exists() or resume:
                args.append("--resume")
            j.write_json(status, {"pid": os.getpid(), "utc": j.utc(), "phase": "training_arm",
                "arm": arm, "screen_id": shared["screen_id"], "shutdown_on_success": False})
            subprocess.run(args, check=True)
        screen_evaluate()
    except BaseException as exc:
        j.write_json(status, {"pid": os.getpid(), "utc": j.utc(), "phase": "needs_review",
            "screen_id": shared["screen_id"], "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(), "shutdown_on_success": False})
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    cfg, _, base, *_ = context()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(cfg["cpu_threads"])
    torch.set_num_interop_threads(1)
    j.setup_runtime(base)
    if args.smoke:
        from test_ijepa_v2 import smoke
        smoke()
    elif args.worker:
        if args.arm is None:
            raise ValueError("--worker requires --arm")
        train_arm(args.arm, args.resume)
    elif args.evaluate:
        screen_evaluate()
    else:
        coordinate(args.resume)


if __name__ == "__main__":
    main()
