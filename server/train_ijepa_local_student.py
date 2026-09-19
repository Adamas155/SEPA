"""I-JEPA ablation: tile-local student, global EMA teacher and predictor.

The baseline adapter owns data, initialization, optimization, targets and loss.
Only student attention edges change. No power-management actions are provided.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import logging
import math
import os
import shutil
import time
import traceback

import ijepa_baseline as j
from evaluate_ijepa_fov import LocalTiles
from src.models.vision_transformer import vit_small

torch = j.torch
ROOT = j.ROOT
BASELINE_ID = "3e242d05b38e789016cceb96b7c1d55983842c5d640b508965442a6576b8558f"


class LocalStudent(torch.nn.Module):
    """Preserve upstream state keys and parameter order for optimizer and EMA."""
    def __init__(self, encoder):
        super().__init__()
        self.pos_embed = encoder.pos_embed
        self.patch_embed = encoder.patch_embed
        self.blocks = encoder.blocks
        self.norm = encoder.norm
        self.embed_dim = encoder.embed_dim
        assert self.pos_embed.shape == (1, 225, 384)
        assert list(self.state_dict()) == list(encoder.state_dict())
        assert [id(p) for p in self.parameters()] == [id(p) for p in encoder.parameters()]

    def forward(self, images, masks=None, *, restrict=True):
        assert images.shape[-2:] == (240, 240)
        x = self.patch_embed(images) + self.pos_embed
        indices = torch.arange(225, device=x.device).expand(len(x), -1)
        if masks is not None:
            x = j.apply_masks(x, masks)
            indices = torch.cat(masks, dim=0)
        # A patch is a 16px square; each 80px tile contains 5x5 patches.
        tile = (indices // 15 // 5) * 3 + (indices % 15 // 5)
        allowed = tile[:, :, None] == tile[:, None, :]
        for block in self.blocks:
            if not restrict:
                x = block(x)
                continue
            a = block.attn
            y = block.norm1(x)
            b, n, c = y.shape
            qkv = a.qkv(y).reshape(b, n, 3, a.num_heads, c // a.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attention = (q @ k.transpose(-2, -1)) * a.scale
            # Self edges always remain, even when a tile has one visible patch.
            attention = attention.masked_fill(~allowed[:, None], float("-inf"))
            attention = a.attn_drop(attention.softmax(dim=-1))
            y = (attention @ v).transpose(1, 2).reshape(b, n, c)
            y = a.proj_drop(a.proj(y))
            x = x + block.drop_path(y)
            x = x + block.drop_path(block.mlp(block.norm2(x)))
        return self.norm(x)


def context():
    cfg, base, train, val, parent = j.context()
    assert parent["run_id"] == BASELINE_ID, "Frozen comparison baseline changed"
    meta = {"schema": 1, "experiment": "ijepa_local_student_global_teacher",
            "baseline": parent, "config": cfg,
            "script_sha": j.file_sha(__file__),
            "test_sha": j.file_sha(ROOT / "server/test_ijepa_local_student.py"),
            "evaluation_adapter_sha": j.file_sha(ROOT / "server/evaluate_ijepa_fov.py"),
            "student": "same visible patches and global positions; attention restricted within canonical 80x80 tiles in every encoder block",
            "teacher": "unchanged global 240x240 forward; EMA weights from local student",
            "predictor": "unchanged global patch-level predictor",
            "initialization": "from scratch; identical seed and initial parameters to baseline",
            "evaluation": {"branches": ["student", "teacher"], "scopes": ["local80", "global240"],
                           "probe": asdict(base.probe), "precision": "fp32",
                           "pooling": "mean of all 225 outputs in raster order"},
            "limitation": "Isolates student pretraining connectivity conditional on a global teacher, global predictor and original global positional embeddings; not a full SEPA recreation."}
    meta["run_id"] = j.digest(meta)
    directory = ROOT / "runs-ijepa" / ("local-student-global-teacher-s0-" + meta["run_id"][:12])
    return cfg, base, train, val, meta, directory


def build(cfg, ipe):
    original = j.build(cfg, ipe)
    enc = LocalStudent(original[0])
    teacher = original[2]
    assert [n for n, _ in enc.named_parameters()] == [n for n, _ in teacher.named_parameters()]
    assert all(torch.equal(a, b) for a, b in zip(enc.parameters(), teacher.parameters()))
    return (enc, *original[1:])


def evaluate(directory, models, base, train, val, meta, progress):
    j.setup_runtime(base)
    checkpoint_sha = j.file_sha(directory / "latest.pt")
    audit = j.record_audit(train["records"])
    health_rows = j.health_indices(train["records"], excluded_ids=audit["conflicting_groups"])
    results = {}
    for branch, trained in (("student", models[0]), ("teacher", models[2])):
        encoder = vit_small(img_size=[240], patch_size=16)
        encoder.load_state_dict(trained.state_dict(), strict=True)
        encoder.requires_grad_(False).eval().cuda()
        for scope in ("local80", "global240"):
            name = f"{branch}-{scope}"
            identity = {"run_id": meta["run_id"], "checkpoint_sha": checkpoint_sha,
                        "branch": branch, "scope": scope, "evaluation": meta["evaluation"]}
            stem = directory / "evaluation" / f"{name}-result.json"
            receipt_path = stem.with_name(f"{name}-receipt.json")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                assert receipt["identity"] == identity
                assert j.file_sha(stem) == receipt["result_sha"]
                assert j.file_sha(stem.with_suffix(".pt")) == receipt["head_sha"]
                results[name] = json.loads(stem.read_text())["scores"]
                continue
            model = (LocalTiles(encoder) if scope == "local80" else j.ProbeAdapter(encoder)).cuda().eval()
            payloads = []
            for split, manifest in (("train", train), ("val", val)):
                payloads.append(j.cached_extract(directory / "evaluation" / f"{name}-{split}.pt",
                    model, "student", manifest, base, "cuda", identity,
                    lambda **kw: progress(phase="feature_extraction", branch=branch, scope=scope, split=split, **kw)))
            tr, va = payloads
            head, mean, std, trace = j.fit_trajectory(tr["features"], tr["labels"], 100, base, 0, "cuda",
                progress=lambda **kw: progress(phase="linear_probe", branch=branch, scope=scope, **kw))
            scores, predictions = j.linear_metrics(head, mean, std, va["features"], va["labels"], "cuda")
            scores["knn"] = j.knn_accuracy(tr["features"], tr["labels"], va["features"], va["labels"],
                                           100, k=20, temperature=0.07, device="cuda")
            health = j.feature_health(tr["features"][health_rows], tr["labels"][health_rows])
            assert len(trace) == base.probe.epochs
            j.save_tensor(stem.with_suffix(".pt"), {"identity": identity, "head": head.cpu().state_dict(),
                                                    "mean": mean, "std": std, "predictions": predictions})
            j.write_json(stem, {"identity": identity, "scores": scores, "trace": trace,
                               "feature_health": health, "n_train": len(tr["labels"]),
                               "n_val": len(va["labels"]), "utc": j.utc()})
            j.write_json(receipt_path, {"identity": identity, "result_sha": j.file_sha(stem),
                                       "head_sha": j.file_sha(stem.with_suffix(".pt"))})
            results[name] = scores
            del model, payloads, tr, va, head
            torch.cuda.empty_cache()
        del encoder
    return {"status": "complete", "metadata": meta, "checkpoint_sha": checkpoint_sha,
            "results": results, "utc": j.utc(), "shutdown_on_success": False}


def run(cfg, base, train, val, meta, directory, resume):
    receipt = json.loads((ROOT / "server-logs/ijepa_local_student_smoke.json").read_text())
    assert receipt["status"] == "passed" and receipt["metadata"] == meta
    directory.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "server-logs/driver.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ipe = math.ceil(len(train["records"]) / cfg["batch_size"])
    total = ipe * cfg["epochs"]
    def progress(**kw):
        row = {"pid": os.getpid(), "utc": j.utc(), "run_dir": str(directory), "run_id": meta["run_id"],
               "shutdown_on_success": False, **kw}
        j.write_json(ROOT / "server-logs/ijepa_local_student_status.json", row)
        print(json.dumps(row), flush=True)
    try:
        progress(phase="initializing", target_step=total)
        j.write_json(directory / "protocol.json", meta)
        j.write_json(directory / "runtime.json", {"torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
            "shutdown_on_success": False, "utc": j.utc()})
        models = build(cfg, ipe)
        checkpoint = directory / "latest.pt"
        start, prior_seconds = 0, 0.0
        if checkpoint.exists():
            if not resume:
                raise FileExistsError("Existing checkpoint requires --resume")
            start, prior_seconds = j.restore(checkpoint, models, meta)
            j.trim_history(directory / "history.jsonl", start)
        elif resume:
            raise FileNotFoundError(checkpoint)
        assert 0 <= start <= total
        started = last_log = time.monotonic()
        count, losses = 0, []
        with (directory / "history.jsonl").open("a", buffering=1) as history:
            for batch in j.loader(train, cfg, start):
                row = j.update(models, batch, cfg, total)
                step = row["step"]
                count += 1
                losses.append(row["loss"])
                if step % 25 == 0 or step == start + 1 or step % ipe == 0:
                    now = time.monotonic()
                    row.update(epoch=step/ipe, mean_loss=sum(losses)/len(losses),
                        recent_seconds_per_step=(now-last_log)/count,
                        training_seconds=prior_seconds+now-started,
                        max_gpu_bytes=torch.cuda.max_memory_allocated(), utc=j.utc())
                    history.write(json.dumps(row) + "\n")
                    progress(phase="training", target_step=total, **row)
                    last_log, count, losses = now, 0, []
                if step % ipe == 0:
                    j.save(checkpoint, models, step, meta, prior_seconds+time.monotonic()-started)
                    if step // ipe in cfg["milestones"]:
                        dest = directory / "milestones" / f"epoch-{step//ipe:03d}.pt"
                        dest.parent.mkdir(exist_ok=True)
                        # Reconstruct a milestone if interrupted between save and copy.
                        shutil.copy2(checkpoint, dest)
                        j.write_json(dest.with_suffix(".json"), {"step": step, "sha256": j.file_sha(dest),
                                                               "run_id": meta["run_id"], "utc": j.utc()})
        assert models[4]._step == models[5]._step == total
        # A process interrupted after the final save can resume directly into evaluation.
        final_milestone = directory / "milestones" / "epoch-100.pt"
        if not final_milestone.exists():
            final_milestone.parent.mkdir(exist_ok=True)
            shutil.copy2(checkpoint, final_milestone)
            j.write_json(final_milestone.with_suffix(".json"), {"step": total,
                "sha256": j.file_sha(final_milestone), "run_id": meta["run_id"], "utc": j.utc()})
        progress(phase="evaluating", step=total, target_step=total)
        summary = evaluate(directory, models, base, train, val, meta, progress)
        summary["training_seconds"] = torch.load(checkpoint, map_location="cpu", weights_only=True)["training_seconds"]
        j.write_json(directory / "summary.json", summary)
        progress(phase="complete", step=total, target_step=total, results=summary["results"])
    except BaseException as exc:
        progress(phase="needs_review", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    cfg, base, train, val, meta, directory = context()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(cfg["cpu_threads"])
    torch.set_num_interop_threads(1)
    j.setup_runtime(base)
    if args.smoke:
        from test_ijepa_local_student import smoke
        smoke(cfg, base, train, val, meta)
    else:
        run(cfg, base, train, val, meta, directory, args.resume)


if __name__ == "__main__":
    main()
