"""Paired receptive-field evaluation of the frozen, completed I-JEPA run.

Only attention connectivity changes. Each 16x16 patch retains its pixels and
original 15x15-grid positional embedding. Nine local calls are batched in the
upstream encoder and reassembled in original raster order before mean pooling.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import ijepa_baseline as j
from src.models.vision_transformer import vit_small
from sepa_plan_b.data import ImageTiles
from diagnostic_lib import cached_extract, file_sha, save_tensor, feature_health, health_indices, record_audit

torch = j.torch
ROOT = j.ROOT
RUN = ROOT / "runs-ijepa/official-small240-s0-3e242d05b38e"
CHECKPOINT_SHA = "7a651508975a97ab68b7a67129bea0cb737f1ef8ff2f3bb9ffb154fa7c0cd412"
BRANCHES = ("student", "teacher")


def context():
    cfg, base, train, val, old_meta = j.context()
    summary = json.loads((RUN / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["metadata"] == old_meta
    assert file_sha(RUN / "latest.pt") == summary["checkpoint_sha"] == CHECKPOINT_SHA
    protocol = {"schema": 1, "experiment": "frozen_ijepa_global_vs_local80",
                "source_run_id": old_meta["run_id"], "checkpoint_sha": CHECKPOINT_SHA,
                "train_fingerprint": train["fingerprint"], "val_fingerprint": val["fingerprint"],
                "script_sha": file_sha(__file__), "baseline_adapter_sha": old_meta["adapter_sha"],
                "diagnostic_sha": file_sha(ROOT / "server/diagnostic_lib.py"),
                "probe": asdict(base.probe), "seed": 0, "branches": list(BRANCHES),
                "precision": "fp32", "input": "same center-square bicubic 240px RGB images",
                "local": "nine independent 5x5 patch sequences; original global position embeddings; no resizing",
                "pooling": "mean of all 225 outputs in original raster order for both conditions",
                "limitation": "Frozen whole-image-pretrained weights. Local evaluation changes the inference distribution; this is not a local-pretraining ablation."}
    protocol["experiment_id"] = j.digest(protocol)
    output = ROOT / "diagnostics" / ("ijepa-fov-seed0-" + protocol["experiment_id"][:12])
    return base, train, val, protocol, output


def configure():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg = j.Config.load(ROOT / "configs/native80_100ep.toml")
    j.setup_runtime(cfg)


def load_encoder(branch):
    ck = torch.load(RUN / "latest.pt", map_location="cpu", weights_only=True)
    assert ck["step"] == 198000
    enc = vit_small(img_size=[240], patch_size=16)
    enc.load_state_dict(ck["encoder" if branch == "student" else "teacher"], strict=True)
    return enc.requires_grad_(False).eval().cuda()


class LocalTiles(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        grid = torch.arange(225).reshape(15, 15)
        masks = torch.stack([grid[r*5:(r+1)*5, c*5:(c+1)*5].flatten()
                             for r in range(3) for c in range(3)])
        assert torch.equal(masks.flatten().sort().values, torch.arange(225))
        self.register_buffer("indices", masks)
        self.register_buffer("raster_order", masks.flatten().argsort())

    def encode(self, tiles):
        b = len(tiles)
        masks = [idx.expand(b, -1) for idx in self.indices]
        # Upstream applies the masks BEFORE any attention, returning tile-major
        # batches: tile0/image0,...,tile0/imageB,...,tile8/imageB.
        output = self.encoder(j.join_tiles(tiles), masks=masks)
        output = output.reshape(9, b, 25, 384).permute(1, 0, 2, 3).reshape(b, 225, 384)
        return output.index_select(1, self.raster_order)


def read_baseline(branch, train, val, base):
    result_path = RUN / "evaluation" / f"{branch}-result.json"
    result = json.loads(result_path.read_text())
    identity = result["identity"]
    assert identity["checkpoint_sha"] == CHECKPOINT_SHA and identity["branch"] == branch
    assert identity["probe_config"] == asdict(base.probe)
    head = torch.load(result_path.with_suffix(".pt"), map_location="cpu", weights_only=True)
    assert head["identity"] == identity
    features = [cached_extract(RUN / "evaluation" / f"{branch}-{split}.pt", None, "student",
                               manifest, base, "cuda", identity, lambda **kw: None)
                for split, manifest in (("train", train), ("val", val))]
    assert len(features[0]["labels"]) == 126684 and len(features[1]["labels"]) == 5000
    return result, head, features


@torch.no_grad()
def smoke(base, train, val, protocol):
    checks = {}
    for branch in BRANCHES:
        print(json.dumps({"phase": "smoke", "branch": branch}), flush=True)
        enc = load_encoder(branch)
        local = LocalTiles(enc).cuda().eval()
        sample = torch.stack([ImageTiles(val, base)[i]["canonical"] for i in (0, 51, 102)]).cuda()
        vectorized = local.encode(sample)
        manual_tiles = []
        for tile in range(9):
            # Independent reference: actually give patch_embed a native 80x80
            # image and attach the corresponding original 25 position vectors.
            x = enc.patch_embed(sample[:, tile]) + enc.pos_embed[:, local.indices[tile]]
            for block in enc.blocks:
                x = block(x)
            manual_tiles.append(enc.norm(x))
        manual = torch.stack(manual_tiles, dim=1).reshape(len(sample), 225, 384)
        manual = manual.index_select(1, local.raster_order)
        torch.testing.assert_close(vectorized, manual, rtol=3e-5, atol=3e-5)
        shuffled = local.encode(sample.flip(0)).flip(0)
        torch.testing.assert_close(vectorized, shuffled, rtol=3e-5, atol=3e-5)
        changed = sample.clone()
        changed[:, :4] = 0
        changed[:, 5:] = 0
        changed_local = local.encode(changed)
        # The central tile receives no information from any other tile.
        torch.testing.assert_close(vectorized[:, local.indices[4]],
                                   changed_local[:, local.indices[4]], rtol=0, atol=0)
        changed_global = (enc(j.join_tiles(sample))[:, local.indices[4]] -
                          enc(j.join_tiles(changed))[:, local.indices[4]]).abs().max().item()
        assert changed_global > 1e-3, "Global negative control did not respond"
        assert all(not p.requires_grad and p.grad is None for p in enc.parameters())
        old, old_head, old_features = read_baseline(branch, train, val, base)
        # Reproduce ALL official validation features and predictions on this GPU.
        fresh = []
        loader = torch.utils.data.DataLoader(ImageTiles(val, base), batch_size=128,
                                             num_workers=4, generator=j.generator(0, "fov_smoke"))
        for batch in loader:
            fresh.append(enc(j.join_tiles(batch["canonical"].cuda())).mean(1).cpu())
        fresh = torch.cat(fresh)
        torch.testing.assert_close(fresh, old_features[1]["features"], rtol=3e-5, atol=3e-5)
        head = torch.nn.Linear(384, 100).cuda()
        head.load_state_dict(old_head["head"])
        scores, predictions = j.linear_metrics(head, old_head["mean"], old_head["std"],
                                               fresh, old_features[1]["labels"], "cuda")
        assert all(scores[key] == old["scores"][key] for key in ("top1", "top5"))
        assert torch.equal(predictions, old_head["predictions"])
        # Also check the final training batch, including manifest ordering.
        train_indices = list(range(len(train["records"])-92, len(train["records"])))
        subset = torch.utils.data.Subset(ImageTiles(train, base), train_indices)
        last = next(iter(torch.utils.data.DataLoader(subset, batch_size=92, num_workers=4)))
        fresh_train = enc(j.join_tiles(last["canonical"].cuda())).mean(1).cpu()
        torch.testing.assert_close(fresh_train, old_features[0]["features"][-92:], rtol=3e-5, atol=3e-5)
        # Confirm independent pooling still yields exactly one vector per image.
        assert vectorized.mean(1).shape == (3, 384)
        checks[branch] = {"native80_reference_max_error": (vectorized-manual).abs().max().item(),
                          "other_tiles_do_not_affect_center": True, "global_control_change": changed_global,
                          "sample_order_preserved": True, "all_225_patches_used_once": True,
                          "global_val_max_error": (fresh-old_features[1]["features"]).abs().max().item(),
                          "global_val_predictions_exact": True, "global_scores": scores,
                          "last_train_batch_max_error": (fresh_train-old_features[0]["features"][-92:]).abs().max().item(),
                          "encoder_frozen": True}
        del local, enc, old_features, head, fresh, sample, changed, vectorized, manual, x
        torch.cuda.empty_cache()
    receipt = {"status": "passed", "protocol": protocol, "checks": checks, "utc": j.utc()}
    j.write_json(ROOT / "server-logs/ijepa_fov_smoke.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def worker(branch, base, train, val, protocol, output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{branch}.status.json"
    def progress(**kw):
        j.write_json(path, {"branch": branch, "pid": os.getpid(), "utc": j.utc(), **kw})
    lock = (output / f"{branch}.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        progress(phase="loading")
        enc = load_encoder(branch)
        model = LocalTiles(enc).cuda().eval()
        metadata = {"protocol": protocol, "branch": branch, "condition": "local80"}
        features = []
        for split, manifest in (("train", train), ("val", val)):
            features.append(cached_extract(output / f"{branch}-{split}.pt", model, "student", manifest,
                base, "cuda", metadata, lambda **kw: progress(phase="extract", split=split, **kw)))
        tr, va = features
        progress(phase="linear_probe", epoch=0, epochs=base.probe.epochs)
        head, mean, std, trace = j.fit_trajectory(tr["features"], tr["labels"], 100, base, 0, "cuda",
                                                progress=lambda **kw: progress(phase="linear_probe", **kw))
        scores, predictions = j.linear_metrics(head, mean, std, va["features"], va["labels"], "cuda")
        progress(phase="knn_and_health")
        scores["knn"] = j.knn_accuracy(tr["features"], tr["labels"], va["features"], va["labels"],
                                       100, k=20, temperature=0.07, device="cuda")
        audit = record_audit(train["records"])
        health_rows = health_indices(train["records"], excluded_ids=audit["conflicting_groups"])
        health = feature_health(tr["features"][health_rows], tr["labels"][health_rows])
        old = json.loads((RUN / "evaluation" / f"{branch}-result.json").read_text())
        old_head = torch.load(RUN / "evaluation" / f"{branch}-result.pt", map_location="cpu", weights_only=True)
        global_correct = old_head["predictions"] == va["labels"]
        local_correct = predictions == va["labels"]
        paired = {"global_only_correct": (global_correct & ~local_correct).sum().item(),
                  "local_only_correct": (~global_correct & local_correct).sum().item(),
                  "both_correct": (global_correct & local_correct).sum().item(),
                  "both_wrong": (~global_correct & ~local_correct).sum().item()}
        result = {"metadata": metadata, "scores": scores, "global_scores": old["scores"],
                  "global_minus_local_top1_pp": 100*(old["scores"]["top1"]-scores["top1"]),
                  "paired": paired, "feature_health": health, "trace": trace,
                  "n_train_labels": len(tr["labels"]), "n_val": len(va["labels"]), "utc": j.utc()}
        stem = output / f"{branch}-result.json"
        save_tensor(stem.with_suffix(".pt"), {"metadata": metadata, "head": head.cpu().state_dict(),
                                             "mean": mean, "std": std, "predictions": predictions})
        j.write_json(stem, result)
        j.write_json(output / f"{branch}-receipt.json", {"result_sha": file_sha(stem),
                     "head_sha": file_sha(stem.with_suffix(".pt")), "metadata": metadata})
        progress(phase="complete", scores=scores, global_minus_local_top1_pp=result["global_minus_local_top1_pp"])
    except BaseException as exc:
        progress(phase="needs_review", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        lock.close()


def completed_result(output, branch, protocol):
    stem = output / f"{branch}-result.json"
    receipt = json.loads((output / f"{branch}-receipt.json").read_text())
    assert file_sha(stem) == receipt["result_sha"]
    assert file_sha(stem.with_suffix(".pt")) == receipt["head_sha"]
    result = json.loads(stem.read_text())
    assert result["metadata"] == receipt["metadata"] == {"protocol": protocol, "branch": branch, "condition": "local80"}
    assert (result["n_train_labels"], result["n_val"]) == (126684, 5000)
    assert len(result["trace"]) == 200
    return result


def coordinator(base, train, val, protocol, output, shutdown):
    smoke_receipt = json.loads((ROOT / "server-logs/ijepa_fov_smoke.json").read_text())
    assert smoke_receipt["status"] == "passed" and smoke_receipt["protocol"] == protocol
    output.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "server-logs/driver.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    processes, handles = {}, []
    status_path = ROOT / "server-logs/ijepa_fov_status.json"
    def progress(**kw):
        state = {"pid": os.getpid(), "utc": j.utc(), "output": str(output),
                 "protocol_id": protocol["experiment_id"], "shutdown_on_success": shutdown, **kw}
        j.write_json(status_path, state)
        print(json.dumps(state), flush=True)
    try:
        j.write_json(output / "protocol.json", protocol)
        for branch in BRANCHES:
            if (output / f"{branch}-receipt.json").exists():
                completed_result(output, branch, protocol)
                continue
            log = (output / f"{branch}.log").open("ab")
            handles.append(log)
            env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}
            processes[branch] = subprocess.Popen([sys.executable, "-u", __file__, "--worker", branch],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
        while True:
            snapshots = {}
            for branch in BRANCHES:
                path = output / f"{branch}.status.json"
                snapshots[branch] = json.loads(path.read_text()) if path.exists() else {"phase": "starting"}
                if branch in processes and processes[branch].poll() not in (None, 0):
                    raise RuntimeError(f"Worker {branch} exited {processes[branch].returncode}")
            progress(phase="running", workers=snapshots)
            if all(p.poll() == 0 for p in processes.values()):
                break
            time.sleep(10)
        results = {b: completed_result(output, b, protocol) for b in BRANCHES}
        summary = {"status": "complete", "utc": j.utc(), "protocol": protocol,
                   "results": {b: {k: results[b][k] for k in ("scores", "global_scores", "global_minus_local_top1_pp", "paired", "feature_health")}
                               for b in BRANCHES}}
        j.write_json(output / "summary.json", summary)
        lines = ["# Frozen I-JEPA receptive-field comparison", "", protocol["limitation"], "",
                 "| Branch | Global Top-1 | Local80 Top-1 | Drop (pp) | Global kNN | Local80 kNN |",
                 "|---|---:|---:|---:|---:|---:|"]
        for branch, result in results.items():
            old, new = result["global_scores"], result["scores"]
            lines.append(f"| {branch} | {100*old['top1']:.2f}% | {100*new['top1']:.2f}% | {result['global_minus_local_top1_pp']:.2f} | {100*old['knn']:.2f}% | {100*new['knn']:.2f}% |")
        (output / "summary.md").write_text("\n".join(lines)+"\n")
        progress(phase="complete", results={b: results[b]["scores"] for b in BRANCHES})
        if shutdown:
            j.write_json(output / "shutdown.json", {"status": "requested", "utc": j.utc(),
                                                    "summary_sha": file_sha(output / "summary.json")})
            os.sync()
            subprocess.run(["/bin/bash", "/usr/bin/shutdown"], check=True)
    except BaseException as exc:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        progress(phase="needs_review", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        for log in handles:
            log.close()
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", choices=BRANCHES)
    parser.add_argument("--shutdown-on-success", action="store_true")
    args = parser.parse_args()
    configure()
    base, train, val, protocol, output = context()
    if args.smoke:
        smoke(base, train, val, protocol)
    elif args.worker:
        worker(args.worker, base, train, val, protocol, output)
    else:
        coordinator(base, train, val, protocol, output, args.shutdown_on_success)
