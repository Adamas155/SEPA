"""Run the seed-0, four-arm 100-epoch encoder experiment.

The coordinator owns the shared driver lock and runs at most three independent
GPU workers. Each worker uses the reviewed core training loop and stops at
epochs 10, 25, 50 and 100 to retain an immutable checkpoint before resuming the
same optimizer, RNG and 100-epoch schedule.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback


MILESTONE_EPOCHS = (10, 25, 50, 100)
EXPECTED_TRAIN_FINGERPRINT = "9a7ff7775290a65a26289b713781f92a9373db05261a96d77a58c3aebc4b13fd"
EXPECTED_VAL_FINGERPRINT = "2fb02ba5781f39d1b6503c4f10b95a5614d5d3ce446c9cf503d76754a0741df4"


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_file_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    if target < 4096:
        raise RuntimeError(f"Open-file hard limit {hard} is too low")
    if soft != resource.RLIM_INFINITY and soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def load_context(root):
    sys.path.insert(0, str(root / "src"))
    from sepa_plan_b.config import Config
    from sepa_plan_b.data import manifests

    config_path = root / "configs/native80_100ep.toml"
    config = Config.load(config_path)
    train_manifest, val_manifest = manifests(config)
    steps_per_epoch = math.ceil(len(train_manifest["records"]) / config.training.batch_size)
    expected = {
        "mode": "native80", "model": "vit_small", "relation": "none",
        "steps": steps_per_epoch * 100, "batch_size": 64,
        "warmup_steps": 4950, "checkpoint_every": steps_per_epoch,
        "probe_epochs": 200, "probe_lr": 0.005,
    }
    actual = {
        "mode": config.geometry.mode, "model": config.model.name,
        "relation": config.model.relation, "steps": config.training.steps,
        "batch_size": config.training.batch_size,
        "warmup_steps": config.training.warmup_steps,
        "checkpoint_every": config.training.checkpoint_every,
        "probe_epochs": config.probe.epochs, "probe_lr": config.probe.learning_rate,
    }
    if actual != expected:
        raise ValueError(f"Long-training configuration mismatch: {actual} != {expected}")
    if (train_manifest["fingerprint"], val_manifest["fingerprint"]) != (
            EXPECTED_TRAIN_FINGERPRINT, EXPECTED_VAL_FINGERPRINT):
        raise ValueError("Long-training data manifests differ from the audited seed-0 pilot")
    return config_path, config, train_manifest, val_manifest, steps_per_epoch


def job_id(job):
    return "full" if job["method"] == "full" else f"k{job['k']}"


def run_info(root, config, train_manifest, val_manifest, job):
    from sepa_plan_b.engine import identity, run_directory

    meta = identity(config, job["seed"], job["k"], job["method"], train_manifest, val_manifest)
    return meta, run_directory(config, meta)


def last_history_row(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 131072))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def verify_recent_history(directory, expected_step):
    rows = []
    with (directory / "history.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["step"] > expected_step - 50:
                rows.append(row)
    if not rows or rows[-1]["step"] != expected_step:
        raise RuntimeError(f"Training history does not end at step {expected_step}")
    warnings = [row for row in rows if row["collapse_flag"] is not False]
    if warnings:
        raise RuntimeError(f"Recent diagnostics need review at step {expected_step}: {len(warnings)} rows")


def snapshot_checkpoint(directory, checkpoint, meta, epoch, step):
    from sepa_plan_b.engine import load_checkpoint

    snapshots = directory / "milestones"
    snapshots.mkdir(exist_ok=True)
    target = snapshots / f"epoch-{epoch:03d}-step-{step:06d}.pt"
    if not target.exists():
        temporary = target.with_name(target.name + ".tmp")
        shutil.copy2(checkpoint, temporary)
        os.replace(temporary, target)
    saved = load_checkpoint(target)
    if saved["metadata"] != meta or saved["step"] != step:
        raise ValueError(f"Milestone checkpoint identity mismatch: {target}")
    return {"epoch": epoch, "step": step, "path": str(target), "sha256": sha256(target)}


def worker(spec_path):
    spec = read_json(spec_path)
    root = Path(spec["project"]).resolve()
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("Project and worker locations differ")
    configure_file_limit()
    sys.path.insert(0, str(root / "src"))
    import torch
    from sepa_plan_b.data import write_json
    from sepa_plan_b.engine import load_checkpoint, train

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config_path, config, train_manifest, val_manifest, steps_per_epoch = load_context(root)
    job = spec["job"]
    meta, directory = run_info(root, config, train_manifest, val_manifest, job)
    worker_dir = root / "server-logs/long100-workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    lock = (worker_dir / f"{job_id(job)}.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = worker_dir / f"{job_id(job)}.status.json"
    milestones = []

    def update(**fields):
        write_json(status_path, {"pid": os.getpid(), "job": job, "run_dir": str(directory),
                                "config": str(config_path), "updated_utc": utc(), **fields})

    try:
        update(status="running", step=0, target_step=config.training.steps, milestones=[])
        for epoch in MILESTONE_EPOCHS:
            target_step = epoch * steps_per_epoch
            checkpoint = directory / "latest.pt"
            current_step = load_checkpoint(checkpoint)["step"] if checkpoint.exists() else 0
            if current_step > target_step:
                snapshot = directory / "milestones" / f"epoch-{epoch:03d}-step-{target_step:06d}.pt"
                if not snapshot.exists():
                    raise RuntimeError(f"Run passed milestone {epoch} without a saved checkpoint")
            else:
                if current_step < target_step:
                    update(status="running", step=current_step, milestone_epoch=epoch,
                           milestone_target_step=target_step, milestones=milestones)
                    train(config, **job, resume=checkpoint.exists(), stop_after=target_step)
                verify_recent_history(directory, target_step)
            receipt = snapshot_checkpoint(directory, checkpoint, meta, epoch, target_step)
            milestones = [x for x in milestones if x["epoch"] != epoch] + [receipt]
            update(status="running", step=target_step, milestone_epoch=epoch, milestones=milestones)
        update(status="complete", step=config.training.steps, target_step=config.training.steps,
               milestones=milestones)
    except BaseException as exc:
        update(status="needs_review", error=f"{type(exc).__name__}: {exc}",
               traceback=traceback.format_exc(), milestones=milestones)
        raise
    finally:
        lock.close()


def coordinator(root, workers):
    root = root.resolve()
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("Project and runner locations differ")
    sys.path.insert(0, str(root / "src"))
    import torch
    from sepa_plan_b.data import write_json
    from sepa_plan_b.engine import source_hash

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config_path, config, train_manifest, val_manifest, steps_per_epoch = load_context(root)
    if not torch.cuda.is_available():
        raise RuntimeError("Long training requires CUDA")
    quota_path = Path("/sys/fs/cgroup/cpu.max")
    if quota_path.exists():
        quota, period = quota_path.read_text().split()
        if quota != "max" and workers * 8 > int(quota) / int(period):
            raise ValueError("Four loader processes plus four CPU threads per worker exceed the CPU quota")
    logs = root / "server-logs"
    logs.mkdir(exist_ok=True)
    (logs / "long100-workers").mkdir(exist_ok=True)
    lock = (logs / "driver.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = logs / "long100_status.json"
    jobs = [{"seed": 0, "k": k, "method": "sepa"} for k in (0, 3, 6)] + [
        {"seed": 0, "k": 0, "method": "full"}]
    pending = list(jobs)
    active = {}
    completed = []
    started = utc()

    def update(**fields):
        snapshots = []
        for pid, (process, job, directory) in active.items():
            row = last_history_row(directory / "history.jsonl")
            worker_status_path = logs / "long100-workers" / f"{job_id(job)}.status.json"
            worker_status = read_json(worker_status_path) if worker_status_path.exists() else {}
            snapshots.append({"pid": pid, "job": job_id(job), "step": row.get("step", worker_status.get("step", 0)),
                              "epoch": row.get("epoch"), "loss": row.get("loss"),
                              "process_alive": process.poll() is None})
        write_json(status_path, {"pid": os.getpid(), "started_utc": started, "updated_utc": utc(),
                                "phase": "training", "workers": workers,
                                "steps_per_epoch": steps_per_epoch, "target_steps": config.training.steps,
                                "active_workers": snapshots, "pending_jobs": [job_id(j) for j in pending],
                                "completed_jobs": completed, **fields})

    try:
        update(source_hash=source_hash(), config=str(config_path), gpu=torch.cuda.get_device_name(),
               train_fingerprint=train_manifest["fingerprint"], val_fingerprint=val_manifest["fingerprint"])
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                _, directory = run_info(root, config, train_manifest, val_manifest, job)
                spec_path = logs / "long100-workers" / f"{job_id(job)}.task.json"
                write_json(spec_path, {"project": str(root), "job": job})
                stream = (logs / "long100-workers" / f"{job_id(job)}.log").open("ab")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                            "--project", str(root), "--worker", str(spec_path)],
                                           stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                stream.close()
                active[process.pid] = (process, job, directory)
            for pid, (process, job, directory) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[pid]
                worker_status_path = logs / "long100-workers" / f"{job_id(job)}.status.json"
                worker_status = read_json(worker_status_path) if worker_status_path.exists() else {}
                if code or worker_status.get("status") != "complete":
                    raise RuntimeError(f"Worker {job_id(job)} exited {code}: "
                                       f"{worker_status.get('error', 'no worker status was written')}")
                completed.append({"job": job_id(job), "run_dir": str(directory),
                                  "milestones": worker_status["milestones"]})
            update()
            if active:
                time.sleep(10)
        update(phase="complete", pid=None, active_workers=[], pending_jobs=[])
    except BaseException as exc:
        update(phase="needs_review", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        for process, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, _, _ in active.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        lock.close()


def self_test(root):
    root = root.resolve()
    sys.path.insert(0, str(root / "src"))
    config_path, config, train_manifest, val_manifest, steps_per_epoch = load_context(root)
    assert steps_per_epoch == 1980
    assert tuple(e * steps_per_epoch for e in MILESTONE_EPOCHS) == (19800, 49500, 99000, 198000)
    assert config.training.steps == 198000
    assert Path(config.training.output_root).name == "runs-long100"
    assert len(train_manifest["records"]) == 126684 and len(val_manifest["records"]) == 5000
    print(json.dumps({"status": "passed", "config": str(config_path), "steps_per_epoch": steps_per_epoch,
                      "milestone_steps": [e * steps_per_epoch for e in MILESTONE_EPOCHS]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--worker")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = Path(args.project).resolve()
    if args.worker:
        worker(args.worker)
    elif args.self_test:
        self_test(root)
    else:
        coordinator(root, args.workers)


if __name__ == "__main__":
    main()
