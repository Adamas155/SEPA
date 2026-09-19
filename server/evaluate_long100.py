"""Evaluate all long100 milestones with the frozen seed-0 probe recipe."""
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
import subprocess
import sys
import time
import traceback


EXPECTED_ARMS = ("k0", "k3", "k6", "full")
EXPECTED_EPOCHS = (10, 25, 50, 100)
EXPECTED_STEPS = {10: 19800, 25: 49500, 50: 99000, 100: 198000}
EXPECTED_TRAIN_FINGERPRINT = "9a7ff7775290a65a26289b713781f92a9373db05261a96d77a58c3aebc4b13fd"
EXPECTED_VAL_FINGERPRINT = "2fb02ba5781f39d1b6503c4f10b95a5614d5d3ce446c9cf503d76754a0741df4"


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def configure_file_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    if target < 4096:
        raise RuntimeError(f"Open-file hard limit {hard} is too low")
    if soft != resource.RLIM_INFINITY and soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


def job_id(job):
    return f"{job['arm']}-e{job['epoch']:03d}"


def load_jobs(root, verify_hashes):
    status = read_json(root / "server-logs/long100_status.json")
    if status.get("phase") != "complete" or status.get("active_workers") or status.get("pending_jobs"):
        raise RuntimeError("The four-arm long100 training is not complete")
    completed = status.get("completed_jobs", [])
    if {row.get("job") for row in completed} != set(EXPECTED_ARMS) or len(completed) != len(EXPECTED_ARMS):
        raise ValueError("Unexpected completed long100 arm set")
    jobs = []
    for arm in EXPECTED_ARMS:
        row = next(item for item in completed if item["job"] == arm)
        milestones = row.get("milestones", [])
        if {item.get("epoch") for item in milestones} != set(EXPECTED_EPOCHS):
            raise ValueError(f"Incomplete milestones for {arm}")
        for epoch in EXPECTED_EPOCHS:
            item = next(value for value in milestones if value["epoch"] == epoch)
            checkpoint = Path(item["path"]).resolve()
            checkpoint.relative_to(root / "runs-long100")
            if item.get("step") != EXPECTED_STEPS[epoch] or not checkpoint.is_file():
                raise ValueError(f"Invalid milestone for {arm} epoch {epoch}")
            if verify_hashes and file_sha(checkpoint) != item.get("sha256"):
                raise ValueError(f"Checkpoint checksum mismatch: {checkpoint}")
            jobs.append({"arm": arm, "epoch": epoch, "step": item["step"],
                         "checkpoint": str(checkpoint), "checkpoint_sha256": item["sha256"]})
    return jobs


def validate_result(result, job):
    if result.get("checkpoint_step") != job["step"] or result.get("limit") != 0:
        raise ValueError(f"Evaluation scope mismatch for {job_id(job)}")
    if result.get("train_manifest") != EXPECTED_TRAIN_FINGERPRINT or result.get("val_manifest") != EXPECTED_VAL_FINGERPRINT:
        raise ValueError(f"Dataset mismatch for {job_id(job)}")
    if result.get("n_train_labels") != 126684 or result.get("n_val") != 5000:
        raise ValueError(f"Evaluation counts mismatch for {job_id(job)}")
    scores = result.get("scores", {})
    if set(scores) != {"top1", "top5", "knn"} or any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
        raise ValueError(f"Invalid scores for {job_id(job)}")


def existing_result(output, job):
    result_path = output / "results" / f"{job_id(job)}.json"
    receipt_path = output / "results" / f"{job_id(job)}.receipt.json"
    if not result_path.exists() and not receipt_path.exists():
        return None
    if not result_path.exists() or not receipt_path.exists() or not result_path.with_suffix(".pt").exists():
        raise ValueError(f"Partial evaluation artifact for {job_id(job)}")
    receipt = read_json(receipt_path)
    expected = {key: job[key] for key in ("arm", "epoch", "step", "checkpoint", "checkpoint_sha256")}
    if receipt.get("job") != expected or receipt.get("result_sha256") != file_sha(result_path):
        raise ValueError(f"Evaluation receipt mismatch for {job_id(job)}")
    if receipt.get("head_sha256") != file_sha(result_path.with_suffix(".pt")):
        raise ValueError(f"Classifier checksum mismatch for {job_id(job)}")
    result = read_json(result_path)
    validate_result(result, job)
    return result


def worker(spec_path):
    spec = read_json(spec_path)
    root, output, job = Path(spec["project"]).resolve(), Path(spec["output"]).resolve(), spec["job"]
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("Project and worker locations differ")
    output.relative_to(root / "diagnostics")
    sys.path.insert(0, str(root / "src"))
    import torch
    from sepa_plan_b.evaluation import probe

    configure_file_limit()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    status_path = output / "workers" / f"{job_id(job)}.status.json"
    lock = (output / "workers" / f"{job_id(job)}.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def update(**fields):
        write_json(status_path, {"pid": os.getpid(), "job": job_id(job), "updated_utc": utc(), **fields})

    try:
        reused = existing_result(output, job)
        if reused is None:
            update(status="running", phase="feature_extraction_and_probe")
            result_path = output / "results" / f"{job_id(job)}.json"
            result = probe(job["checkpoint"], output=result_path, device="cuda")
            saved = read_json(result_path)
            validate_result(saved, job)
            receipt = {
                "job": {key: job[key] for key in ("arm", "epoch", "step", "checkpoint", "checkpoint_sha256")},
                "result_sha256": file_sha(result_path), "head_sha256": file_sha(result_path.with_suffix(".pt")),
                "scores": saved["scores"], "completed_utc": utc(),
            }
            write_json(output / "results" / f"{job_id(job)}.receipt.json", receipt)
            reused = saved
        update(status="complete", phase="complete", scores=reused["scores"])
    except BaseException as exc:
        update(status="needs_review", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        lock.close()


def write_summary(output, jobs):
    rows = []
    for job in jobs:
        result = existing_result(output, job)
        if result is None:
            raise RuntimeError(f"Missing completed result: {job_id(job)}")
        rows.append({"arm": job["arm"], "epoch": job["epoch"], "step": job["step"], **result["scores"]})
    rows.sort(key=lambda row: (row["epoch"], EXPECTED_ARMS.index(row["arm"])))
    summary = {"status": "complete", "completed_utc": utc(), "seed": 0,
               "probe_recipe": {"learning_rate": 0.005, "epochs": 200, "batch_size": 128},
               "train_records": 126684, "val_records": 5000, "results": rows,
               "interpretation": "Single-seed milestone trend; official validation was not used to select the frozen probe recipe."}
    write_json(output / "summary.json", summary)
    lines = ["# Long100 milestone frozen evaluation", "",
             "Seed=0; frozen encoder; SGD LR=0.005, 200 classifier epochs, batch=128.", "",
             "| Encoder epoch | k=0 | k=3 | k=6 | full |", "|---:|---:|---:|---:|---:|"]
    for epoch in EXPECTED_EPOCHS:
        values = {row["arm"]: row for row in rows if row["epoch"] == epoch}
        lines.append("| " + str(epoch) + " | " + " | ".join(f"{100*values[arm]['top1']:.2f}%" for arm in EXPECTED_ARMS) + " |")
    lines.extend(["", "Each cell is ImageNet-100 validation Top-1. Top-5 and kNN are in summary.json.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def coordinator(root, workers, shutdown_on_success):
    root = root.resolve()
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("Project and coordinator locations differ")
    sys.path.insert(0, str(root / "src"))
    import torch

    configure_file_limit()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Milestone evaluation requires CUDA")
    jobs = load_jobs(root, verify_hashes=True)
    output = root / "diagnostics/long100-milestones-seed0"
    (output / "workers").mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(exist_ok=True)
    logs = root / "server-logs"
    lock = (logs / "driver.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending, active, completed = [], {}, []
    for job in jobs:
        result = existing_result(output, job)
        if result is None:
            pending.append(job)
        else:
            completed.append({"job": job_id(job), "scores": result["scores"], "reused": True})
    status_path = logs / "long100_eval_status.json"
    started = utc()

    def update(**fields):
        active_rows = []
        for pid, (process, job) in active.items():
            path = output / "workers" / f"{job_id(job)}.status.json"
            detail = read_json(path) if path.exists() else {}
            active_rows.append({"pid": pid, "job": job_id(job), "process_alive": process.poll() is None,
                                "phase": detail.get("phase"), "updated_utc": detail.get("updated_utc")})
        write_json(status_path, {"pid": os.getpid(), "started_utc": started, "updated_utc": utc(),
                                "phase": "evaluating", "workers": workers, "output": str(output),
                                "active_workers": active_rows, "pending_jobs": [job_id(job) for job in pending],
                                "completed_jobs": completed, "shutdown_on_success": shutdown_on_success, **fields})

    try:
        update(gpu=torch.cuda.get_device_name(), jobs=len(jobs), runner_sha256=file_sha(__file__))
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                spec_path = output / "workers" / f"{job_id(job)}.task.json"
                write_json(spec_path, {"project": str(root), "output": str(output), "job": job})
                stream = (output / "workers" / f"{job_id(job)}.log").open("ab")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                            "--project", str(root), "--worker", str(spec_path)],
                                           stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                stream.close()
                active[process.pid] = (process, job)
            for pid, (process, job) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[pid]
                path = output / "workers" / f"{job_id(job)}.status.json"
                state = read_json(path) if path.exists() else {}
                if code or state.get("status") != "complete":
                    raise RuntimeError(f"Worker {job_id(job)} exited {code}: {state.get('error', 'no status')}")
                result = existing_result(output, job)
                completed.append({"job": job_id(job), "scores": result["scores"], "reused": False})
            update()
            if active:
                time.sleep(10)
        summary = write_summary(output, jobs)
        update(phase="complete", pid=None, active_workers=[], pending_jobs=[], summary=str(output / "summary.json"))
        os.sync()
        if shutdown_on_success:
            write_json(logs / "long100_eval_shutdown.json", {"status": "triggered", "triggered_utc": utc(),
                                                              "summary": str(output / "summary.json"),
                                                              "command": "/bin/bash /usr/bin/shutdown"})
            os.sync()
            subprocess.run(["/bin/bash", "/usr/bin/shutdown"], check=True)
        return summary
    except BaseException as exc:
        update(phase="needs_review", pid=None, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        for process, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, _ in active.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        lock.close()


def self_test(root):
    jobs = load_jobs(root.resolve(), verify_hashes=False)
    assert len(jobs) == 16
    assert [job_id(job) for job in jobs] == [f"{arm}-e{epoch:03d}" for arm in EXPECTED_ARMS for epoch in EXPECTED_EPOCHS]
    assert all(Path(job["checkpoint"]).stat().st_size > 100_000_000 for job in jobs)
    print(json.dumps({"status": "passed", "jobs": len(jobs), "first": job_id(jobs[0]), "last": job_id(jobs[-1])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--worker")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--shutdown-on-success", action="store_true")
    args = parser.parse_args()
    root = Path(args.project).resolve()
    if args.worker:
        worker(args.worker)
    elif args.self_test:
        self_test(root)
    else:
        coordinator(root, args.workers, args.shutdown_on_success)


if __name__ == "__main__":
    main()
