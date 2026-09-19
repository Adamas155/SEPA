"""Power off the instance only after the four-arm long100 run is complete."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


EXPECTED_JOBS = {"k0", "k3", "k6", "full"}
EXPECTED_EPOCHS = {10, 25, 50, 100}
EXPECTED_STEPS = {10: 19800, 25: 49500, 50: 99000, 100: 198000}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_completion(project, status):
    if status.get("phase") != "complete" or status.get("pid") is not None:
        return False, "coordinator has not reported complete"
    if status.get("active_workers") or status.get("pending_jobs"):
        return False, "workers or pending jobs remain"
    completed = status.get("completed_jobs", [])
    jobs = {item.get("job") for item in completed}
    if jobs != EXPECTED_JOBS or len(completed) != len(EXPECTED_JOBS):
        return False, f"completed jobs differ: {sorted(x for x in jobs if x)}"
    for item in completed:
        milestones = item.get("milestones", [])
        epochs = {row.get("epoch") for row in milestones}
        if epochs != EXPECTED_EPOCHS or len(milestones) != len(EXPECTED_EPOCHS):
            return False, f"incomplete milestones for {item.get('job')}"
        for row in milestones:
            epoch = row["epoch"]
            path = Path(row["path"]).resolve()
            try:
                path.relative_to(project / "runs-long100")
            except ValueError:
                return False, f"milestone path escapes runs-long100: {path}"
            if row.get("step") != EXPECTED_STEPS[epoch]:
                return False, f"wrong milestone step for {item.get('job')} epoch {epoch}"
            if not path.is_file() or path.stat().st_size < 100_000_000:
                return False, f"missing or truncated milestone: {path}"
            checksum = row.get("sha256", "")
            if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                return False, f"invalid checkpoint receipt for {path}"
    return True, "all four jobs and sixteen milestone checkpoints are complete"


def watch(project, poll_seconds):
    project = project.resolve()
    if project != Path(__file__).resolve().parents[1]:
        raise ValueError("Project and watcher locations differ")
    status_path = project / "server-logs/long100_status.json"
    receipt_path = project / "server-logs/long100_shutdown.json"
    shutdown = Path("/usr/bin/shutdown")
    if not shutdown.is_file() or not os.access(shutdown, os.X_OK):
        raise FileNotFoundError("AutoDL shutdown command is unavailable: /usr/bin/shutdown")
    state = {"pid": os.getpid(), "armed_utc": utc(), "status": "armed",
             "training_status": str(status_path), "shutdown_command": str(shutdown),
             "condition": "four jobs complete with epochs 10/25/50/100 checkpoints"}
    write_json(receipt_path, state)
    last_recorded = 0.0
    try:
        while True:
            try:
                status = read_json(status_path)
                ready, reason = validate_completion(project, status)
                phase = status.get("phase")
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                ready, reason, phase = False, f"status unavailable: {exc}", None
            now = time.monotonic()
            if ready:
                # Ensure the coordinator has finished its final file writes and
                # the completion state remains stable before powering off.
                time.sleep(30)
                stable = read_json(status_path)
                ready, reason = validate_completion(project, stable)
                if not ready:
                    continue
                state.update(status="triggered", triggered_utc=utc(), reason=reason)
                write_json(receipt_path, state)
                os.sync()
                # AutoDL supplies /usr/bin/shutdown as a shell fragment without
                # a shebang, so direct exec raises ENOEXEC. Invoke it explicitly
                # through bash, matching an interactive shell invocation.
                subprocess.run(["/bin/bash", str(shutdown)], check=True)
                state.update(status="shutdown_command_returned", command_returned_utc=utc())
                write_json(receipt_path, state)
                return
            if now - last_recorded >= 300 or phase == "needs_review":
                state.update(status="armed", checked_utc=utc(), training_phase=phase, reason=reason)
                write_json(receipt_path, state)
                last_recorded = now
            time.sleep(poll_seconds)
    except BaseException as exc:
        state.update(status="watcher_error", error=f"{type(exc).__name__}: {exc}",
                     traceback=traceback.format_exc(), failed_utc=utc())
        write_json(receipt_path, state)
        raise


def self_test(project):
    good = {
        "phase": "complete", "pid": None, "active_workers": [], "pending_jobs": [],
        "completed_jobs": [
            {"job": job, "milestones": [
                {"epoch": epoch, "step": EXPECTED_STEPS[epoch],
                 "path": str(project / "runs-long100" / job / f"epoch-{epoch}.pt"),
                 "sha256": "0" * 64}
                for epoch in sorted(EXPECTED_EPOCHS)]}
            for job in sorted(EXPECTED_JOBS)]}
    # Structural test: nonexistent checkpoint files must prevent shutdown.
    ready, reason = validate_completion(project, good)
    assert not ready and "missing or truncated" in reason
    bad = dict(good, phase="needs_review")
    assert validate_completion(project, bad)[0] is False
    assert Path("/usr/bin/shutdown").is_file()
    print(json.dumps({"status": "passed", "shutdown_command": "/usr/bin/shutdown"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.poll_seconds <= 600:
        parser.error("--poll-seconds must be between 10 and 600")
    project = Path(args.project).resolve()
    if args.self_test:
        self_test(project)
    else:
        watch(project, args.poll_seconds)


if __name__ == "__main__":
    main()
