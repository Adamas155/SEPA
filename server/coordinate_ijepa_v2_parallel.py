"""Run the second V2 arm beside an existing first arm, then evaluate both."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import train_ijepa_v2 as v

ROOT = v.ROOT
LOGS = ROOT / "server-logs"


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    _, spec, _, _, _, shared = v.context()
    k0_path = LOGS / "ijepa_v2_k0_status.json"
    k3_path = LOGS / "ijepa_v2_k3_status.json"
    state_path = LOGS / "ijepa_v2_parallel_status.json"
    k0 = read(k0_path)
    if not k0 or k0.get("phase") not in {"training", "screen_training_complete"}:
        raise RuntimeError(f"k0 is not resumably running/complete: {k0}")
    k0_pid = k0["pid"]
    k3_log = (LOGS / "ijepa_v2_k3.log").open("ab")
    args = [sys.executable, "-u", str(ROOT / "server/train_ijepa_v2.py"),
            "--worker", "--arm", "k3"]
    *_, k3_dir = v.context("k3")
    if (k3_dir / "latest.pt").exists():
        args.append("--resume")
    k3_proc = subprocess.Popen(args, stdout=k3_log, stderr=subprocess.STDOUT)
    def update(**kw):
        v.j.write_json(state_path, {"pid": os.getpid(), "utc": v.j.utc(),
            "screen_id": shared["screen_id"], "k0_pid": k0_pid,
            "k3_pid": k3_proc.pid, "shutdown_on_success": False, **kw})
    try:
        update(phase="parallel_training")
        while True:
            a, b = read(k0_path), read(k3_path)
            if a and a.get("phase") == "needs_review":
                raise RuntimeError(f"k0 failed: {a.get('error')}")
            if b and b.get("phase") == "needs_review":
                raise RuntimeError(f"k3 failed: {b.get('error')}")
            k0_done = bool(a and a.get("phase") == "screen_training_complete")
            k3_done = bool(b and b.get("phase") == "screen_training_complete")
            update(phase="parallel_training", k0_step=a.get("step") if a else None,
                   k3_step=b.get("step") if b else None,
                   k0_complete=k0_done, k3_complete=k3_done)
            if not k0_done and not alive(k0_pid):
                raise RuntimeError("k0 process exited without a complete status")
            if k3_proc.poll() not in (None, 0):
                raise RuntimeError(f"k3 process exited {k3_proc.returncode}")
            if k0_done and k3_done:
                break
            time.sleep(10)
        assert k3_proc.wait() == 0
        update(phase="evaluating", k0_complete=True, k3_complete=True)
        evaluation_log = (LOGS / "ijepa_v2_evaluation.log").open("ab")
        try:
            result = subprocess.run([sys.executable, "-u", str(ROOT / "server/train_ijepa_v2.py"),
                                     "--evaluate"], stdout=evaluation_log,
                                    stderr=subprocess.STDOUT)
        finally:
            evaluation_log.close()
        if result.returncode:
            raise RuntimeError(f"Evaluation exited {result.returncode}")
        update(phase="complete", k0_complete=True, k3_complete=True)
    except BaseException as exc:
        update(phase="needs_review", error=f"{type(exc).__name__}: {exc}",
               traceback=traceback.format_exc())
        if k3_proc.poll() is None:
            k3_proc.terminate()
        raise
    finally:
        k3_log.close()


if __name__ == "__main__":
    main()
