"""Power off only after the complete SEPA V2 screen passes artifact checks."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import train_ijepa_v2 as v

ROOT = v.ROOT
LOGS = ROOT / "server-logs"
EXPECTED = {"ijepa-baseline-e25", "v2-k0-e25", "v2-k3-e25"}


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    _, spec, _, _, _, shared = v.context()
    screen_id = shared["screen_id"]
    parallel = read(LOGS / "ijepa_v2_parallel_status.json")
    assert parallel["screen_id"] == screen_id and parallel["phase"] == "complete"
    output = ROOT / "diagnostics" / f"ijepa-v2-screen-{screen_id[:12]}"
    summary_path = output / "summary.json"
    summary = read(summary_path)
    assert summary["status"] == "complete" and summary["shared"] == shared
    assert summary["imageNet100_val_used"] is False
    assert summary["shutdown_on_success"] is False
    assert set(summary["results"]) == EXPECTED
    for candidate in EXPECTED:
        result_path = output / f"{candidate}-result.json"
        head_path = result_path.with_suffix(".pt")
        receipt = read(output / f"{candidate}-receipt.json")
        result = read(result_path)
        assert result["identity"] == receipt["identity"]
        assert receipt["result_sha"] == v.j.file_sha(result_path)
        assert receipt["head_sha"] == v.j.file_sha(head_path)
        assert result["scores"] == summary["results"][candidate]["scores"]
        assert result["n_probe_train"] > 100000 and result["n_dev"] > 10000
    checkpoints = {}
    for arm in v.ARMS:
        *_, meta, directory = v.context(arm)
        status = read(LOGS / f"ijepa_v2_{arm}_status.json")
        assert status["phase"] == "screen_training_complete"
        assert status["step"] == 1980 * spec["screen_epochs"] == 49500
        assert status["run_id"] == meta["run_id"]
        milestone = directory / "milestones/epoch-025.pt"
        milestone_receipt = read(milestone.with_suffix(".json"))
        sha = v.j.file_sha(milestone)
        assert milestone_receipt["sha256"] == sha
        assert milestone_receipt["run_id"] == meta["run_id"]
        assert v.j.file_sha(directory / "latest.pt") == sha
        checkpoints[arm] = sha
    return {"screen_id": screen_id, "summary": str(summary_path),
            "summary_sha": v.j.file_sha(summary_path), "checkpoints": checkpoints,
            "results": {k: x["scores"] for k, x in summary["results"].items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    lock = (LOGS / "ijepa_v2_shutdown.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = LOGS / "ijepa_v2_shutdown_status.json"
    try:
        if args.check_only:
            print(json.dumps(verify(), indent=2))
            return
        last_update = 0.0
        while True:
            parallel_path = LOGS / "ijepa_v2_parallel_status.json"
            if parallel_path.exists():
                parallel = read(parallel_path)
                if parallel.get("phase") == "needs_review":
                    raise RuntimeError(f"V2 needs review: {parallel.get('error')}")
                if parallel.get("phase") == "complete":
                    break
            now = time.monotonic()
            if now-last_update >= 300:
                v.j.write_json(state_path, {"status": "waiting", "utc": v.j.utc(),
                    "user_requested": True, "shutdown_only_after_verified_success": True})
                last_update = now
            time.sleep(20)
        verified = verify()
        receipt = {"status": "requested", "utc": v.j.utc(), "user_requested": True,
                   "verification": verified}
        v.j.write_json(state_path, receipt)
        v.j.write_json(ROOT / "diagnostics" / f"ijepa-v2-screen-{verified['screen_id'][:12]}" /
                       "shutdown.json", receipt)
        os.sync()
        subprocess.run(["/bin/bash", "/usr/bin/shutdown"], check=True)
    except BaseException as exc:
        v.j.write_json(state_path, {"status": "needs_review", "utc": v.j.utc(),
            "user_requested": True, "server_preserved": True,
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
