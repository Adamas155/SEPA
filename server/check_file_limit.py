"""Reproduce shared-label FD pressure and verify the server runtime limit fix."""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys


def worker(mode):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from run_long100 import configure_file_limit

    torch.set_num_threads(2)
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard))
    if mode == "raised":
        configure_file_limit()

    class Records(Dataset):
        def __len__(self):
            return 4096

        def __getitem__(self, index):
            # Multiple storages in flight, matching a multi-tensor image batch.
            return {"label": index, **{f"tensor_{i}": torch.ones(2) for i in range(8)}}

    labels = []
    before = len(os.listdir('/proc/self/fd'))
    peak = before
    error = None
    loader = DataLoader(Records(), batch_size=1, num_workers=4)
    iterator = iter(loader)
    try:
        for batch in iterator:
            labels.append(batch['label'])
            if len(labels) % 128 == 0:
                peak = max(peak, len(os.listdir('/proc/self/fd')))
        actual = torch.cat(labels)
        assert torch.equal(actual, torch.arange(4096))
    except (RuntimeError, OSError, EOFError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        count = len(labels)
        labels.clear()
        iterator._shutdown_workers()
        del iterator, loader
        gc.collect()
    after = len(os.listdir('/proc/self/fd'))
    result = {"mode": mode, "batches": count, "fd_before": before,
              "fd_peak_sampled": peak, "fd_after": after, "error": error,
              "limit": resource.getrlimit(resource.RLIMIT_NOFILE)}
    print(json.dumps(result), flush=True)
    if mode == "baseline":
        # FD exhaustion may surface either in PyTorch's explicit limit check
        # or in the IPC receive path when no descriptor arrives with ancdata.
        signatures = ('too many open files', 'received 0 items of ancdata')
        assert error and any(text in error.lower() for text in signatures), result
        assert 900 <= count <= 1024, result
    else:
        assert error is None and count == 4096 and peak > 1024 and after < 128, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['baseline', 'raised'])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    results = []
    for mode in ('baseline', 'raised'):
        run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', mode],
                             capture_output=True, text=True, timeout=90)
        if run.returncode:
            raise RuntimeError(f"{mode} failed:\n{run.stdout}\n{run.stderr[-2000:]}")
        result = json.loads(run.stdout.strip().splitlines()[-1])
        results.append(result)
        print(json.dumps(result), flush=True)
    print(json.dumps({'passed': True, 'results': results}), flush=True)


if __name__ == '__main__':
    main()
