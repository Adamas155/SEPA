"""Run without installing: python run.py --help."""

import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from sepa_plan_b.commands import main

if __name__ == "__main__":
    raise SystemExit(main())
