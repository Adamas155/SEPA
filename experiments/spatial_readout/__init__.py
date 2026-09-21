"""Independent, frozen-encoder spatial readout experiment."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "server"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
