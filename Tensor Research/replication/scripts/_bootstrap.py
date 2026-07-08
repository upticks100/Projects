"""Make the replication package importable when a script is run directly
(`python scripts/02_build_caches.py`) instead of via `python -m`."""
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
