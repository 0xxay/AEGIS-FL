"""
Qelvari — Privacy-Preserving Federated Intrusion Detection System.
"""

import os
import shutil
from pathlib import Path

__version__ = "2.5.0"

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def is_serverless_env() -> bool:
    return bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))


def get_storage_root() -> Path:
    """
    Returns writable runtime root for `data/` and `results/`.
    Locally and in Docker: uses repository root.
    On Vercel Functions (`/var/task` read-only): uses `/tmp/qelvari` and seeds
    baseline CSV/audit artifacts from bundled `results/` on cold start.
    """
    if not is_serverless_env() and os.access(PACKAGE_ROOT, os.W_OK):
        return PACKAGE_ROOT

    tmp_root = Path("/tmp/qelvari")
    tmp_results = tmp_root / "results"
    if not tmp_results.exists():
        tmp_results.mkdir(parents=True, exist_ok=True)
        bundled_results = PACKAGE_ROOT / "results"
        if bundled_results.exists():
            for item in bundled_results.glob("*"):
                if item.is_file():
                    shutil.copy2(item, tmp_results / item.name)
    return tmp_root
