"""Bootstrap the raw Telco-Customer-Churn dataset.

Use this on a fresh clone when no DVC remote is configured (the default
project state). Once a DVC remote is wired up, prefer `dvc pull`.

Run from the repo root:
    python scripts/download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlopen

URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
DEST = Path("data/raw/Telco-Customer-Churn.csv")


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"already present: {DEST} ({DEST.stat().st_size} bytes)")
        return

    print(f"downloading {URL} -> {DEST}")
    with urlopen(URL) as resp:
        body = resp.read()
    DEST.write_bytes(body)

    if not DEST.exists() or DEST.stat().st_size < 100_000:
        sys.exit(f"download appears incomplete: {DEST}")
    print(f"done: {DEST} ({DEST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
