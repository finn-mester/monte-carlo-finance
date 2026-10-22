"""Run the whole project in order: data download, then Parts 1 to 4.

Run from anywhere (paths are resolved from this file):

    python scripts/run_all.py

The download is skipped if data/prices.csv already exists. Each script runs
with the same Python interpreter, and the first failure stops everything.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["run_part1.py", "run_part2.py", "run_part3.py", "run_part4.py"]


def run(script: str) -> None:
    print(f"\n=== {script} ===", flush=True)
    start = time.perf_counter()
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n{script} failed with exit code {result.returncode}; stopping.", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"=== {script} done in {time.perf_counter() - start:.0f} s ===", flush=True)


def main() -> None:
    if (ROOT / "data" / "prices.csv").exists():
        print("data/prices.csv exists: skipping the download.")
    else:
        run("download_data.py")
    for script in SCRIPTS:
        run(script)


if __name__ == "__main__":
    main()
