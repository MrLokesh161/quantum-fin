"""Refresh free public data, then rebuild the local next-session risk estimate."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh public market data and retrain QETI locally."
    )
    parser.add_argument(
        "--start",
        default="1990-01-01",
        help="Inclusive history start date (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    commands = [
        [sys.executable, "-m", "src.ingestion", "--start", args.start],
        [sys.executable, "main.py"],
    ]
    for label, command in zip(("Public-data refresh", "Model and next-session estimate"), commands):
        print(f"\n=== {label} ===", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            print(f"{label} failed with exit code {result.returncode}.", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
