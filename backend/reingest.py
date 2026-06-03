"""Re-standardize and recompute existing default libraries in place.

Existing parquets were ingested before structure standardization existed, so
their descriptors/fingerprints (and stored SMILES) still include salts and
charges. This reads each library's name + SMILES and re-runs the full ingest
(standardize -> descriptors -> fingerprints), overwriting the parquet.

Usage:
    python reingest.py --all                 # every library in data/defaults/
    python reingest.py "5K Fragments" "CBCS Primary Screening Set 50K" ...

Pass library names (the parquet stem, i.e. the name shown in the UI).
"""
from __future__ import annotations

import sys

import pandas as pd

import store


def _reingest(path) -> int:
    df = pd.read_parquet(path, columns=["name", "smiles"])
    n = store.ingest_dataframe(df, path.stem, path)
    print(f"  {path.stem}: {n} molecules re-standardized -> {path.name}")
    return n


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    if argv == ["--all"]:
        paths = sorted(store.DEFAULTS_DIR.glob("*.parquet"))
    else:
        paths = [store.DEFAULTS_DIR / f"{name}.parquet" for name in argv]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("error: not found: " + ", ".join(p.stem for p in missing))
        return 1
    print(f"re-ingesting {len(paths)} librar{'y' if len(paths)==1 else 'ies'}…")
    for p in paths:
        _reingest(p)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
