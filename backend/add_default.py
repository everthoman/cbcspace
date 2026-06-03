"""Add (or replace) a bundled default library from a structure file.

Default libraries are just parquet files in data/defaults/; the filename stem
is the name shown in the UI. This computes descriptors + fingerprints via the
same ingest path as the GUI upload, then writes the parquet into that folder.

Usage:
    python add_default.py <library-name> <path/to/compounds.csv>

The input may be CSV/TSV/XLSX with a SMILES column (headers smiles/smi/
canonical_smiles/structure are auto-detected) and an optional name column
(name/id/title/molecule/compound/chembl_id/zinc_id). Invalid rows are dropped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import store


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    raw_name, src = argv
    name = store.safe_set_name(raw_name)
    path = Path(src)
    if not path.exists():
        print(f"error: file not found: {path}")
        return 1

    df = store.read_table(path)
    dest = store.DEFAULTS_DIR / f"{name}.parquet"
    existed = dest.exists()
    n = store.ingest_dataframe(df, name, dest)
    if n == 0:
        dest.unlink(missing_ok=True)
        print("error: no valid molecules found")
        return 1

    verb = "replaced" if existed else "added"
    print(f"{verb} default library '{name}': {n} molecules -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
