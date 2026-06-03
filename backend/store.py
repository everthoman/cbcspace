"""Disk-backed session + dataset store.

Each session is a directory under SESSIONS_DIR. Each dataset ("set") within a
session is a parquet file holding: name, smiles, and one column per descriptor.
Parquet keeps descriptors columnar so projections read only the columns they
need, and lets datasets of millions of rows live on disk instead of in RAM.
"""
from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from descriptors import DESCRIPTOR_KEYS, FP_BYTES, compute_descriptors, compute_fingerprints

BASE = Path(__file__).resolve().parent
SESSIONS_DIR = BASE / "sessions"
DEFAULTS_DIR = BASE / "data" / "defaults"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULTS_DIR.mkdir(parents=True, exist_ok=True)

_SMILES_HINTS = ("smiles", "smi", "canonical_smiles", "structure")
_NAME_HINTS = ("name", "id", "title", "molecule", "compound", "compound_id",
               "chembl_id", "zinc_id")
_SAFE = re.compile(r"[^A-Za-z0-9 _.\-]")


def safe_set_name(raw: str) -> str:
    name = _SAFE.sub("", Path(raw).stem).strip() or "dataset"
    return name[:60]


def new_session() -> str:
    sid = uuid.uuid4().hex[:16]
    (SESSIONS_DIR / sid).mkdir(parents=True, exist_ok=True)
    return sid


def session_dir(sid: str) -> Path:
    d = SESSIONS_DIR / sid
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
    return d


def _detect_columns(df: pd.DataFrame) -> tuple[str, str | None]:
    cols = {c.lower(): c for c in df.columns}
    smi_col = next((cols[h] for h in _SMILES_HINTS if h in cols), None)
    if smi_col is None:
        # Fall back to first column whose first non-null value parses as SMILES.
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        for c in df.columns:
            sample = df[c].dropna().astype(str).head(20)
            if len(sample) and sum(Chem.MolFromSmiles(s) is not None for s in sample) >= len(sample) * 0.6:
                smi_col = c
                break
    if smi_col is None:
        raise ValueError("could not find a SMILES column")
    name_col = next((cols[h] for h in _NAME_HINTS if h in cols), None)
    return smi_col, name_col


def _sniff_delimiter(path: Path) -> str:
    """Pick a delimiter from a fixed candidate set by majority count on the
    header line. Avoids csv.Sniffer, which mis-detects SMILES characters (e.g.
    'smiles' with no comma gets split on 's')."""
    with open(path, "r", newline="") as f:
        header = f.readline()
    counts = {d: header.count(d) for d in (",", "\t", ";", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    sep = "\t" if suf == ".tsv" else _sniff_delimiter(path)
    return pd.read_csv(path, sep=sep, engine="c", dtype=str)


def ingest_dataframe(df: pd.DataFrame, set_name: str, dest: Path) -> int:
    """Compute descriptors for `df` and write a parquet dataset. Returns rows kept."""
    smi_col, name_col = _detect_columns(df)
    smiles = df[smi_col].astype(str).fillna("").tolist()
    if name_col is not None:
        names = df[name_col].astype(str).fillna("").tolist()
    else:
        names = [f"{set_name}_{i+1}" for i in range(len(smiles))]

    desc = compute_descriptors(smiles)        # (n, N_DESC) float32
    fp = compute_fingerprints(smiles)         # (n, FP_BYTES) uint8, packed ECFP4
    valid = ~np.isnan(desc).any(axis=1)

    out = pd.DataFrame(desc[valid], columns=DESCRIPTOR_KEYS)
    out.insert(0, "smiles", [s for s, v in zip(smiles, valid) if v])
    out.insert(0, "name", [n for n, v in zip(names, valid) if v])
    out["fp"] = [row.tobytes() for row in fp[valid]]  # packed bits as bytes
    out.to_parquet(dest, index=False)
    return int(valid.sum())


def set_path(sid: str, set_name: str, is_default: bool = False) -> Path:
    if is_default:
        return DEFAULTS_DIR / f"{set_name}.parquet"
    return session_dir(sid) / f"{safe_set_name(set_name)}.parquet"


def resolve_set(sid: str, set_name: str) -> Path | None:
    """A set name may refer to a session upload or a shared default library."""
    p = session_dir(sid) / f"{safe_set_name(set_name)}.parquet"
    if p.exists():
        return p
    d = DEFAULTS_DIR / f"{set_name}.parquet"
    return d if d.exists() else None


def list_defaults() -> dict[str, int]:
    import pyarrow.parquet as pq
    out: dict[str, int] = {}
    for p in sorted(DEFAULTS_DIR.glob("*.parquet")):
        out[p.stem] = pq.ParquetFile(p).metadata.num_rows
    return out


def load_columns(sid: str, set_name: str, cols: list[str]) -> pd.DataFrame:
    p = resolve_set(sid, set_name)
    if p is None:
        raise KeyError(set_name)
    return pd.read_parquet(p, columns=cols)


def load_fingerprints(sid: str, set_name: str):
    """Return (meta_df[name,smiles], packed_fp (n, FP_BYTES) uint8) for one set."""
    p = resolve_set(sid, set_name)
    if p is None:
        raise KeyError(set_name)
    df = pd.read_parquet(p, columns=["name", "smiles", "fp"])
    if "fp" not in df or df["fp"].isna().all():
        raise ValueError(f"'{set_name}' has no fingerprints (re-upload to add them)")
    packed = np.frombuffer(b"".join(df["fp"].values), dtype=np.uint8).reshape(-1, FP_BYTES)
    return df[["name", "smiles"]], packed


def delete_set(sid: str, set_name: str) -> None:
    p = session_dir(sid) / f"{safe_set_name(set_name)}.parquet"
    if p.exists():
        p.unlink()


def cleanup_session(sid: str) -> None:
    d = SESSIONS_DIR / sid
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
