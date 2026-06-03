"""RDKit structure standardization (full MolStandardize pipeline).

Applied once at ingest, before descriptors/fingerprints, so salts/solvents and
charge/representation noise don't distort chemical space:

    Cleanup  ->  FragmentParent (strip salts/solvents, keep parent)  ->  Uncharge

Returns the canonical parent SMILES, or "" when the input can't be parsed
(callers drop empty/NaN rows). Parallel + chunked to match the descriptor and
fingerprint compute paths.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable


def _standardize_chunk(smiles_chunk: list[str]) -> list[str]:
    """Worker: full standardization for a chunk of SMILES -> parent SMILES."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    uncharger = rdMolStandardize.Uncharger()
    out: list[str] = []
    for smi in smiles_chunk:
        if not smi:
            out.append("")
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append("")
            continue
        try:
            mol = rdMolStandardize.Cleanup(mol)
            mol = rdMolStandardize.FragmentParent(mol)   # drop salts/solvents
            mol = uncharger.uncharge(mol)                # neutralize
            out.append(Chem.MolToSmiles(mol))
        except Exception:
            out.append(smi)  # keep the original if standardization fails
    return out


def _chunked(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def standardize_smiles(
    smiles: list[str],
    *,
    n_workers: int | None = None,
    chunk_size: int = 5000,
) -> list[str]:
    """Return the standardized parent SMILES for each input, same length/order."""
    n = len(smiles)
    if n == 0:
        return []
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1))
    chunks = list(_chunked(smiles, chunk_size))
    if n_workers == 1 or len(chunks) == 1:
        out: list[str] = []
        for c in chunks:
            out.extend(_standardize_chunk(c))
        return out
    results: list[list[str]] = [None] * len(chunks)  # type: ignore
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_standardize_chunk, c): idx for idx, c in enumerate(chunks)}
        for fut in futures:
            results[futures[fut]] = fut.result()
    out = []
    for r in results:
        out.extend(r)
    return out
