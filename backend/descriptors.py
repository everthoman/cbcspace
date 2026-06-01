"""RDKit molecular descriptor computation.

Parallel, chunked computation designed to scale to multi-million-row libraries.
Invalid SMILES yield NaN rows (dropped by the caller). Results are meant to be
cached to parquet so they are computed only once per dataset.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

import numpy as np

# ── Descriptor registry ──────────────────────────────────────────────────────
# key -> human label, in display order. The compute functions are built in the
# worker (see _descriptor_funcs) in this exact same order.
DESCRIPTOR_LABELS: dict[str, str] = {
    "MW":             "Molecular Weight",
    "ExactMW":        "Exact Molecular Weight",
    "cLogP":          "Calculated LogP",
    "tPSA":           "Topological PSA",
    "HBD":            "H-Bond Donors",
    "HBA":            "H-Bond Acceptors",
    "RotBonds":       "Rotatable Bonds",
    "AromaticRings":  "Aromatic Rings",
    "Rings":          "Total Rings",
    "HeavyAtoms":     "Heavy Atom Count",
    "FractionCSP3":   "Fraction C sp3",
    "Stereocenters":  "Stereocenters",
    "RingFusionDeg":  "Ring Fusion Degree",
    "BertzCT":        "Bertz Complexity (CT)",
    "QED":            "Drug-likeness (QED)",
    "MolRefract":     "Molar Refractivity",
    "NumHeteroatoms": "Heteroatom Count",
    "NumAmideBonds":  "Amide Bonds",
    "NumBridgeheads": "Bridgehead Atoms",
    "NumSpiro":       "Spiro Atoms",
    "MaxRingSize":    "Max Ring Size",
}
DESCRIPTOR_KEYS = list(DESCRIPTOR_LABELS.keys())
N_DESC = len(DESCRIPTOR_KEYS)


def _descriptor_funcs():
    """Build the ordered list of (key, fn) inside the worker process.

    Order MUST match DESCRIPTOR_LABELS so column i corresponds to key i.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, Lipinski, QED, rdMolDescriptors

    def stereocenters(m):
        return len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))

    def ring_fusion_deg(m):
        # atoms shared by >= 2 rings (degree of ring fusion)
        ri = m.GetRingInfo()
        return sum(1 for a in range(m.GetNumAtoms()) if ri.NumAtomRings(a) >= 2)

    def max_ring_size(m):
        ri = m.GetRingInfo()
        return max((len(r) for r in ri.AtomRings()), default=0)

    return [
        ("MW",             Descriptors.MolWt),
        ("ExactMW",        Descriptors.ExactMolWt),
        ("cLogP",          Crippen.MolLogP),
        ("tPSA",           Descriptors.TPSA),
        ("HBD",            Lipinski.NumHDonors),
        ("HBA",            Lipinski.NumHAcceptors),
        ("RotBonds",       Descriptors.NumRotatableBonds),
        ("AromaticRings",  rdMolDescriptors.CalcNumAromaticRings),
        ("Rings",          rdMolDescriptors.CalcNumRings),
        ("HeavyAtoms",     lambda m: m.GetNumHeavyAtoms()),
        ("FractionCSP3",   rdMolDescriptors.CalcFractionCSP3),
        ("Stereocenters",  stereocenters),
        ("RingFusionDeg",  ring_fusion_deg),
        ("BertzCT",        GraphDescriptors.BertzCT),
        ("QED",            QED.qed),
        ("MolRefract",     Crippen.MolMR),
        ("NumHeteroatoms", rdMolDescriptors.CalcNumHeteroatoms),
        ("NumAmideBonds",  rdMolDescriptors.CalcNumAmideBonds),
        ("NumBridgeheads", rdMolDescriptors.CalcNumBridgeheadAtoms),
        ("NumSpiro",       rdMolDescriptors.CalcNumSpiroAtoms),
        ("MaxRingSize",    max_ring_size),
    ]


def _compute_chunk(smiles_chunk: list[str]) -> np.ndarray:
    """Worker: compute the full descriptor matrix for a chunk of SMILES."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    funcs = _descriptor_funcs()
    out = np.full((len(smiles_chunk), len(funcs)), np.nan, dtype=np.float32)
    for i, smi in enumerate(smiles_chunk):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for j, (_, fn) in enumerate(funcs):
            try:
                out[i, j] = fn(mol)
            except Exception:
                out[i, j] = np.nan
    return out


# ── Fingerprints (ECFP4 / Morgan radius 2, 2048-bit) ─────────────────────────
FP_RADIUS = 2
FP_BITS = 2048
FP_BYTES = FP_BITS // 8  # packed width


def _fp_chunk(smiles_chunk: list[str]) -> np.ndarray:
    """Worker: packed ECFP4 bits for a chunk -> (len, FP_BYTES) uint8.
    Invalid SMILES yield an all-zero row (dropped by the caller's descriptor mask)."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    out = np.zeros((len(smiles_chunk), FP_BYTES), dtype=np.uint8)
    bits = np.zeros((FP_BITS,), dtype=np.uint8)
    for i, smi in enumerate(smiles_chunk):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), bits)
        out[i] = np.packbits(bits)
    return out


def compute_fingerprints(
    smiles: list[str],
    *,
    n_workers: int | None = None,
    chunk_size: int = 5000,
) -> np.ndarray:
    """Return packed ECFP4 fingerprints, (n, FP_BYTES) uint8.

    Packed (256 bytes/mol) so multi-million-row libraries stay compact on disk;
    callers unpack in chunks for projection.
    """
    n = len(smiles)
    if n == 0:
        return np.empty((0, FP_BYTES), dtype=np.uint8)
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1))
    chunks = list(_chunked(smiles, chunk_size))
    if n_workers == 1 or len(chunks) == 1:
        return np.vstack([_fp_chunk(c) for c in chunks])
    results: list[np.ndarray] = [None] * len(chunks)  # type: ignore
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_fp_chunk, c): idx for idx, c in enumerate(chunks)}
        for fut in futures:
            results[futures[fut]] = fut.result()
    return np.vstack(results)


def _chunked(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def compute_descriptors(
    smiles: list[str],
    *,
    n_workers: int | None = None,
    chunk_size: int = 5000,
) -> np.ndarray:
    """Return an (n, N_DESC) float32 array of descriptors for `smiles`.

    Parallelised across processes in chunks so memory stays bounded and large
    libraries (millions of rows) compute in a streaming fashion.
    """
    n = len(smiles)
    if n == 0:
        return np.empty((0, N_DESC), dtype=np.float32)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1))

    chunks = list(_chunked(smiles, chunk_size))

    # Single-process path: avoids pool overhead for small inputs and works even
    # where process spawning is restricted.
    if n_workers == 1 or len(chunks) == 1:
        return np.vstack([_compute_chunk(c) for c in chunks])

    results: list[np.ndarray] = [None] * len(chunks)  # type: ignore
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_compute_chunk, c): idx for idx, c in enumerate(chunks)}
        for fut in futures:
            results[futures[fut]] = fut.result()
    return np.vstack(results)
