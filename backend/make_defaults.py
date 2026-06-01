"""Generate the bundled default libraries.

Builds three chemically-distinct libraries by combinatorial enumeration so they
occupy different regions of descriptor space (making overlap/spread meaningful).
Invalid SMILES are dropped during ingest. Run once: `python make_defaults.py`.
"""
from __future__ import annotations

import itertools

import pandas as pd
from rdkit import Chem, RDLogger

import store

RDLogger.DisableLog("rdApp.*")

AROM_CORES = ["c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1",
              "c1ccc2ncccc2c1", "c1ccsc1", "c1cc2ccccc2o1", "c1ccc(cc1)c1ccccc1"]
ALIPH_CORES = ["C1CCCCC1", "C1CCCC1", "C1CCCCCC1", "C1CCC2CCCCC2C1",
               "CCCCCC", "CCCCCCCC", "C1CCNCC1", "C1CCOCC1"]
LINKERS = ["", "C(=O)N", "NC(=O)", "OC", "S", "CC", "C=C", "OCC"]
AROM_TAILS = ["c1ccccc1", "c1ccncc1", "c1ccc(F)cc1", "c1ccc(Cl)cc1",
              "c1ccc(C)cc1", "c1ccc(OC)cc1", "C(F)(F)F", "c1cccs1"]
ALIPH_TAILS = ["C", "CC", "CCC", "CCCC", "CCCCC", "C(C)C", "CC(C)C", "C1CCCCC1"]
POLAR_TAILS = ["O", "N", "C(=O)O", "C(=O)N", "S(=O)(=O)N", "OCCO",
               "NCCO", "C(=O)NCCO", "NC(=N)N", "OCCN"]


def _enumerate(cores, linkers, tails, limit):
    seen, out = set(), []
    for core, link, tail in itertools.product(cores, linkers, tails):
        smi = core + link + tail
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen:
            continue
        seen.add(canon)
        out.append(canon)
        if len(out) >= limit:
            break
    return out


def build(name: str, smiles: list[str]):
    df = pd.DataFrame({"smiles": smiles, "name": [f"{name}-{i+1}" for i in range(len(smiles))]})
    dest = store.DEFAULTS_DIR / f"{name}.parquet"
    n = store.ingest_dataframe(df, name, dest)
    print(f"  {name}: {n} molecules -> {dest.name}")


if __name__ == "__main__":
    print("generating default libraries…")
    build("aromatic-leads", _enumerate(AROM_CORES, LINKERS, AROM_TAILS, 600))
    build("aliphatic-scaffolds", _enumerate(ALIPH_CORES, LINKERS, ALIPH_TAILS, 400))
    build("polar-fragments", _enumerate(AROM_CORES + ALIPH_CORES, ["", "C", "CC"], POLAR_TAILS, 400))
    print("done.")
