# CBCSpace (Python)

A Python replica of [sculpturatus.com/chemscape](https://sculpturatus.com/chemscape/):
a cheminformatics tool for projecting chemical libraries into 2D/3D space
(PCA, UMAP, t-SNE, or custom descriptor axes) and comparing their coverage.
Built to scale to **multi-million-compound libraries** on a single workstation.

The projection basis is selectable via a **feature-space toggle**:
- **Descriptors** — a 21-descriptor physicochemical panel (standardized).
- **Fingerprint** — ECFP4 (Morgan radius 2, 2048-bit). UMAP/t-SNE use the
  **Jaccard/Tanimoto** metric; PCA runs on the bit matrix. Custom axes are
  descriptor-only. Fingerprints are computed and stored (packed, 256 bytes/mol)
  at upload, alongside the descriptors.
  - Fingerprint t-SNE offers a **PCA pre-reduce to 50D** toggle (on by default):
    reduces the bits to 50 PCA components, then runs Euclidean t-SNE. Much faster
    than Tanimoto-on-raw-bits at scale (`engine.TSNE_PCA_DIMS`). Untick it to run
    true Tanimoto t-SNE.

## Run

```bash
./run.sh
# open http://127.0.0.1:8077
```

Three default libraries are bundled. Drop a CSV / TSV / Excel file (with a
`smiles` column, optional `name`/`id` column) to add your own.

## Architecture

```
backend/
  app.py           FastAPI: /session/new /upload /project /overlap /molecule/image, DELETE set
  descriptors.py   RDKit descriptor computation (parallel, chunked)
  engine.py        PCA / IncrementalPCA / UMAP / custom-axis projection + overlap stats
  store.py         Disk-backed sessions; datasets stored columnar as parquet
  make_defaults.py one-off generator for the bundled libraries
  data/defaults/   bundled libraries (parquet)
  sessions/        per-session uploads (parquet), created at runtime
frontend/
  index.html       rebuilt Plotly UI (WebGL scattergl for 2D, scatter3d for 3D)
```

## How it scales to large libraries

| Concern | Approach |
|---|---|
| Descriptor computation | RDKit, parallelised across processes in 5k-row chunks (`descriptors.py`) |
| Fingerprints | ECFP4 computed in parallel at upload, stored packed (256 bytes/mol); unpacked in 20k-row chunks for projection so the full bit matrix never sits in RAM (`engine.FP_BATCH`) |
| Memory | Datasets stored columnar in **parquet**; projections read only the selected descriptor columns from disk |
| PCA | `IncrementalPCA` with chunked `partial_fit` above 200k rows (`engine.INCREMENTAL_PCA_THRESHOLD`) |
| UMAP | Fitted on a bounded 50k sample, then `.transform()` of the full set (`engine.UMAP_FIT_CAP`) |
| t-SNE | `openTSNE` (FFT-accelerated in 2D, Barnes-Hut in 3D); fitted on a bounded 50k sample, then `.transform()` of the full set (`engine.TSNE_FIT_CAP`) |
| Browser rendering | Coordinates are computed on **all** rows, then stratified-downsampled per set to 150k points for the plot (`engine.DISPLAY_CAP`); the UI shows "showing N of M (sampled)". Overlap/spread stats always use the full data. |

Tuning knobs live at the top of `backend/engine.py`.

## Descriptors

MW, cLogP, TPSA, H-bond donors/acceptors, rotatable bonds, ring count,
aromatic rings, fraction Csp3, heavy atoms, QED, formal charge.

## Notes / deviations from the original

- The original's exact default libraries and descriptor list aren't published,
  so libraries are generated combinatorially (`make_defaults.py`) and the
  descriptor set is a standard medicinal-chemistry panel.
- 2D plots use WebGL (`scattergl`) so large point clouds stay interactive.
- Added a visible "showing N of M (sampled)" indicator when downsampling is active.
