"""Projection + overlap engine.

Scales to multi-million-row inputs via:
  - IncrementalPCA (chunked partial_fit) above a row threshold,
  - UMAP fit on a bounded sample then .transform() of the full set,
  - stratified display downsampling so the browser receives a bounded number of
    points even when the underlying library is huge (overlap stats still use
    the full data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler

from descriptors import DESCRIPTOR_LABELS, FP_BITS
from store import load_columns, load_fingerprints

# Thresholds (tuned for a single workstation, ~5M rows max).
INCREMENTAL_PCA_THRESHOLD = 200_000   # rows above which IncrementalPCA is used
UMAP_FIT_CAP = 50_000                 # rows UMAP is fitted on; rest transformed
TSNE_FIT_CAP = 50_000                 # rows t-SNE is fitted on; rest transformed
DISPLAY_CAP = 150_000                 # max points streamed to the browser
FP_BATCH = 20_000                     # rows per chunk when unpacking fingerprints
RNG = np.random.default_rng(0)


def _load_matrix(sid: str, set_names: list[str], desc_cols: list[str]):
    """Concatenate selected descriptor columns (+ name/smiles/label) across sets."""
    frames = []
    for name in set_names:
        df = load_columns(sid, name, ["name", "smiles", *desc_cols])
        df["__label"] = name
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    X = full[desc_cols].to_numpy(dtype=np.float32)
    ok = np.isfinite(X).all(axis=1)
    return full.loc[ok].reset_index(drop=True), X[ok]


def _load_fp_matrix(sid: str, set_names: list[str]):
    """Concatenate packed ECFP4 fingerprints (+ name/smiles/label) across sets."""
    metas, packs = [], []
    for name in set_names:
        meta, packed = load_fingerprints(sid, name)
        meta = meta.copy()
        meta["__label"] = name
        metas.append(meta)
        packs.append(packed)
    full = pd.concat(metas, ignore_index=True)
    return full, np.vstack(packs)


def _unpack(packed_slice: np.ndarray) -> np.ndarray:
    """Packed bytes (k, FP_BYTES) -> float32 bit matrix (k, FP_BITS)."""
    return np.unpackbits(packed_slice, axis=1).astype(np.float32)


def _fit_pca_fp(packed: np.ndarray, n: int):
    nrows = len(packed)
    if nrows > INCREMENTAL_PCA_THRESHOLD:
        ipca = IncrementalPCA(n_components=n)
        for i in range(0, nrows, FP_BATCH):
            ipca.partial_fit(_unpack(packed[i:i + FP_BATCH]))
        coords = np.vstack([ipca.transform(_unpack(packed[i:i + FP_BATCH]))
                            for i in range(0, nrows, FP_BATCH)])
        evr = ipca.explained_variance_ratio_
    else:
        pca = PCA(n_components=n)
        coords = pca.fit_transform(_unpack(packed))
        evr = pca.explained_variance_ratio_
    return coords, (evr * 100.0).tolist()


def _fit_umap_fp(packed: np.ndarray, n: int):
    import umap
    reducer = umap.UMAP(n_components=n, n_neighbors=15, min_dist=0.1,
                        metric="jaccard", random_state=42)  # Tanimoto on bit vectors
    nrows = len(packed)
    if nrows > UMAP_FIT_CAP:
        idx = RNG.choice(nrows, UMAP_FIT_CAP, replace=False)
        reducer.fit(_unpack(packed[idx]))
        coords = np.vstack([reducer.transform(_unpack(packed[i:i + FP_BATCH]))
                            for i in range(0, nrows, FP_BATCH)])
    else:
        coords = reducer.fit_transform(_unpack(packed))
    return np.asarray(coords)


TSNE_PCA_DIMS = 50   # target dims when PCA pre-reducing fingerprints before t-SNE


def _fp_to_pca(packed: np.ndarray, dims: int) -> np.ndarray:
    """Reduce packed fingerprints to `dims` PCA components (chunked, dense float32)."""
    nrows = len(packed)
    dims = min(dims, FP_BITS, nrows)
    if nrows > INCREMENTAL_PCA_THRESHOLD:
        ipca = IncrementalPCA(n_components=dims)
        for i in range(0, nrows, FP_BATCH):
            ipca.partial_fit(_unpack(packed[i:i + FP_BATCH]))
        return np.vstack([ipca.transform(_unpack(packed[i:i + FP_BATCH]))
                          for i in range(0, nrows, FP_BATCH)]).astype(np.float32)
    return PCA(n_components=dims).fit_transform(_unpack(packed)).astype(np.float32)


def _fit_tsne_fp(packed: np.ndarray, n: int, pca_prereduce: bool = False):
    # Fast path: PCA-reduce bits to ~50D, then Euclidean t-SNE on the dense matrix.
    if pca_prereduce:
        return _fit_tsne(_fp_to_pca(packed, TSNE_PCA_DIMS), n)
    # Default: Tanimoto/Jaccard t-SNE directly on the bit vectors.
    from openTSNE import TSNE
    grad = "interpolation" if n <= 2 else "bh"
    tsne = TSNE(n_components=n, perplexity=30, metric="jaccard", neighbors="approx",
                negative_gradient_method=grad, random_state=42, n_jobs=-1)
    nrows = len(packed)
    if nrows > TSNE_FIT_CAP:
        idx = RNG.choice(nrows, TSNE_FIT_CAP, replace=False)
        emb = tsne.fit(_unpack(packed[idx]))
        coords = np.vstack([emb.transform(_unpack(packed[i:i + FP_BATCH]))
                            for i in range(0, nrows, FP_BATCH)])
    else:
        coords = tsne.fit(_unpack(packed))
    return np.asarray(coords)


def _fp_centroid_spread(packed: np.ndarray):
    """Mean bit-frequency centroid and mean euclidean spread, computed in chunks."""
    m = len(packed)
    s = np.zeros(FP_BITS, dtype=np.float64)
    for i in range(0, m, FP_BATCH):
        s += _unpack(packed[i:i + FP_BATCH]).sum(axis=0)
    c = s / m
    ss = 0.0
    for i in range(0, m, FP_BATCH):
        d = _unpack(packed[i:i + FP_BATCH]) - c
        ss += float((d * d).sum())
    return c, float(np.sqrt(ss / m))


def _fit_pca(Xs: np.ndarray, n: int):
    if Xs.shape[0] > INCREMENTAL_PCA_THRESHOLD:
        ipca = IncrementalPCA(n_components=n)
        batch = 50_000
        for i in range(0, Xs.shape[0], batch):
            ipca.partial_fit(Xs[i:i + batch])
        coords = np.vstack([ipca.transform(Xs[i:i + batch])
                            for i in range(0, Xs.shape[0], batch)])
        evr = ipca.explained_variance_ratio_
    else:
        pca = PCA(n_components=n)
        coords = pca.fit_transform(Xs)
        evr = pca.explained_variance_ratio_
    return coords, (evr * 100.0).tolist()


def _fit_umap(Xs: np.ndarray, n: int):
    import umap
    reducer = umap.UMAP(n_components=n, n_neighbors=15, min_dist=0.1, random_state=42)
    if Xs.shape[0] > UMAP_FIT_CAP:
        idx = RNG.choice(Xs.shape[0], UMAP_FIT_CAP, replace=False)
        reducer.fit(Xs[idx])
        coords = reducer.transform(Xs)
    else:
        coords = reducer.fit_transform(Xs)
    return np.asarray(coords)


def _fit_tsne(Xs: np.ndarray, n: int):
    from openTSNE import TSNE
    # FFT acceleration only supports 2D; Barnes-Hut handles up to 3D.
    grad = "interpolation" if n <= 2 else "bh"
    tsne = TSNE(n_components=n, perplexity=30, metric="euclidean",
                negative_gradient_method=grad, random_state=42, n_jobs=-1)
    if Xs.shape[0] > TSNE_FIT_CAP:
        idx = RNG.choice(Xs.shape[0], TSNE_FIT_CAP, replace=False)
        emb = tsne.fit(Xs[idx])      # openTSNE embeddings support out-of-sample
        coords = emb.transform(Xs)   # transform the remaining rows
    else:
        coords = tsne.fit(Xs)
    return np.asarray(coords)


def _downsample(n_rows: int, labels: np.ndarray) -> np.ndarray:
    """Return row indices to display, stratified per set, capped at DISPLAY_CAP."""
    if n_rows <= DISPLAY_CAP:
        return np.arange(n_rows)
    keep = []
    uniq, counts = np.unique(labels, return_counts=True)
    for lab, cnt in zip(uniq, counts):
        quota = max(1, int(round(DISPLAY_CAP * cnt / n_rows)))
        idx = np.where(labels == lab)[0]
        keep.append(idx if cnt <= quota else RNG.choice(idx, quota, replace=False))
    return np.sort(np.concatenate(keep))


def project(sid, set_names, desc_cols, method, n_dimensions,
            x_col=None, y_col=None, z_col=None, feature="descriptors",
            tsne_pca_reduce=False):
    if feature == "fingerprint":
        full, packed = _load_fp_matrix(sid, set_names)
        if method == "UMAP":
            coords = _fit_umap_fp(packed, n_dimensions)
            evr = None
        elif method == "tSNE":
            coords = _fit_tsne_fp(packed, n_dimensions, pca_prereduce=tsne_pca_reduce)
            evr = None
        else:  # PCA (Custom axes don't apply to fingerprints)
            coords, evr = _fit_pca_fp(packed, n_dimensions)
        axis_labels = None
    elif method == "Custom":
        if not desc_cols and (x_col is None or y_col is None):
            raise ValueError("select custom axes")
        cols = [x_col, y_col] + ([z_col] if n_dimensions == 3 else [])
        full, _ = _load_matrix(sid, set_names, list(dict.fromkeys(cols)))
        coords = full[cols].to_numpy(dtype=np.float32)
        evr = None
        axis_labels = [f"{c} · {DESCRIPTOR_LABELS.get(c, c)}" for c in cols]
    else:
        if not desc_cols:
            raise ValueError("select at least one descriptor")
        full, X = _load_matrix(sid, set_names, desc_cols)
        Xs = StandardScaler().fit_transform(X)
        if method == "UMAP":
            coords = _fit_umap(Xs, n_dimensions)
            evr = None
        elif method == "tSNE":
            coords = _fit_tsne(Xs, n_dimensions)
            evr = None
        else:  # PCA
            coords, evr = _fit_pca(Xs, n_dimensions)
        axis_labels = None

    labels = full["__label"].to_numpy()
    keep = _downsample(len(full), labels)

    c = coords[keep]
    return {
        "coords": c.tolist(),
        "labels": labels[keep].tolist(),
        "names": full["name"].to_numpy()[keep].tolist(),
        "smiles": full["smiles"].to_numpy()[keep].tolist(),
        "method": method,
        "explained_variance": evr,
        "axis_labels": axis_labels,
        "n_total": int(len(full)),
        "n_displayed": int(len(keep)),
    }


def overlap(sid, set_names, desc_cols, feature="descriptors"):
    """Centroid distances between sets + per-set spread (full data, independent
    of the display cap). Descriptor mode uses standardized descriptor space;
    fingerprint mode uses bit-frequency (Tanimoto-style) space."""
    centroids, spread = {}, {}
    if feature == "fingerprint":
        for name in set_names:
            _, packed = load_fingerprints(sid, name)
            if len(packed) == 0:
                continue
            centroids[name], spread[name] = _fp_centroid_spread(packed)
    else:
        if not desc_cols:
            raise ValueError("select at least one descriptor")
        full, X = _load_matrix(sid, set_names, desc_cols)
        Xs = StandardScaler().fit_transform(X)
        labels = full["__label"].to_numpy()
        for name in set_names:
            pts = Xs[labels == name]
            if len(pts) == 0:
                continue
            c = pts.mean(axis=0)
            centroids[name] = c
            spread[name] = float(np.sqrt(((pts - c) ** 2).sum(axis=1)).mean())

    names = list(centroids)
    distances = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = float(np.linalg.norm(centroids[names[i]] - centroids[names[j]]))
            distances[f"{names[i]} ↔ {names[j]}"] = d
    return {"centroid_distances": distances, "spread": spread}
