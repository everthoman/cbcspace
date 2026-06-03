"""Projection + overlap engine.

Scales to multi-million-row inputs via:
  - IncrementalPCA (chunked partial_fit) above a row threshold,
  - UMAP fit on a bounded sample then .transform() of the full set,
  - stratified display downsampling so the browser receives a bounded number of
    points even when the underlying library is huge (overlap stats still use
    the full data).
"""
from __future__ import annotations

import os

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
TSNE_FFT_MIN = 10_000                 # below this, Barnes-Hut beats FFT (interpolation)
# openTSNE n_jobs: -1 (all cores) badly oversubscribes on this shared 36-core host
# (~17x slower on small N). A moderate cap is near-optimal and neighbourly.
TSNE_JOBS = min(8, os.cpu_count() or 1)
DISPLAY_CAP = 150_000                 # max points streamed to the browser
FP_BATCH = 20_000                     # rows per chunk when unpacking fingerprints
COVERAGE_QUERY_CAP = 4_000            # query molecules sampled per set for NN coverage
COVERAGE_REF_CAP = 30_000            # reference molecules sampled per set for NN coverage
COVERAGE_Q_CHUNK = 2_048              # query rows per matmul chunk
RNG = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Optional GPU acceleration (RAPIDS cuML).
#
# Enabled with CBCSPACE_GPU=1. cuML accelerates the *euclidean* descriptor
# paths (PCA / UMAP / t-SNE) by ~5-180x at tens of thousands of molecules.
# It is NOT a full drop-in, so the helpers below return None (-> CPU fallback)
# whenever cuML can't match CPU semantics:
#   - no Tanimoto/Jaccard metric        -> fingerprint UMAP/t-SNE stay on CPU
#   - cuML t-SNE is 2D-only             -> 3D t-SNE stays on CPU
#   - tiny N (< GPU_MIN_ROWS)           -> CPU (numerical floor, already instant)
# When enabled, GPU is used for every euclidean fit above the floor (it wins from
# ~1k rows here). Any cuML error also falls back, so a GPU hiccup never fails a
# projection. The CUDA context is warmed at startup (see warmup()).
# ---------------------------------------------------------------------------
GPU_MIN_ROWS = 100                    # safety floor only (cuML needs n_samples > n_neighbors
                                      # /perplexity); GPU wins from ~1k rows on this hardware
_CUML = None                          # cached module after first probe, or False


def _gpu_enabled() -> bool:
    return os.environ.get("CBCSPACE_GPU", "").lower() in ("1", "true", "yes", "on")


def _cuml():
    """Lazily import cuML once; cache the module, or False if unavailable."""
    global _CUML
    if _CUML is None:
        try:
            import cuml
            _CUML = cuml
        except Exception:
            _CUML = False
    return _CUML or None


def _to_numpy(a) -> np.ndarray:
    """Bring a cuML result (cupy array / cudf frame) back to a float32 ndarray."""
    if hasattr(a, "to_numpy"):       # cudf DataFrame/Series
        a = a.to_numpy()
    elif hasattr(a, "get"):          # cupy ndarray
        a = a.get()
    return np.asarray(a, dtype=np.float32)


def _use_gpu(nrows: int) -> bool:
    return _gpu_enabled() and nrows >= GPU_MIN_ROWS and _cuml() is not None


def _gpu_pca(Xs: np.ndarray, n: int):
    """GPU PCA; returns (coords, evr%) or None to fall back to CPU."""
    if not _use_gpu(Xs.shape[0]) or Xs.shape[0] > INCREMENTAL_PCA_THRESHOLD:
        return None
    try:
        from cuml import PCA as cuPCA
        pca = cuPCA(n_components=n)
        coords = _to_numpy(pca.fit_transform(np.ascontiguousarray(Xs)))
        evr = _to_numpy(pca.explained_variance_ratio_)
        return coords, (evr * 100.0).tolist()
    except Exception:
        return None


def _gpu_umap(Xs: np.ndarray, n: int):
    """GPU UMAP (euclidean only); returns coords or None to fall back to CPU."""
    if not _use_gpu(Xs.shape[0]):
        return None
    try:
        from cuml.manifold import UMAP as cuUMAP
        reducer = cuUMAP(n_components=n, n_neighbors=15, min_dist=0.1, random_state=42)
        X = np.ascontiguousarray(Xs)
        if Xs.shape[0] > UMAP_FIT_CAP:
            idx = RNG.choice(Xs.shape[0], UMAP_FIT_CAP, replace=False)
            reducer.fit(X[idx])
            coords = reducer.transform(X)
        else:
            coords = reducer.fit_transform(X)
        return _to_numpy(coords)
    except Exception:
        return None


def _gpu_tsne(Xs: np.ndarray, n: int):
    """GPU t-SNE (euclidean, 2D-only, no out-of-sample); else None for CPU."""
    # cuML t-SNE is 2D-only and has no .transform(), so only when the whole
    # set fits in one fit (<= cap). Larger sets use the CPU sample+transform.
    if n != 2 or Xs.shape[0] > TSNE_FIT_CAP or not _use_gpu(Xs.shape[0]):
        return None
    try:
        from cuml.manifold import TSNE as cuTSNE
        tsne = cuTSNE(n_components=n, perplexity=30, random_state=42)
        return _to_numpy(tsne.fit_transform(np.ascontiguousarray(Xs)))
    except Exception:
        return None


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


def _fit_umap_fp(packed: np.ndarray, n: int, pca_prereduce: bool = False):
    # Fast path: PCA-reduce bits to ~50D, then euclidean UMAP (GPU-eligible).
    if pca_prereduce:
        return _fit_umap(_fp_to_pca(packed, TSNE_PCA_DIMS), n)
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


def _tsne_grad(n: int, fit_rows: int) -> str:
    """openTSNE negative-gradient method. FFT (interpolation) is 2D-only and only
    wins on large fit sets; Barnes-Hut handles up to 3D and is far faster on small N
    (FFT's fixed grid-setup cost dominates otherwise)."""
    return "interpolation" if (n <= 2 and fit_rows > TSNE_FFT_MIN) else "bh"


def _fit_tsne_fp(packed: np.ndarray, n: int, pca_prereduce: bool = False):
    # Fast path: PCA-reduce bits to ~50D, then Euclidean t-SNE on the dense matrix.
    if pca_prereduce:
        return _fit_tsne(_fp_to_pca(packed, TSNE_PCA_DIMS), n)
    # Default: Tanimoto/Jaccard t-SNE directly on the bit vectors.
    from openTSNE import TSNE
    nrows = len(packed)
    grad = _tsne_grad(n, min(nrows, TSNE_FIT_CAP))
    tsne = TSNE(n_components=n, perplexity=30, metric="jaccard", neighbors="approx",
                negative_gradient_method=grad, random_state=42, n_jobs=TSNE_JOBS)
    if nrows > TSNE_FIT_CAP:
        idx = RNG.choice(nrows, TSNE_FIT_CAP, replace=False)
        emb = tsne.fit(_unpack(packed[idx]))
        coords = np.vstack([emb.transform(_unpack(packed[i:i + FP_BATCH]))
                            for i in range(0, nrows, FP_BATCH)])
    else:
        coords = tsne.fit(_unpack(packed))
    return np.asarray(coords)


def _sample_rows(packed: np.ndarray, cap: int) -> np.ndarray:
    """Random row subsample (without replacement) to bound coverage compute."""
    if len(packed) <= cap:
        return packed
    idx = RNG.choice(len(packed), cap, replace=False)
    return packed[idx]


def _tanimoto_coverage(packed_q: np.ndarray, packed_r: np.ndarray,
                       thr: float, gpu: bool) -> float:
    """Percent of query fingerprints whose nearest neighbour in the reference set
    has Tanimoto >= thr. Tanimoto via bit dot-products: |a&b| = a·b,
    |a|b| = popcount(a)+popcount(b)-|a&b|. Chunked over queries; GPU when enabled."""
    if len(packed_q) == 0 or len(packed_r) == 0:
        return 0.0
    xp = np
    if gpu:
        try:
            import cupy as cp
            xp = cp
        except Exception:
            xp = np
    R = xp.asarray(np.unpackbits(packed_r, axis=1), dtype=xp.float32)   # (r, FP_BITS)
    cntR = R.sum(axis=1)                                                # (r,)
    hits = 0
    for i in range(0, len(packed_q), COVERAGE_Q_CHUNK):
        Q = xp.asarray(np.unpackbits(packed_q[i:i + COVERAGE_Q_CHUNK], axis=1),
                       dtype=xp.float32)                                # (c, FP_BITS)
        cntQ = Q.sum(axis=1)
        inter = Q @ R.T                                                 # (c, r)
        denom = cntQ[:, None] + cntR[None, :] - inter
        tan = xp.where(denom > 0, inter / denom, xp.float32(0.0))
        hits += int((tan.max(axis=1) >= thr).sum())
    return 100.0 * hits / len(packed_q)


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
    gpu = _gpu_pca(Xs, n)
    if gpu is not None:
        return gpu
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
    gpu = _gpu_umap(Xs, n)
    if gpu is not None:
        return gpu
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
    gpu = _gpu_tsne(Xs, n)
    if gpu is not None:
        return gpu
    from openTSNE import TSNE
    grad = _tsne_grad(n, min(Xs.shape[0], TSNE_FIT_CAP))
    tsne = TSNE(n_components=n, perplexity=30, metric="euclidean",
                negative_gradient_method=grad, random_state=42, n_jobs=TSNE_JOBS)
    if Xs.shape[0] > TSNE_FIT_CAP:
        idx = RNG.choice(Xs.shape[0], TSNE_FIT_CAP, replace=False)
        emb = tsne.fit(Xs[idx])      # openTSNE embeddings support out-of-sample
        coords = emb.transform(Xs)   # transform the remaining rows
    else:
        coords = tsne.fit(Xs)
    return np.asarray(coords)


def warmup() -> None:
    """Compile the numba/pynndescent kernels used by t-SNE on tiny dummy data so
    the first real projection after a restart isn't slowed by JIT (~8-27s). Safe
    to call in a background thread; failures are swallowed."""
    try:
        rng = np.random.default_rng(0)
        Z = rng.random((200, 50)).astype(np.float32)          # euclidean t-SNE + UMAP
        if _gpu_enabled() and _cuml() is not None:            # init CUDA context (~1.7s)
            _gpu_umap(Z, 2)
            _gpu_tsne(Z, 2)
        _fit_tsne(Z, 2)
        _fit_umap(Z, 2)
        bits = (rng.random((200, FP_BITS)) > 0.9).astype(np.float32)
        from openTSNE import TSNE                              # jaccard t-SNE (approx NN)
        TSNE(n_components=2, perplexity=30, metric="jaccard", neighbors="approx",
             negative_gradient_method="bh", random_state=42, n_jobs=TSNE_JOBS).fit(bits)
        _fit_umap_fp(np.packbits((bits > 0).astype(np.uint8), axis=1), 2)  # jaccard UMAP
    except Exception:
        pass


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
            tsne_pca_reduce=False, umap_pca_reduce=False):
    if feature == "fingerprint":
        full, packed = _load_fp_matrix(sid, set_names)
        if method == "UMAP":
            coords = _fit_umap_fp(packed, n_dimensions, pca_prereduce=umap_pca_reduce)
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


def overlap(sid, set_names, desc_cols, feature="descriptors", tanimoto_threshold=0.7):
    """Pairwise overlap metrics across sets (full data, independent of the
    display cap):
      - centroid_distance: distance between set centroids in the feature space
      - separation:        centroid_distance / pooled spread (unitless effect
                           size; ~0 = centres coincide, >~2 = well separated)
      - coverage:          directional Tanimoto NN coverage — % of A with a
                           neighbour in B at Tanimoto >= threshold (always from
                           fingerprints, independent of the feature toggle)
    Descriptor mode centroid/spread use standardized descriptor space;
    fingerprint mode uses bit-frequency (Tanimoto-style) space."""
    centroids, spread, counts = {}, {}, {}
    fps = {}
    for name in set_names:
        try:
            _, packed = load_fingerprints(sid, name)
            fps[name] = packed if len(packed) else None
        except Exception:
            fps[name] = None

    if feature == "fingerprint":
        for name in set_names:
            packed = fps[name]
            if packed is None:
                continue
            centroids[name], spread[name] = _fp_centroid_spread(packed)
            counts[name] = int(len(packed))
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
            counts[name] = int(len(pts))

    names = list(centroids)
    distance = {a: {a: 0.0} for a in names}
    separation = {a: {a: 0.0} for a in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d = float(np.linalg.norm(centroids[a] - centroids[b]))
            pooled = (spread[a] + spread[b]) / 2 or 1e-9
            distance[a][b] = distance[b][a] = round(d, 3)
            separation[a][b] = separation[b][a] = round(d / pooled, 3)

    # Structural overlap: directional Tanimoto nearest-neighbour coverage.
    coverage = None
    struct = [n for n in names if fps.get(n) is not None]
    if len(struct) >= 2:
        gpu = _gpu_enabled()
        coverage = {a: {a: 100.0} for a in struct}
        for a in struct:
            q = _sample_rows(fps[a], COVERAGE_QUERY_CAP)
            for b in struct:
                if a == b:
                    continue
                r = _sample_rows(fps[b], COVERAGE_REF_CAP)
                coverage[a][b] = round(_tanimoto_coverage(q, r, tanimoto_threshold, gpu), 1)

    return {
        "sets": [{"name": n, "n": counts[n], "spread": round(spread[n], 3)} for n in names],
        "centroid_distance": distance,
        "separation": separation,
        "coverage": coverage,
        "tanimoto_threshold": tanimoto_threshold,
        "structural_available": coverage is not None,
    }
