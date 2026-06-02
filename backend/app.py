"""ChemScape backend — FastAPI.

Replicates the sculpturatus.com/chemscape API surface:
  GET    /session/new
  POST   /upload
  POST   /project
  POST   /overlap
  POST   /molecule/image
  DELETE /session/{sid}/set/{name}

Also serves the rebuilt frontend at /.
"""
from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import engine
import store
from descriptors import DESCRIPTOR_KEYS, DESCRIPTOR_LABELS

app = FastAPI(title="ChemScape", version="1.0-py")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _warmup_jit() -> None:
    """Pre-compile t-SNE numba/pynndescent kernels off the request path so the
    first projection after a restart is fast."""
    import threading
    threading.Thread(target=engine.warmup, daemon=True).start()

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


# ── Models ────────────────────────────────────────────────────────────────────
class ProjectRequest(BaseModel):
    session_id: str
    set_names: list[str]
    descriptor_cols: list[str] = []
    method: str = "PCA"
    n_dimensions: int = 2
    x_col: str | None = None
    y_col: str | None = None
    z_col: str | None = None
    feature: str = "descriptors"  # "descriptors" | "fingerprint"
    tsne_pca_reduce: bool = False  # fingerprint t-SNE: PCA pre-reduction (faster)
    umap_pca_reduce: bool = False  # fingerprint UMAP: PCA pre-reduce -> euclidean (GPU)


class OverlapRequest(BaseModel):
    session_id: str
    set_names: list[str]
    descriptor_cols: list[str] = []
    method: str = "PCA"
    n_dimensions: int = 2
    feature: str = "descriptors"


class ImageRequest(BaseModel):
    smiles: str
    width: int = 440
    height: int = 360


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/session/new")
def session_new():
    sid = store.new_session()
    sets = {name: {"n_molecules": n} for name, n in store.list_defaults().items()}
    return {"session_id": sid, "descriptor_labels": DESCRIPTOR_LABELS, "sets": sets}


@app.post("/upload")
async def upload(file: UploadFile = File(...), session_id: str | None = Form(None)):
    sid = session_id or store.new_session()
    set_name = store.safe_set_name(file.filename or "dataset")
    suffix = Path(file.filename or "data.csv").suffix or ".csv"

    size = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "file too large")
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    try:
        df = store.read_table(tmp_path)
        dest = store.set_path(sid, set_name)
        n = store.ingest_dataframe(df, set_name, dest)
    except Exception as e:
        raise HTTPException(400, f"could not process file: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    if n == 0:
        raise HTTPException(400, "no valid molecules found")
    return {"session_id": sid, "set_name": set_name, "n_molecules": n}


@app.post("/project")
def project(req: ProjectRequest):
    if not req.set_names:
        raise HTTPException(400, "no datasets selected")
    try:
        return engine.project(
            req.session_id, req.set_names, req.descriptor_cols, req.method,
            req.n_dimensions, req.x_col, req.y_col, req.z_col, req.feature,
            req.tsne_pca_reduce, req.umap_pca_reduce,
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.post("/overlap")
def overlap(req: OverlapRequest):
    if len(req.set_names) < 2:
        raise HTTPException(400, "need at least 2 datasets")
    try:
        return engine.overlap(req.session_id, req.set_names, req.descriptor_cols, req.feature)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.post("/molecule/image")
def molecule_image(req: ImageRequest):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Draw import rdMolDraw2D

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(400, "invalid SMILES")
    d = rdMolDraw2D.MolDraw2DCairo(req.width, req.height)
    d.drawOptions().padding = 0.08  # white background keeps it readable in both themes
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    png = d.GetDrawingText()
    b64 = base64.b64encode(png).decode()
    return {"image": f"data:image/png;base64,{b64}"}


@app.delete("/session/{sid}/set/{name}")
def delete_set(sid: str, name: str):
    store.delete_set(sid, name)
    return {"ok": True}


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
