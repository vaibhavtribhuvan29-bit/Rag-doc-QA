"""FastAPI REST layer: validation, error handling, and routes."""
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from app import db, rag
from app.config import get_settings
from app.errors import ConfigError, DocumentLoadError, LLMError, UnsupportedFileType
from app.loader import SUPPORTED_EXTENSIONS
from app.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentList,
    HistoryItem,
    HistoryList,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.vectorstore import delete_source, list_sources

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    db.init_db()
    yield


app = FastAPI(
    title="RAG Document Q&A Assistant",
    description="Upload documents, then ask questions answered strictly from their content.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---- Error handling: map domain errors to clean HTTP responses ----
@app.exception_handler(UnsupportedFileType)
async def _unsupported(_, exc: UnsupportedFileType):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(DocumentLoadError)
async def _load_error(_, exc: DocumentLoadError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ConfigError)
async def _config_error(_, exc: ConfigError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMError)
async def _llm_error(_, exc: LLMError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


def _safe_filename(name: str | None) -> str:
    base = Path(name or "").name  # strips any path components
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip()
    if not base or base.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return base


# ---- Routes ----
@app.get("/health")
def health():
    return {"status": "ok"}
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/documents", response_model=UploadResponse, status_code=201)
def upload_document(file: UploadFile = File(...)):
    s = get_settings()
    filename = _safe_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    limit = s.max_upload_mb * 1024 * 1024
    content = file.file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=413, detail=f"File exceeds {s.max_upload_mb} MB limit.")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    s.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = s.upload_dir / filename
    dest.write_bytes(content)

    try:
        count, replaced = rag.ingest_file(dest, filename)
    except Exception:
        dest.unlink(missing_ok=True)  # don't keep files we failed to index
        raise
    db.save_document(filename, len(content), count)
    return UploadResponse(filename=filename, chunks_indexed=count, replaced_existing=replaced)


@app.get("/documents", response_model=DocumentList)
def get_documents():
    meta = db.documents_by_name()
    docs = []
    for item in list_sources():
        rec = meta.get(item["source"], {})
        docs.append(
            DocumentInfo(
                source=item["source"],
                chunks=item["chunks"],
                size_bytes=rec.get("size_bytes"),
                uploaded_at=rec.get("uploaded_at"),
            )
        )
    return DocumentList(documents=docs)


@app.delete("/documents/{filename}", response_model=DeleteResponse)
def remove_document(filename: str):
    name = _safe_filename(filename)
    removed = delete_source(name)
    had_record = db.delete_document(name)
    if removed == 0 and not had_record:
        raise HTTPException(status_code=404, detail=f"No indexed document named '{name}'.")
    (get_settings().upload_dir / name).unlink(missing_ok=True)
    return DeleteResponse(filename=name, chunks_removed=removed)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start = time.perf_counter()
    result = rag.answer_question(req.question, req.top_k)
    result.latency_ms = int((time.perf_counter() - start) * 1000)
    try:  # history is a nice-to-have: never fail the answer because of it
        db.log_query(
            req.question,
            result.answer,
            result.grounded,
            [src.model_dump() for src in result.sources],
            result.latency_ms,
        )
    except Exception:
        log.exception("Could not save query to history")
    return result


@app.get("/history", response_model=HistoryList)
def get_history(limit: int = Query(20, ge=1, le=200)):
    """Most recent questions first."""
    return HistoryList(items=[HistoryItem(**row) for row in db.list_history(limit)])


@app.delete("/history")
def clear_history():
    return {"deleted": db.clear_history()}