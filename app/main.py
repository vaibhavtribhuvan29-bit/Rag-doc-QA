"""FastAPI REST layer: validation, error handling, and routes."""
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app import rag
from app.config import get_settings
from app.errors import ConfigError, DocumentLoadError, LLMError, UnsupportedFileType
from app.loader import SUPPORTED_EXTENSIONS
from app.schemas import (
    DeleteResponse,
    DocumentList,
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
    return UploadResponse(filename=filename, chunks_indexed=count, replaced_existing=replaced)


@app.get("/documents", response_model=DocumentList)
def get_documents():
    return DocumentList(documents=list_sources())


@app.delete("/documents/{filename}", response_model=DeleteResponse)
def remove_document(filename: str):
    name = _safe_filename(filename)
    removed = delete_source(name)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No indexed document named '{name}'.")
    (get_settings().upload_dir / name).unlink(missing_ok=True)
    return DeleteResponse(filename=name, chunks_removed=removed)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return rag.answer_question(req.question, req.top_k)
