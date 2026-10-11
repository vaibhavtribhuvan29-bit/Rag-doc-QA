"""Steps 1 & 2 of the pipeline: document loading and chunking."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.errors import DocumentLoadError, UnsupportedFileType

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document(path: Path, display_name: str | None = None) -> list[Document]:
    """Load a file into Documents with clean `source` (and 1-indexed `page`) metadata."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    name = display_name or path.name
    try:
        if ext == ".pdf":
            raw = PyPDFLoader(str(path)).load()
        else:
            raw = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
    except Exception as exc:  # parser errors vary by library
        raise DocumentLoadError(f"Could not read '{name}': {exc}") from exc

    docs: list[Document] = []
    for d in raw:
        if not d.page_content.strip():
            continue
        meta = {"source": name}
        if ext == ".pdf":
            meta["page"] = int(d.metadata.get("page", 0)) + 1
        docs.append(Document(page_content=d.page_content, metadata=meta))

    if not docs:
        raise DocumentLoadError(
            f"'{name}' contains no extractable text (scanned PDFs need OCR first)."
        )
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split into overlapping chunks. Size/overlap are tunable via settings."""
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks
