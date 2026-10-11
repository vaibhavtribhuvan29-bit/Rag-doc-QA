"""Step 3: embeddings + persistent Chroma vector store."""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.errors import ConfigError


class LocalEmbeddings(Embeddings):
    """Free, offline embeddings using ChromaDB's built-in ONNX MiniLM model.

    No API key and no PyTorch needed. The ~80 MB model downloads once on first use.
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache
def get_embeddings() -> Embeddings:
    s = get_settings()
    if s.embedding_provider == "local":
        return LocalEmbeddings()

    if s.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=s.hf_embedding_model)

    if s.embedding_provider == "openai":
        if not s.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is not set (or use EMBEDDING_PROVIDER=local).")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=s.embedding_model, api_key=s.openai_api_key)

    raise ConfigError(f"Unknown EMBEDDING_PROVIDER '{s.embedding_provider}'")


@lru_cache
def get_vectorstore() -> Chroma:
    s = get_settings()
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=s.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(s.chroma_dir),
        # cosine distance -> relevance score in [0, 1] that we can threshold on
        collection_metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[Document]) -> int:
    get_vectorstore().add_documents(chunks)
    return len(chunks)


def delete_source(source: str) -> int:
    """Remove every chunk that came from `source`. Returns number of chunks removed."""
    vs = get_vectorstore()
    ids = vs.get(where={"source": source}).get("ids", [])
    if ids:
        vs.delete(ids=ids)
    return len(ids)


def list_sources() -> list[dict]:
    """Return [{'source': name, 'chunks': n}, ...] for everything indexed."""
    metas = get_vectorstore().get(include=["metadatas"]).get("metadatas") or []
    counts: dict[str, int] = {}
    for m in metas:
        name = (m or {}).get("source", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return [{"source": k, "chunks": v} for k, v in sorted(counts.items())]