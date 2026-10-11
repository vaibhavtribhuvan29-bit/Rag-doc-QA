"""Central configuration, loaded from environment variables / .env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (any OpenAI-compatible API: Gemini free tier by default; also Groq, OpenAI, ...)
    llm_api_key: str = ""
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-flash-latest"
    llm_temperature: float = 0.0

    # Embeddings: "local" (free, runs on your PC) | "huggingface" | "openai"
    embedding_provider: str = "local"
    openai_api_key: str = ""  # only needed if embedding_provider="openai"
    embedding_model: str = "text-embedding-3-small"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking & retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    score_threshold: float = 0.3

    # Storage
    chroma_dir: Path = Path("data/chroma")
    upload_dir: Path = Path("data/uploads")
    collection_name: str = "documents"
    max_upload_mb: int = 20

    # Relational database for document records + query history.
    # SQLite by default (zero setup). For PostgreSQL:
    #   DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/ragdb
    database_url: str = "sqlite:///data/app.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()