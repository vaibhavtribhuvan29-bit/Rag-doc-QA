"""Relational database: document records + query history.

SQLite by default (no setup). Set DATABASE_URL to use PostgreSQL instead.
Vector embeddings still live in ChromaDB; this DB stores the app's own data.
"""
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text, create_engine, delete, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import get_settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(dt: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; treat them as UTC."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    chunks: Mapped[int] = mapped_column(Integer)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    grounded: Mapped[bool] = mapped_column(Boolean)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


@lru_cache
def get_engine():
    url = make_url(get_settings().database_url)
    kwargs: dict = {}
    if url.get_backend_name() == "sqlite":
        if url.database and url.database != ":memory:":
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}  # FastAPI uses a thread pool
    return create_engine(url, pool_pre_ping=True, **kwargs)


@lru_cache
def _sessionmaker() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def init_db() -> None:
    """Create tables if they don't exist yet (called on app startup)."""
    Base.metadata.create_all(get_engine())


# ---- documents ----
def save_document(filename: str, size_bytes: int, chunks: int) -> None:
    with _sessionmaker()() as s:
        rec = s.scalar(select(DocumentRecord).where(DocumentRecord.filename == filename))
        if rec:
            rec.size_bytes, rec.chunks, rec.uploaded_at = size_bytes, chunks, _now()
        else:
            s.add(DocumentRecord(filename=filename, size_bytes=size_bytes, chunks=chunks))
        s.commit()


def delete_document(filename: str) -> bool:
    with _sessionmaker()() as s:
        result = s.execute(delete(DocumentRecord).where(DocumentRecord.filename == filename))
        s.commit()
        return (result.rowcount or 0) > 0


def documents_by_name() -> dict[str, dict]:
    with _sessionmaker()() as s:
        rows = s.scalars(select(DocumentRecord)).all()
        return {
            r.filename: {"size_bytes": r.size_bytes, "uploaded_at": _utc(r.uploaded_at)}
            for r in rows
        }


# ---- query history ----
def log_query(question: str, answer: str, grounded: bool, sources: list[dict], latency_ms: int) -> None:
    with _sessionmaker()() as s:
        s.add(
            QueryLog(
                question=question,
                answer=answer,
                grounded=grounded,
                sources=sources,
                latency_ms=latency_ms,
            )
        )
        s.commit()


def list_history(limit: int = 20) -> list[dict]:
    """Newest first."""
    with _sessionmaker()() as s:
        rows = s.scalars(select(QueryLog).order_by(QueryLog.id.desc()).limit(limit)).all()
        return [
            {
                "id": r.id,
                "question": r.question,
                "answer": r.answer,
                "grounded": r.grounded,
                "sources": r.sources or [],
                "latency_ms": r.latency_ms,
                "created_at": _utc(r.created_at),
            }
            for r in rows
        ]


def clear_history() -> int:
    with _sessionmaker()() as s:
        result = s.execute(delete(QueryLog))
        s.commit()
        return result.rowcount or 0