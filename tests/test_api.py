"""Lightweight API tests that don't need an OpenAI key or network."""
from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_query_validation_rejects_short_question():
    with TestClient(app) as client:
        r = client.post("/query", json={"question": "hi"})
        assert r.status_code == 422


def test_query_validation_rejects_bad_top_k():
    with TestClient(app) as client:
        r = client.post("/query", json={"question": "What is RAG?", "top_k": 500})
        assert r.status_code == 422


def test_upload_rejects_unsupported_extension():
    with TestClient(app) as client:
        r = client.post("/documents", files={"file": ("evil.exe", b"MZ...", "application/octet-stream")})
        assert r.status_code == 400


def test_upload_rejects_empty_file():
    with TestClient(app) as client:
        r = client.post("/documents", files={"file": ("empty.txt", b"", "text/plain")})
        assert r.status_code == 400
