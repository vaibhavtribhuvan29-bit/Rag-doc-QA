# RAG Document Q&A Assistant

A Retrieval-Augmented Generation (RAG) application that lets you upload documents and ask questions about them. Answers are generated **only from the retrieved document content**, with source citations, and the system declines to answer when nothing relevant is found.

**Stack:** Python · LangChain · FastAPI · ChromaDB (vector database) · OpenAI (LLM + embeddings)

## How it works

```
 Upload                                          Ask
   │                                              │
   ▼                                              ▼
Load (PDF/TXT/MD) → Chunk → Embed → ChromaDB ──► Semantic retrieval (top-k + score threshold)
                                                  │
                                                  ▼
                                   Grounded prompt + LLM → Answer + cited sources
```

1. **Document loading** – PDF, TXT and Markdown files, with source/page metadata (`app/loader.py`).
2. **Chunking** – `RecursiveCharacterTextSplitter` with configurable size and overlap.
3. **Vector storage** – embeddings persisted in a local ChromaDB collection using cosine similarity (`app/vectorstore.py`).
4. **Semantic retrieval** – top-k search with relevance scores; weak matches are filtered out.
5. **Answer generation** – a LangChain (LCEL) chain with a strict grounding prompt (`app/rag.py`).
6. **REST API** – FastAPI with request validation and error handling (`app/main.py`).

### Reducing hallucinations

- **Strict prompt**: answer only from numbered context, cite `[1]`, `[2]`, admit gaps, never invent facts.
- **Score threshold**: if no chunk is relevant enough, the LLM is *not called* and the API returns `grounded: false`.
- **Temperature 0** by default.
- **Citations returned** with every answer (source file, page, score, snippet) so answers can be verified.
- **Tunable retrieval**: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `SCORE_THRESHOLD`.

## Quick start

```bash
git clone https://github.com/<your-username>/rag-document-qa.git
cd rag-document-qa

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then add your OPENAI_API_KEY
uvicorn app.main:app --reload
```

Open the interactive docs at **http://localhost:8000/docs**.

> No OpenAI key for embeddings? Set `EMBEDDING_PROVIDER=huggingface` to embed locally (the LLM still needs a key).

### Docker

```bash
docker build -t rag-document-qa .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/code/data rag-document-qa
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/documents` | Upload and index a `.pdf`, `.txt` or `.md` file |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{filename}` | Remove a document from the index |
| `POST` | `/query` | Ask a question |

### Example

```bash
# 1. Upload
curl -X POST http://localhost:8000/documents -F "file=@sample_docs/what_is_rag.md"

# 2. Ask
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Why does RAG reduce hallucination?"}'
```

Response:

```json
{
  "answer": "RAG instructs the model to answer only from retrieved context, and when nothing relevant is found a well-built system declines instead of guessing [1].",
  "grounded": true,
  "sources": [
    {"source": "what_is_rag.md", "page": null, "score": 0.62, "snippet": "..."}
  ]
}
```

If nothing relevant is found: `{"answer": "I couldn't find that in the uploaded documents.", "grounded": false, "sources": []}`

### Error handling

| Status | Cause |
|--------|-------|
| 400 | Unsupported file type, empty file, invalid filename |
| 413 | File larger than `MAX_UPLOAD_MB` |
| 422 | Invalid request body (e.g. question too short) or unreadable/empty document |
| 502 | Upstream LLM/embedding provider failure |
| 503 | Missing configuration (e.g. no API key) |

## Tuning guide

| Symptom | Try |
|---------|-----|
| Answers miss details | Increase `TOP_K`, or increase `CHUNK_SIZE` |
| Answers include irrelevant info | Raise `SCORE_THRESHOLD`, lower `TOP_K` |
| Answers cut off mid-fact | Increase `CHUNK_OVERLAP` |
| Says "not found" too often | Lower `SCORE_THRESHOLD` (e.g. 0.2) |

## Project structure

```
rag-document-qa/
├── app/
│   ├── main.py          # FastAPI routes, validation, error handlers
│   ├── rag.py           # ingestion + retrieval + answer chain
│   ├── loader.py        # document loading and chunking
│   ├── vectorstore.py   # embeddings + ChromaDB
│   ├── schemas.py       # Pydantic models
│   ├── config.py        # settings from .env
│   └── errors.py        # domain exceptions
├── tests/test_api.py    # API tests (no API key needed)
├── sample_docs/         # sample document to try
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## License

MIT
