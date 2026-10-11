"""Steps 4 & 5: semantic retrieval + grounded answer generation."""
import logging
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.errors import ConfigError, LLMError
from app.loader import load_document, split_documents
from app.schemas import QueryResponse, SourceChunk
from app.vectorstore import add_chunks, delete_source, get_vectorstore

log = logging.getLogger(__name__)

NO_ANSWER = "I couldn't find that in the uploaded documents."

# Prompt engineering for groundedness: answer ONLY from context, admit gaps, cite sources.
SYSTEM_PROMPT = """You are a careful document question-answering assistant.

Rules:
1. Answer ONLY using the numbered context passages below. Do not use outside knowledge.
2. If the context does not contain the answer, reply exactly: "{no_answer}"
3. If the context only partially answers the question, answer the supported part and
   clearly say what is missing.
4. Cite the passages you used like [1], [2] at the end of the relevant sentences.
5. Be concise and do not invent names, numbers, dates, or quotes."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    if not s.llm_api_key:
        raise ConfigError("LLM_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatOpenAI(
        model=s.llm_model,
        temperature=s.llm_temperature,
        api_key=s.llm_api_key,
        base_url=s.llm_base_url or None,
    )


def ingest_file(path: Path, display_name: str) -> tuple[int, bool]:
    """Load -> chunk -> embed -> store. Re-uploading a file replaces its old chunks."""
    docs = load_document(path, display_name)
    chunks = split_documents(docs)
    try:
        replaced = delete_source(display_name) > 0
        count = add_chunks(chunks)
    except ConfigError:
        raise
    except Exception as exc:
        log.exception("Indexing failed")
        raise LLMError(f"Embedding/indexing failed: {exc}") from exc
    return count, replaced


def _format_context(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page")
        where = f"{d.metadata.get('source', 'unknown')}" + (f", p.{page}" if page else "")
        parts.append(f"[{i}] ({where})\n{d.page_content.strip()}")
    return "\n\n".join(parts)


def answer_question(question: str, top_k: int | None = None) -> QueryResponse:
    s = get_settings()
    k = top_k or s.top_k

    # Semantic retrieval with relevance scores
    try:
        results = get_vectorstore().similarity_search_with_relevance_scores(question, k=k)
    except ConfigError:
        raise
    except Exception as exc:
        log.exception("Retrieval failed")
        raise LLMError(f"Retrieval failed: {exc}") from exc

    # Hallucination guard #1: drop weak matches; if nothing is relevant, don't call the LLM at all.
    relevant = [(d, score) for d, score in results if score >= s.score_threshold]
    if not relevant:
        return QueryResponse(answer=NO_ANSWER, grounded=False, sources=[])

    docs = [d for d, _ in relevant]
    chain = PROMPT | _get_llm() | StrOutputParser()
    try:
        answer = chain.invoke(
            {"context": _format_context(docs), "question": question, "no_answer": NO_ANSWER}
        )
    except ConfigError:
        raise
    except Exception as exc:
        log.exception("LLM call failed")
        raise LLMError(f"LLM call failed: {exc}") from exc

    sources = [
        SourceChunk(
            source=d.metadata.get("source", "unknown"),
            page=d.metadata.get("page"),
            score=round(float(score), 3),
            snippet=d.page_content.strip()[:300],
        )
        for d, score in relevant
    ]
    return QueryResponse(answer=answer.strip(), grounded=True, sources=sources)