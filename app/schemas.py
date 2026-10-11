"""Pydantic request/response models (request validation lives here)."""
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20, description="Override default chunks retrieved")

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question must contain at least 3 non-space characters")
        return v


class SourceChunk(BaseModel):
    source: str
    page: int | None = None
    score: float = Field(..., description="Relevance score, 0-1 (higher is better)")
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    grounded: bool = Field(..., description="False when no relevant context was found")
    sources: list[SourceChunk]


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    replaced_existing: bool


class DocumentInfo(BaseModel):
    source: str
    chunks: int


class DocumentList(BaseModel):
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    filename: str
    chunks_removed: int
