# What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is a technique that improves large language model
answers by retrieving relevant passages from an external knowledge source and supplying
them to the model as context at question time.

## Pipeline

1. **Load** documents (PDF, text, Markdown).
2. **Chunk** them into overlapping segments so each fits comfortably in a prompt.
3. **Embed** each chunk into a vector and store it in a vector database.
4. **Retrieve** the chunks whose embeddings are closest to the question's embedding.
5. **Generate** an answer from the question plus the retrieved chunks.

## Why it reduces hallucination

The model is instructed to answer only from the retrieved context. When retrieval finds
nothing relevant, a well-built system declines to answer instead of guessing.

## Tuning knobs

- Chunk size: smaller chunks are more precise, larger chunks keep more context.
- Chunk overlap: prevents answers from being split across chunk boundaries.
- Top-k: how many chunks to retrieve per question.
- Score threshold: minimum similarity a chunk needs before it is used.
