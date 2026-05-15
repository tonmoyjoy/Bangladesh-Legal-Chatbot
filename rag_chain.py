"""
rag_chain.py — Retrieval-Augmented Generation Chain
════════════════════════════════════════════════════
Wires together:
  • ChromaDB retriever  (similarity search over embedded legal chunks)
  • Groq LLM (LLaMA 3, free tier)           (answer generation)
  • LangChain LCEL pipeline                  (orchestration)
"""
from __future__ import annotations

from typing import Any, Dict, Generator, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from config import cfg
from utils.embedder import load_vectorstore


# ─── Singleton cache so we don't reload the model on every call ──────────────
_vectorstore = None
_chain       = None


def _format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved chunks into a numbered context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        ref  = f"{meta.get('filename', 'Unknown')} — Page {meta.get('page', '?')}"
        parts.append(f"[{i}] {ref}\n{doc.page_content}")
    return "\n\n".join(parts)


def _build_chain():
    """Construct and cache the full RAG chain."""
    global _vectorstore, _chain

    # ── Load vector store ────────────────────────────────────────────────────
    _vectorstore = load_vectorstore(
        embedding_model=cfg.EMBEDDING_MODEL,
        chroma_dir=cfg.CHROMA_DIR,
        collection_name=cfg.COLLECTION_NAME,
    )

    retriever = _vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": cfg.TOP_K},
    )

    # ── LLM (Groq — free) ────────────────────────────────────────────────────
    llm = ChatGroq(
        model=cfg.GROQ_MODEL,
        api_key=cfg.GROQ_API_KEY,
        temperature=0.1,           # low temperature for factual legal answers
        max_tokens=1024,
        streaming=True,
    )

    # ── Prompt ───────────────────────────────────────────────────────────────
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=cfg.SYSTEM_PROMPT,
    )

    # ── LCEL Chain: retrieve → format → prompt → LLM → parse ─────────────────
    _chain = (
        {
            "context":  retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return _chain


def get_chain():
    """Return the cached RAG chain, building it on first call."""
    global _chain
    if _chain is None:
        _chain = _build_chain()
    return _chain


def query(question: str) -> Generator[str, None, None]:
    """
    Stream the answer to *question* token-by-token.

    Yields
    ------
    str
        Incremental text tokens from the LLM.
    """
    chain = get_chain()
    for token in chain.stream(question):
        yield token


def query_with_sources(question: str) -> Dict[str, Any]:
    """
    Non-streaming query that returns both the answer and source chunks.

    Returns
    -------
    dict with keys:
        answer   : str
        sources  : List[Document]
    """
    global _vectorstore

    # Ensure vectorstore is loaded
    if _vectorstore is None:
        _build_chain()

    # Retrieve relevant chunks
    sources: List[Document] = _vectorstore.similarity_search(
        question, k=cfg.TOP_K
    )

    # Get full answer (non-streaming)
    chain = get_chain()
    answer = chain.invoke(question)

    return {"answer": answer, "sources": sources}
