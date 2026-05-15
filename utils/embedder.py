"""
utils/embedder.py
Build (or load) a ChromaDB vector store from document chunks.
Uses HuggingFace sentence-transformers for local, free embeddings.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def _get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Return a HuggingFace embedding model (downloaded & cached locally)."""
    print(f"🔄  Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},       # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    chunks: List[Document],
    embedding_model: str,
    chroma_dir: str,
    collection_name: str,
) -> Chroma:
    """
    Embed *chunks* and persist them into a ChromaDB collection.

    If the collection already exists at *chroma_dir*, it is **replaced**
    (call `load_vectorstore` instead to reuse an existing one).
    """
    embeddings = _get_embeddings(embedding_model)

    Path(chroma_dir).mkdir(parents=True, exist_ok=True)

    print(f"🔄  Embedding {len(chunks)} chunks and saving to ChromaDB…")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name=collection_name,
    )
    print(f"✅  Vector store saved to: {chroma_dir}")
    return vectorstore


def load_vectorstore(
    embedding_model: str,
    chroma_dir: str,
    collection_name: str,
) -> Chroma:
    """
    Load an already-built ChromaDB collection from disk.
    Raises FileNotFoundError if the collection doesn't exist yet.
    """
    if not Path(chroma_dir).exists():
        raise FileNotFoundError(
            f"No vector store found at '{chroma_dir}'.\n"
            "Run  python ingest.py  first to build it."
        )

    embeddings = _get_embeddings(embedding_model)
    print(f"✅  Loaded existing vector store from: {chroma_dir}")

    return Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def vectorstore_exists(chroma_dir: str) -> bool:
    """Return True if a persisted ChromaDB collection exists."""
    return Path(chroma_dir).exists() and any(Path(chroma_dir).iterdir())
