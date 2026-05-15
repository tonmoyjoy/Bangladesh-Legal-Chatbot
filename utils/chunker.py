"""
utils/chunker.py
Split raw page documents into overlapping chunks suitable for embedding.
Uses LangChain's RecursiveCharacterTextSplitter with legal-aware separators.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Legal documents use sections, sub-sections, clauses — respect those boundaries
LEGAL_SEPARATORS = [
    "\n\n",      # paragraph break
    "\n",        # line break
    ". ",        # sentence
    "; ",        # clause
    ", ",        # sub-clause
    " ",         # word
    "",          # character (last resort)
]


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split *docs* into smaller overlapping chunks.

    Parameters
    ----------
    docs : List[Document]
        Page-level documents from the PDF loader.
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Characters of overlap between consecutive chunks.

    Returns
    -------
    List[Document]
        Chunk-level documents, each inheriting the parent's metadata plus
        a ``chunk_id`` field.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=LEGAL_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(docs)

    # Attach a unique chunk index for traceability
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    print(f"✅  Created {len(chunks)} chunks from {len(docs)} pages.")
    return chunks
