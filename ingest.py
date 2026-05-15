"""
ingest.py — Bangladesh Legal Chatbot Ingestion Pipeline
═══════════════════════════════════════════════════════
Run this script ONCE (or whenever you add new PDFs) to:
  1. Parse all PDFs in data/
  2. Chunk the text
  3. Embed chunks using HuggingFace sentence-transformers (local, free)
  4. Persist embeddings in ChromaDB (local, free)

Usage:
    python ingest.py
"""
import sys
import time

from config import cfg
from utils.pdf_loader import load_pdfs
from utils.chunker import chunk_documents
from utils.embedder import build_vectorstore


def main() -> None:
    print("=" * 60)
    print("  Bangladesh Legal Chatbot — Ingestion Pipeline")
    print("=" * 60)

    # ── Step 1: Load PDFs ────────────────────────────────────────────────────
    print("\n[1/3] Loading PDFs from:", cfg.PDF_DIR)
    t0 = time.time()
    try:
        docs = load_pdfs(cfg.PDF_DIR)
    except FileNotFoundError as exc:
        print(f"\n❌  {exc}")
        sys.exit(1)

    # ── Step 2: Chunk ────────────────────────────────────────────────────────
    print(f"\n[2/3] Chunking documents (size={cfg.CHUNK_SIZE}, overlap={cfg.CHUNK_OVERLAP})")
    chunks = chunk_documents(docs, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)

    # ── Step 3: Embed & Store ────────────────────────────────────────────────
    print(f"\n[3/3] Embedding & storing in ChromaDB → {cfg.CHROMA_DIR}")
    build_vectorstore(
        chunks=chunks,
        embedding_model=cfg.EMBEDDING_MODEL,
        chroma_dir=cfg.CHROMA_DIR,
        collection_name=cfg.COLLECTION_NAME,
    )

    elapsed = time.time() - t0
    print(f"\n✅  Ingestion complete in {elapsed:.1f}s — {len(chunks)} chunks indexed.")
    print("    Now run:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
