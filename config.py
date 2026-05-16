"""
config.py — Centralised configuration loaded from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(Path(__file__).parent / ".env")


class Config:
    # ── Groq / LLM ────────────────────────────────────────────────────────────
    GROQ_API_KEY: str   = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str     = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Vector Store ──────────────────────────────────────────────────────────
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./vectorstore/chroma_db")
    COLLECTION_NAME: str = "bangladesh_law"

    # ── PDF Directory ─────────────────────────────────────────────────────────
    PDF_DIR: str = os.getenv("PDF_DIR", "./data")

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE: int     = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int  = int(os.getenv("CHUNK_OVERLAP", 150))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K: int = int(os.getenv("TOP_K", 5))

    # ── RAG Prompt ────────────────────────────────────────────────────────────
    SYSTEM_PROMPT: str = """You are an expert legal assistant specialising in the laws of Bangladesh. \
Your role is to help citizens, lawyers, and researchers understand Bangladeshi legislation clearly and accurately.

Guidelines:
- Answer ONLY based on the legal context provided below.
- If the answer is not found in the context, say "I could not find relevant information in the loaded legal documents. Please consult a qualified lawyer."
- Quote specific sections, acts, or articles when available.
- Use plain, accessible language while remaining legally precise.
- Do NOT provide personal legal advice; always recommend consulting a qualified advocate for personal matters.
- Format your response with clear headings and bullet points when listing provisions.

Context from Bangladesh Legal Documents:
─────────────────────────────────────────
{context}
─────────────────────────────────────────

User Question: {question}

Answer:"""


cfg = Config()
