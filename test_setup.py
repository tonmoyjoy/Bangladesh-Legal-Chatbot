"""
test_setup.py — Pre-flight check for Bangladesh Legal Chatbot
Run before ingest.py to verify all dependencies and configs are correct.

Usage:
    python test_setup.py
"""
import sys


def check(label: str, ok: bool, hint: str = "") -> bool:
    icon = "✅" if ok else "❌"
    print(f"  {icon}  {label}")
    if not ok and hint:
        print(f"       ↳ {hint}")
    return ok


def main() -> None:
    print("\n" + "=" * 55)
    print("  Bangladesh Legal Chatbot — Setup Verification")
    print("=" * 55)
    all_ok = True

    # ── Python version ────────────────────────────────────────────────────────
    print("\n[1] Python")
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 9
    all_ok &= check(f"Python {v.major}.{v.minor}.{v.micro}", ok,
                    "Requires Python 3.9+")

    # ── Required packages ─────────────────────────────────────────────────────
    print("\n[2] Package imports")
    packages = {
        "fitz":                    "PyMuPDF — pip install pymupdf",
        "langchain":               "pip install langchain",
        "langchain_community":     "pip install langchain-community",
        "langchain_groq":          "pip install langchain-groq",
        "langchain_huggingface":   "pip install langchain-huggingface",
        "chromadb":                "pip install chromadb",
        "sentence_transformers":   "pip install sentence-transformers",
        "streamlit":               "pip install streamlit",
        "dotenv":                  "pip install python-dotenv",
        "groq":                    "pip install groq",
    }
    for pkg, hint in packages.items():
        try:
            __import__(pkg)
            all_ok &= check(pkg, True)
        except ImportError:
            all_ok &= check(pkg, False, hint)

    # ── .env / config ─────────────────────────────────────────────────────────
    print("\n[3] Environment / Config")
    from config import cfg
    all_ok &= check(
        f"GROQ_API_KEY set ({cfg.GROQ_API_KEY[:6]}…)" if cfg.GROQ_API_KEY else "GROQ_API_KEY",
        bool(cfg.GROQ_API_KEY),
        "Set GROQ_API_KEY in .env (get free key at https://console.groq.com)",
    )
    all_ok &= check(f"GROQ_MODEL = {cfg.GROQ_MODEL}", True)
    all_ok &= check(f"PDF_DIR    = {cfg.PDF_DIR}",    True)
    all_ok &= check(f"CHROMA_DIR = {cfg.CHROMA_DIR}", True)

    # ── PDF files ─────────────────────────────────────────────────────────────
    print("\n[4] Data files")
    from pathlib import Path
    pdf_files = list(Path(cfg.PDF_DIR).rglob("*.pdf"))
    all_ok &= check(
        f"PDFs found: {len(pdf_files)} file(s)",
        len(pdf_files) > 0,
        f"Put Bangladesh law PDFs in '{cfg.PDF_DIR}/' — see data/README.txt",
    )

    # ── Vector store ─────────────────────────────────────────────────────────
    print("\n[5] Vector store")
    from utils.embedder import vectorstore_exists
    vs = vectorstore_exists(cfg.CHROMA_DIR)
    check(
        "ChromaDB vector store" + (" exists" if vs else " not found (run ingest.py)"),
        vs,
        "Run:  python ingest.py",
    )
    # Not fatal — just informational before first ingest

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    if all_ok:
        print("  🎉  All checks passed! You're ready to go.")
        print("\n  Next steps:")
        print("    1. python ingest.py     ← build vector store")
        print("    2. streamlit run app.py ← launch the chatbot")
    else:
        print("  ⚠️  Some checks failed. Fix the issues above and re-run.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
