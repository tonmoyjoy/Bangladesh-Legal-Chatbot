"""Build or update the vector index from the PDFs in ``data/``.

Usage:
    python ingest.py             # incremental: upsert changed chunks, drop stale ones
    python ingest.py --rebuild   # delete the collection and rebuild from scratch
    python ingest.py --dry-run   # parse + chunk only, print stats, write nothing
    python ingest.py --allow-suspicious   # index PDFs whose title contradicts the file name

Exit codes: 0 ok · 1 no PDFs / no text / OCR unavailable · 2 configuration or
index problem · 3 suspicious documents found (nothing written).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter

from legalbot.chunker import chunk_document_pages, group_by_source
from legalbot.config import ConfigError, load_settings
from legalbot.diagnostics import ERROR, has_scanned_pdf, ocr_diagnostics, pdf_files
from legalbot.embeddings import build_embedder
from legalbot.pdf_loader import load_pdfs
from legalbot.vectorstore import IncompatibleIndexError, VectorStore, make_client

logger = logging.getLogger("ingest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="delete and rebuild the collection")
    parser.add_argument("--dry-run", action="store_true", help="parse and chunk only")
    parser.add_argument("--pdf-dir", help="override PDF_DIR")
    parser.add_argument(
        "--allow-suspicious",
        action="store_true",
        help="index PDFs whose printed title contradicts their file name (also: ALLOW_SUSPICIOUS_DOCS=true)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = load_settings()
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2
    pdf_dir = args.pdf_dir or settings.pdf_dir

    t0 = time.time()
    ocr_warnings: list[str] = []
    files = pdf_files(pdf_dir)
    if files and has_scanned_pdf(files):
        for diag in ocr_diagnostics(settings):
            if diag.severity == ERROR:
                logger.error("%s\n    fix: %s", diag.message, diag.fix)
                return 1
            ocr_warnings.append(f"{diag.message}\n    fix: {diag.fix}")
            logger.warning("%s", diag.message)
    try:
        pages = load_pdfs(
            pdf_dir,
            settings.ocr_lang,
            ocr_cache_dir=settings.ocr_cache_dir,
            ocr_workers=settings.ocr_workers or None,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    groups = group_by_source(pages)
    chunks_by_source = {
        source: chunk_document_pages(doc_pages, settings.chunk_size, settings.chunk_overlap)
        for source, doc_pages in groups.items()
    }
    total = sum(len(c) for c in chunks_by_source.values())
    for source, chunks in chunks_by_source.items():
        with_section = sum(1 for c in chunks if c.metadata.get("section"))
        logger.info("%s → %d chunks (%d with section numbers)", source, len(chunks), with_section)
    if total == 0:
        logger.error("No text could be extracted (scanned PDFs need Tesseract OCR).")
        return 1

    suspicious = {
        chunks[0].metadata["filename"]: chunks[0].metadata["suspicious_reason"]
        for chunks in chunks_by_source.values()
        if chunks and chunks[0].metadata.get("suspicious")
    }
    allow_suspicious = args.allow_suspicious or settings.allow_suspicious_docs
    for name, reason in suspicious.items():
        print(f"⚠ SUSPICIOUS: {name}\n    {reason}")
    for warning in ocr_warnings:
        print(f"⚠ OCR: {warning}")

    if args.dry_run:
        acts = Counter(c.metadata["act"] for cs in chunks_by_source.values() for c in cs)
        for act, n in acts.most_common():
            print(f"{n:6d}  {act}")
        print(f"Dry run: {total} chunks from {len(pages)} pages (nothing written).")
        return 0

    if suspicious and not allow_suspicious:
        print(
            f"Stopped: {len(suspicious)} suspicious document(s) — nothing was written.\n"
            "Replace the PDF(s) with the correct statute, or re-run with --allow-suspicious "
            "(or ALLOW_SUSPICIOUS_DOCS=true) to index them anyway; the app will show a warning."
        )
        return 3

    embedder = build_embedder(settings.embedding_backend, settings.embedding_model, settings.model_cache_dir)
    try:
        store = VectorStore.create(
            make_client(settings), settings.collection_name, embedder, rebuild=args.rebuild
        )
    except IncompatibleIndexError as exc:
        logger.error("%s", exc)
        return 2

    removed = 0
    for source, chunks in chunks_by_source.items():
        store.upsert(chunks)
        removed += store.remove_stale(source, (str(c.metadata["chunk_id"]) for c in chunks))

    logger.info(
        "Done in %.1fs: %d chunks indexed, %d stale removed, collection size %d",
        time.time() - t0,
        total,
        removed,
        store.count(),
    )
    print("Next: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
