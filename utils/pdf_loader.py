"""
utils/pdf_loader.py
Parse all PDFs in the data/ folder using PyMuPDF (fitz).
Returns a list of LangChain Document objects with rich metadata.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document
from tqdm import tqdm
from io import BytesIO

# Optional OCR dependencies (pytesseract + Pillow). If present, we'll
# attempt OCR on pages that contain no extractable text (scanned PDFs).
try:
    import pytesseract
    from PIL import Image
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False


def _configure_tesseract() -> bool:
    """Find a working tesseract.exe on Windows and attach it to pytesseract."""
    if not _HAS_OCR:
        return False

    candidates = []

    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd:
        candidates.append(env_cmd)

    path_cmd = shutil.which("tesseract")
    if path_cmd:
        candidates.append(path_cmd)

    candidates.extend(
        [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    )

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return True

    return False


_TESSERACT_READY = _configure_tesseract()
_OCR_WARNING_SHOWN = False


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and garbled characters."""
    import re
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove stray form-feed / non-printable chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def load_pdfs(pdf_dir: str) -> List[Document]:
    """
    Recursively load every PDF found in *pdf_dir*.

    Returns
    -------
    List[Document]
        Each document corresponds to one PDF page and carries metadata:
        source, page, total_pages, filename.
    """
    pdf_dir_path = Path(pdf_dir)
    pdf_files = sorted(pdf_dir_path.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{pdf_dir}'.\n"
            "Place your Bangladesh law PDFs inside the data/ folder and re-run ingest.py."
        )

    all_docs: List[Document] = []
    global _OCR_WARNING_SHOWN

    for pdf_path in tqdm(pdf_files, desc="📄 Parsing PDFs"):
        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            filename = pdf_path.name

            for page_num, page in enumerate(doc, start=1):
                raw_text = page.get_text("text")          # plain text extraction
                cleaned  = _clean_text(raw_text)

                # If page has very little extracted text, try OCR (optional)
                if len(cleaned) < 30:
                    if _TESSERACT_READY:
                        try:
                            pix = page.get_pixmap(dpi=300)
                            img_bytes = pix.tobytes("png")
                            with Image.open(BytesIO(img_bytes)) as img:
                                ocr_image = img.convert("L")
                                ocr_image.load()
                            ocr_text = pytesseract.image_to_string(ocr_image)
                            cleaned = _clean_text(ocr_text)
                        except Exception as exc:
                            print(f"⚠️  OCR failed for {filename} page {page_num}: {exc}")
                    elif _HAS_OCR and not _OCR_WARNING_SHOWN:
                        print(
                            "⚠️  OCR skipped because tesseract.exe was not found. "
                            "Install Tesseract and restart VS Code, then run ingest.py again."
                        )
                        _OCR_WARNING_SHOWN = True

                if len(cleaned) < 30:                     # still near-empty → skip
                    continue

                all_docs.append(
                    Document(
                        page_content=cleaned,
                        metadata={
                            "source":      str(pdf_path.relative_to(pdf_dir_path)),
                            "filename":    filename,
                            "page":        page_num,
                            "total_pages": total_pages,
                        },
                    )
                )

            doc.close()
        except Exception as exc:
            print(f"⚠️  Could not parse {pdf_path.name}: {exc}")
            continue

    print(f"✅  Loaded {len(all_docs)} pages from {len(pdf_files)} PDF(s).")
    return all_docs
