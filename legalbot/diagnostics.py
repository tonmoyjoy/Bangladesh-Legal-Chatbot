"""Setup diagnostics: say exactly what is missing and the command that fixes it.

Used by the app (blocking problems before chatting), by ``ingest.py`` (OCR
prerequisites) and as a CLI::

    python -m legalbot.diagnostics
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from legalbot.config import Settings
from legalbot.vectorstore import IndexStatus

logger = logging.getLogger(__name__)

ERROR = "error"  # blocks the feature
WARNING = "warning"  # works, but with reduced capability the user must know about


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str
    message: str
    fix: str  # exact command or action


def pdf_files(pdf_dir: str | Path) -> list[Path]:
    root = Path(pdf_dir)
    return sorted(root.rglob("*.pdf")) if root.exists() else []


def has_scanned_pdf(files: list[Path], sample_pages: int = 3) -> bool:
    import pymupdf

    for path in files:
        try:
            with pymupdf.open(str(path)) as pdf:
                pages = [pdf[i] for i in range(min(sample_pages, len(pdf)))]
                if pages and all(len(p.get_text("text").strip()) < 30 for p in pages):
                    return True
        except Exception as exc:  # unreadable PDFs are reported by ingest itself
            logger.debug("Could not inspect %s: %s", path.name, type(exc).__name__)
    return False


def ocr_diagnostics(settings: Settings) -> list[Diagnostic]:
    """Tesseract + language packs (only needed to ingest scanned PDFs)."""
    from legalbot.pdf_loader import find_tesseract

    cmd = find_tesseract()
    if cmd is None:
        return [
            Diagnostic(
                "missing_tesseract",
                ERROR,
                "Tesseract OCR is not installed, so scanned (image-only) PDFs cannot be read.",
                "Windows: install from https://github.com/UB-Mannheim/tesseract/wiki (tick Bengali), "
                "or set TESSERACT_CMD in .env · Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-ben "
                "· macOS: brew install tesseract tesseract-lang",
            )
        ]
    wanted = [code for code in settings.ocr_lang.split("+") if code]
    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = cmd
        installed = set(pytesseract.get_languages(config=""))
    except Exception as exc:
        return [
            Diagnostic(
                "tesseract_unusable",
                ERROR,
                f"Tesseract was found but could not be queried ({type(exc).__name__}).",
                f'Run "{cmd}" --list-langs to check the installation.',
            )
        ]
    found: list[Diagnostic] = []
    if "ben" in wanted and "ben" not in installed:
        found.append(
            Diagnostic(
                "missing_bengali_ocr",
                WARNING,
                "The Bengali OCR language pack (ben.traineddata) is not installed: Bangla scans will "
                "not be recognised (English scans are unaffected).",
                'Windows: copy ben.traineddata into "C:\\Program Files\\Tesseract-OCR\\tessdata\\" · '
                "Ubuntu: sudo apt install tesseract-ocr-ben · or set OCR_LANG=eng to silence this.",
            )
        )
    missing_other = [c for c in wanted if c not in installed and c != "ben"]
    if missing_other:
        found.append(
            Diagnostic(
                "missing_ocr_language",
                WARNING,
                f"Tesseract language pack(s) not installed: {', '.join(missing_other)}.",
                "Install the language data or change OCR_LANG in .env.",
            )
        )
    return found


def index_diagnostics(settings: Settings, status: IndexStatus) -> list[Diagnostic]:
    """Problems that stop the app from answering (vector index)."""
    if status.error:
        return [
            Diagnostic(
                "index_unavailable",
                ERROR,
                f"The vector index could not be opened (CHROMA_MODE={settings.chroma_mode}).",
                "Check CHROMA_* settings, then run: python scripts/check_index.py",
            )
        ]
    if not status.exists:
        files = pdf_files(settings.pdf_dir)
        if not files:
            return [
                Diagnostic(
                    "missing_pdfs",
                    ERROR,
                    f"No vector index and no PDFs in {settings.pdf_dir}.",
                    "Put the law PDFs in data/ (see data/README.txt), then run: python ingest.py",
                )
            ]
        found = [
            Diagnostic(
                "missing_index",
                ERROR,
                f"The vector index has not been built yet ({len(files)} PDF(s) found in data/).",
                "python ingest.py",
            )
        ]
        if has_scanned_pdf(files):
            found += ocr_diagnostics(settings)
        return found
    if status.embedding_model and status.embedding_model != settings.embedding_model:
        return [
            Diagnostic(
                "incompatible_embedding_model",
                ERROR,
                f"The index was built with '{status.embedding_model}' but EMBEDDING_MODEL is "
                f"'{settings.embedding_model}'.",
                "python ingest.py --rebuild   (or set EMBEDDING_MODEL back to the model used for the index)",
            )
        ]
    return []


def api_key_diagnostics(api_key: str) -> list[Diagnostic]:
    if api_key:
        return []
    return [
        Diagnostic(
            "missing_api_key",
            ERROR,
            "No Groq API key is configured.",
            "Paste a key in the sidebar, or add GROQ_API_KEY=gsk_... to .env "
            "(Streamlit Cloud: App settings → Secrets). Free key: https://console.groq.com/keys",
        )
    ]


def main() -> int:  # pragma: no cover - thin CLI wrapper
    from legalbot.config import load_settings
    from legalbot.vectorstore import index_status

    settings = load_settings()
    found = (
        api_key_diagnostics(settings.groq_api_key)
        + index_diagnostics(settings, index_status(settings))
        + ocr_diagnostics(settings)
    )
    unique = list({d.code: d for d in found}.values())
    if not unique:
        print("✓ All checks passed.")
        return 0
    for diag in unique:
        print(f"[{diag.severity.upper()}] {diag.message}\n    fix: {diag.fix}")
    return 1 if any(d.severity == ERROR for d in unique) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
