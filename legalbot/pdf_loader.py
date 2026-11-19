"""Parse PDFs into page-level LangChain ``Document`` objects.

* Text PDFs: PyMuPDF text extraction.
* Scanned PDFs (the four sample statutes are scans of bdlaws.minlaw.gov.bd
  print-outs): layout-aware Tesseract OCR. Each section's title sits in a
  left margin column ("marginal note"); Tesseract reads that column
  separately, so we rebuild the page from word positions and re-insert each
  marginal note just above the line it belongs to (prefixed with
  ``SIDE_NOTE_MARK`` so the chunker can use it as the section title).
* Repeated page headers/footers (print date, act title, URL, "61/200") are removed.
* OCR results are cached on disk, so re-running ingest is fast.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

import pymupdf
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

MIN_PAGE_CHARS = 30
SIDE_NOTE_MARK = "\ue000"  # private-use character: never appears in real text
OCR_CACHE_VERSION = "v3"

_WINDOWS_TESSERACT = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def find_tesseract() -> str | None:
    """Locate a tesseract binary (TESSERACT_CMD env, PATH, Windows defaults)."""
    candidates = [os.getenv("TESSERACT_CMD"), shutil.which("tesseract"), *_WINDOWS_TESSERACT]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_BOILERPLATE_SHAPES = (
    re.compile(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$"),  # print date
    re.compile(r"^\d{1,4}\s*/\s*\d{1,4}$"),  # page counter "61/200"
    re.compile(r"^(https?://)?[\w.-]+\.(gov\.bd|com|org|net)/\S*$", re.I),  # URL
)


# ── layout reconstruction (pure functions, unit-tested) ─────────────────────
def _lines_from_ocr_data(data: dict[str, list[Any]], page_width: int) -> list[dict[str, Any]]:
    """Group OCR words into lines; split a line where a wide horizontal gap
    separates a margin note from the main text on the same baseline."""
    grouped: dict[tuple[int, int, int], list[tuple[int, int, int, int, str]]] = {}
    for i, raw in enumerate(data["text"]):
        word = str(raw).strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not word or conf < 0:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        left, top = int(data["left"][i]), int(data["top"][i])
        grouped.setdefault(key, []).append(
            (left, top, left + int(data["width"][i]), top + int(data["height"][i]), word)
        )

    max_gap = max(page_width * 0.04, 1.0)
    result: list[dict[str, Any]] = []
    for words in grouped.values():
        words.sort()
        segments: list[list[tuple[int, int, int, int, str]]] = [[words[0]]]
        for box in words[1:]:
            if box[0] - segments[-1][-1][2] > max_gap:
                segments.append([box])
            else:
                segments[-1].append(box)
        for seg in segments:
            result.append(
                {
                    "left": min(w[0] for w in seg),
                    "top": min(w[1] for w in seg),
                    "right": max(w[2] for w in seg),
                    "bottom": max(w[3] for w in seg),
                    "text": " ".join(w[4] for w in seg),
                }
            )
    return sorted(result, key=lambda ln: (ln["top"], ln["left"]))


def layout_text(data: dict[str, list[Any]], page_width: int) -> str:
    """Rebuild reading order from Tesseract word data, re-attaching margin notes."""
    lines = _lines_from_ocr_data(data, page_width)
    if not lines:
        return ""
    heights = sorted(ln["bottom"] - ln["top"] for ln in lines if ln["bottom"] > ln["top"])
    line_h = float(heights[len(heights) // 2]) if heights else 20.0

    # Main-text margin = most common left edge of long lines (1 % width buckets).
    bucket = max(page_width // 100, 1)
    lefts = [ln["left"] // bucket for ln in lines if len(ln["text"]) >= 25]
    if not lefts:
        lefts = [ln["left"] // bucket for ln in lines]
    margin = Counter(lefts).most_common(1)[0][0] * bucket
    side: list[dict[str, Any]] = []
    main: list[dict[str, Any]] = []
    for ln in lines:
        is_side = ln["right"] < margin - bucket
        # dates / page counters in the margin are page furniture, not section titles
        if is_side and not any(p.match(ln["text"].strip()) for p in _BOILERPLATE_SHAPES):
            side.append(ln)
        else:
            main.append(ln)

    # Merge consecutive margin lines ("Safeguards / as to arrest / and / detention").
    notes: list[list[Any]] = []  # [top, last_top, [texts]]
    for ln in side:
        if notes and ln["top"] - notes[-1][1] <= 2.5 * line_h:
            notes[-1][1] = ln["top"]
            notes[-1][2].append(ln["text"])
        else:
            notes.append([ln["top"], ln["top"], [ln["text"]]])

    gaps = sorted(b["top"] - a["bottom"] for a, b in itertools.pairwise(main) if b["top"] > a["bottom"])
    para_gap = 2.0 * float(gaps[len(gaps) // 2]) if gaps else 2.0 * line_h

    out: list[str] = []
    note_idx = 0
    prev_bottom: int | None = None
    for ln in main:
        while note_idx < len(notes) and notes[note_idx][0] <= ln["top"] + line_h:
            out.append(SIDE_NOTE_MARK + " ".join(notes[note_idx][2]))
            note_idx += 1
        if prev_bottom is not None and ln["top"] - prev_bottom > para_gap:
            out.append("")
        out.append(ln["text"])
        prev_bottom = ln["bottom"]
    out.extend(SIDE_NOTE_MARK + " ".join(note[2]) for note in notes[note_idx:])
    return "\n".join(out)


_TITLE_WORDS = re.compile(
    r"\b(Act|Ordinance|Code|Constitution|Order|Rules|Regulations)\b|আইন|অধ্যাদেশ|সংবিধান|বিধি"
)


def detect_title(pages: Sequence[str], edge_lines: int = 3) -> str:
    """The document's title: the repeated running header that names an act."""
    counts: Counter[str] = Counter()
    for text in pages:
        top = [ln.replace(SIDE_NOTE_MARK, "").strip() for ln in text.splitlines() if ln.strip()]
        counts.update(set(top[:edge_lines]))
    threshold = max(2, int(0.4 * len(pages)))
    for line, n in counts.most_common():
        if n >= threshold and _TITLE_WORDS.search(line) and 8 <= len(line) <= 150:
            return line
    if pages:  # single page / no running header: look at the first lines of page 1
        first = " ".join(ln.strip() for ln in pages[0].splitlines()[:4] if ln.strip())
        m = re.search(
            r"((?:The\s+)?[A-Z][\w ,'’&()\-]{3,120}?\b(?:Act|Ordinance|Code|Order)\b,?\s*(?:\d{4})?)", first
        )
        if m:
            return m.group(1).strip()
    return ""


def _normalise_line(line: str) -> str:
    bare = line.replace(SIDE_NOTE_MARK, "").strip().lower()
    # Only short lines (page counters, dates) are compared with digits masked.
    return re.sub(r"\d+", "#", bare) if len(bare) <= 25 else bare


def strip_boilerplate(pages: Sequence[str], edge_lines: int = 3) -> list[str]:
    """Remove repeated header/footer lines (first/last few lines of each page)."""
    if not pages:
        return []
    split = [p.splitlines() for p in pages]
    counts: Counter[str] = Counter()
    for lines in split:
        nonblank = [ln for ln in lines if ln.strip()]
        counts.update({_normalise_line(ln) for ln in nonblank[:edge_lines] + nonblank[-edge_lines:]})
    threshold = max(3, int(0.4 * len(pages)))
    repeated = {line for line, n in counts.items() if n >= threshold and len(line) <= 120}

    cleaned = []
    for lines in split:
        nonblank_idx = [i for i, ln in enumerate(lines) if ln.strip()]
        edge_idx = set(nonblank_idx[:edge_lines] + nonblank_idx[-edge_lines:])
        kept = []
        for i, ln in enumerate(lines):
            bare = ln.replace(SIDE_NOTE_MARK, "").strip()
            if i in edge_idx and (
                _normalise_line(ln) in repeated or any(p.match(bare) for p in _BOILERPLATE_SHAPES)
            ):
                continue
            kept.append(ln)
        cleaned.append(clean_text("\n".join(kept)))
    return cleaned


# ── OCR engine ──────────────────────────────────────────────────────────────
class _Ocr:
    def __init__(
        self,
        lang: str,
        cache_dir: str | Path | None = None,
        workers: int | None = None,
        dpi: int = 300,
    ) -> None:
        self.lang = lang
        self.dpi = dpi
        self.workers = workers or max(1, min(4, (os.cpu_count() or 2) - 1))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.available = False
        self._warned = False
        try:
            import pytesseract
        except ImportError:
            return
        cmd = find_tesseract()
        if not cmd:
            return
        pytesseract.pytesseract.tesseract_cmd = cmd
        self.lang = resolve_languages(lang, pytesseract)
        self.available = bool(self.lang)

    def warn_unavailable(self) -> None:
        if not self._warned:
            logger.warning("Scanned pages found but Tesseract is not installed — OCR skipped (see README).")
            self._warned = True

    def image_data(self, png: bytes) -> dict[str, Any]:
        """Raw Tesseract word boxes (cached, so layout logic can change without re-OCR)."""
        import pytesseract
        from PIL import Image

        with Image.open(BytesIO(png)) as img:
            gray = img.convert("L")
            data = pytesseract.image_to_data(gray, lang=self.lang, output_type=pytesseract.Output.DICT)
            return {"width": gray.width, "data": data}

    def _cache_path(self, file_hash: str, page: int) -> Path | None:
        if self.cache_dir is None:
            return None
        key = f"{OCR_CACHE_VERSION}-{file_hash}-{page}-{self.lang}-{self.dpi}"
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:40]}.json"

    def ocr_pages(self, pdf: Any, page_numbers: Iterable[int], file_hash: str, label: str) -> dict[int, str]:
        results: dict[int, str] = {}
        todo: list[int] = []
        for num in page_numbers:
            path = self._cache_path(file_hash, num)
            if path is not None and path.exists():
                try:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                    results[num] = layout_text(cached["data"], int(cached["width"]))
                    continue
                except (ValueError, KeyError, TypeError):
                    pass  # corrupt cache entry → OCR again
                todo.append(num)
            else:
                todo.append(num)
        if not todo:
            return results
        logger.info("OCR %s: %d page(s), %d worker(s)…", label, len(todo), self.workers)
        batch = self.workers * 2
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for start in range(0, len(todo), batch):
                nums = todo[start : start + batch]
                images = [
                    pdf[n - 1].get_pixmap(dpi=self.dpi, colorspace=pymupdf.csGRAY).tobytes("png")
                    for n in nums
                ]
                for num, raw in zip(nums, pool.map(self.safe_image_data, images), strict=True):
                    if raw is None:
                        results[num] = ""
                        continue
                    results[num] = layout_text(raw["data"], int(raw["width"]))
                    path = self._cache_path(file_hash, num)
                    if path is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(raw), encoding="utf-8")
                logger.info("  %s: %d/%d pages OCR'd", label, min(start + batch, len(todo)), len(todo))
        return results

    def safe_image_data(self, png: bytes) -> dict[str, Any] | None:
        try:
            return self.image_data(png)
        except Exception as exc:  # OCR is best-effort per page
            logger.warning("OCR failed for a page: %s", exc)
            return None


def resolve_languages(requested: str, pytesseract: Any) -> str:
    """Keep only installed Tesseract languages (e.g. drop 'ben' if missing)."""
    try:
        installed = set(pytesseract.get_languages(config=""))
    except Exception:
        return requested
    wanted = [code for code in requested.split("+") if code]
    usable = [code for code in wanted if code in installed]
    missing = [code for code in wanted if code not in installed]
    if missing:
        logger.warning(
            "Tesseract language pack(s) %s not installed; OCR uses %s. "
            "Install the Bengali pack ('ben') to OCR Bangla scans.",
            "+".join(missing),
            "+".join(usable) or "eng",
        )
    if not usable and "eng" in installed:
        usable = ["eng"]
    return "+".join(usable)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pdf(path: Path, root: Path, ocr: _Ocr | None = None) -> list[Document]:
    """Load one PDF into page documents. Raises on unreadable files."""
    with pymupdf.open(str(path)) as pdf:
        total_pages = len(pdf)
        texts = {num: clean_text(page.get_text("text")) for num, page in enumerate(pdf, start=1)}
        scanned = [num for num, text in texts.items() if len(text) < MIN_PAGE_CHARS]
        if scanned and ocr is not None:
            if ocr.available:
                texts.update(ocr.ocr_pages(pdf, scanned, _file_hash(path), path.name))
            else:
                ocr.warn_unavailable()

    numbers = sorted(texts)
    title = detect_title([texts[n] for n in numbers])
    cleaned = strip_boilerplate([texts[n] for n in numbers])
    docs: list[Document] = []
    for num, text in zip(numbers, cleaned, strict=True):
        if len(text.replace(SIDE_NOTE_MARK, "")) < MIN_PAGE_CHARS:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.relative_to(root).as_posix(),
                    "filename": path.name,
                    "page": num,
                    "total_pages": total_pages,
                    "doc_title": title,
                },
            )
        )
    return docs


def load_pdfs(
    pdf_dir: str | Path,
    ocr_lang: str = "eng+ben",
    *,
    ocr_cache_dir: str | Path | None = None,
    ocr_workers: int | None = None,
) -> list[Document]:
    """Recursively load every PDF in *pdf_dir* (page-level documents)."""
    root = Path(pdf_dir)
    files = sorted(root.rglob("*.pdf")) if root.exists() else []
    if not files:
        raise FileNotFoundError(
            f"No PDF files found in '{root}'. Put the law PDFs in data/ and re-run ingest."
        )
    ocr = _Ocr(ocr_lang, cache_dir=ocr_cache_dir, workers=ocr_workers)
    all_docs: list[Document] = []
    for path in files:
        try:
            pages = load_pdf(path, root, ocr)
        except Exception as exc:
            logger.error("Could not parse %s: %s", path.name, exc)
            continue
        logger.info("Loaded %s: %d text pages", path.name, len(pages))
        all_docs.extend(pages)
    return all_docs
