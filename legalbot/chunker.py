"""Legal-structure-aware chunking.

Bangladesh statutes are organised as Parts → Chapters → Sections (Acts) or
Articles (Constitution). Instead of cutting text every N characters, we:

1. join a PDF's pages into one string (remembering page offsets),
2. detect section/article headings — English (``302. Whoever…``,
   ``Section 302.``, ``Article 27``) and Bangla (``ধারা ৩০২।``, ``অনুচ্ছেদ ২৭``),
3. keep the sequence monotonic so numbered lists inside a section are not
   mistaken for headings (and a table of contents followed by the body works),
4. emit one chunk per section (long sections are sub-split with overlap),
   each prefixed with a header such as
   ``Penal Code, 1860 — Section 302: Punishment for murder`` and carrying
   act / section / chapter / part / page metadata.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from legalbot.pdf_loader import SIDE_NOTE_MARK
from legalbot.text import bangla_ratio, bn_to_ascii_digits

logger = logging.getLogger(__name__)

LEGAL_SEPARATORS = ["\n\n", "\n", "। ", ". ", "; ", ", ", " ", ""]
MIN_SECTION_CHARS = 40  # drops table-of-contents entries and "[Repealed]" stubs
MAX_SECTION_STEP = 15
RESTART_PENALTY = -20.0

_NUM = r"[0-9০-৯]{1,3}[A-Z]{0,2}"
KEYWORD_HEADING = re.compile(
    rf"^[ \t]*(?P<kw>Section|Sec\.|SECTION|Article|Art\.|ARTICLE|ধারা|অনুচ্ছেদ)[ \t]*"
    rf"(?P<num>{_NUM})(?![0-9০-৯])[ \t]*(?:[.।:—–-][ \t]*(?P<rest>[^\n]*))?$",
    re.MULTILINE,
)
# Optional "[" (amended provision) with an optional footnote number before it: "24[33."
_PREFIX = r"(?:[0-9*]{1,3}[ \t]*\[|\[)?[ \t]*"
BARE_HEADING = re.compile(
    rf"^[ \t]*{_PREFIX}(?P<num>{_NUM})[ \t]*[.।][ \t]*(?P<rest>[A-Z0-9ঀ-৿(\"'“‘\[*][^\n]*)$",
    re.MULTILINE,
)
# OCR often reads a footnote + "[" as digits: "24[33." → "24733.". The section
# number is then one of the trailing 1-3 digits; the chain DP picks the one that fits.
LOOSE_HEADING = re.compile(
    r"^[ \t]*(?P<digits>[0-9]{4,6})(?P<suffix>[A-Z]{0,2})[ \t]*\.[ \t]*(?P<rest>[A-Z(\"'“‘\[][^\n]*)$",
    re.MULTILINE,
)
CHAPTER_RE = re.compile(
    r"^[ \t]*(?:CHAPTER|Chapter|অধ্যায়|অধ্যায়)[ \t]+(?:[IVXLC]+|\d+|[০-৯]+)\b[^\n]{0,100}$",
    re.MULTILINE,
)
PART_RE = re.compile(
    r"^[ \t]*(?:PART|Part|ভাগ)[ \t]+(?:[IVXLC]+|\d+|[০-৯]+)\b[^\n]{0,100}$",
    re.MULTILINE,
)
_TITLE_DASH = re.compile(r"^(?P<title>.{3,150}?)\s*(?:\.\s*[—–-]|[—–]|\.-)")

_KNOWN_ACTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"penal", re.I), "Penal Code, 1860"),
    (re.compile(r"criminal[\s_-]*procedu", re.I), "Code of Criminal Procedure, 1898"),
    (re.compile(r"constitu", re.I), "Constitution of Bangladesh"),
    (re.compile(r"muslim[\s_-]*family", re.I), "Muslim Family Laws Ordinance, 1961"),
    (
        re.compile(r"land[\s_-]*acquisition", re.I),
        "Acquisition and Requisition of Immovable Property Act, 2017",
    ),
    (re.compile(r"guardian", re.I), "Guardians and Wards Act, 1890"),
)


def _known_act(text: str) -> str | None:
    for pattern, name in _KNOWN_ACTS:
        if pattern.search(text):
            return name
    return None


@dataclass(frozen=True)
class DocumentIdentity:
    act: str
    suspicious: bool = False
    reason: str = ""


def identify_document(filename: str, doc_title: str = "") -> DocumentIdentity:
    """Act name, preferring the title printed in the document over the file name.

    A document is *suspicious* when its file name names a known statute but
    the title printed inside it is a different act (e.g. a mislabelled download).
    """
    from_name = _known_act(filename)
    title = re.sub(r"^the\s+", "", doc_title.strip(), flags=re.I).strip(" .,;:")
    if title:
        from_title = _known_act(title) or title
        if from_name and from_name != from_title:
            reason = (
                f"File name suggests '{from_name}', but the document's own title is '{title}'. "
                "Check that the right PDF was downloaded."
            )
            logger.warning("Suspicious document '%s': %s", filename, reason)
            return DocumentIdentity(from_title, suspicious=True, reason=reason)
        return DocumentIdentity(from_title)
    if from_name:
        return DocumentIdentity(from_name)
    stem = re.sub(r"[_\-]+", " ", Path(filename).stem).strip()
    return DocumentIdentity(stem.title() if stem else "Unknown document")


def act_name_for(filename: str, doc_title: str = "") -> str:
    return identify_document(filename, doc_title).act


def suspicious_reason(metadata: dict[str, object]) -> str:
    """Reason a chunk's document is suspicious ("" if not).

    Works for indexes built before the ``suspicious`` flag existed by comparing
    the file name with the stored act name.
    """
    if metadata.get("suspicious"):
        return str(metadata.get("suspicious_reason") or "Document title does not match its file name.")
    if "suspicious" in metadata:
        return ""
    filename = str(metadata.get("filename", ""))
    act = str(metadata.get("act", ""))
    expected = _known_act(filename)
    if expected and act and act != expected:
        return f"File name suggests '{expected}', but the indexed document is '{act}'."
    return ""


def unit_label(act: str, bangla: bool) -> str:
    constitution = "constitution" in act.lower()
    if bangla:
        return "অনুচ্ছেদ" if constitution else "ধারা"
    return "Article" if constitution else "Section"


@dataclass
class _Heading:
    start: int  # offset where the section text begins (title line if used)
    num: str  # ASCII section number, e.g. "302" or "304A"
    title: str


def _split_num(num: str) -> tuple[int, str]:
    m = re.match(r"(\d+)([A-Z]*)", num)
    if not m:
        return 0, ""
    return int(m.group(1)), m.group(2)


def _step_score(prev: tuple[int, str], num: tuple[int, str]) -> float:
    """Score for linking heading *prev* → *num* in a chain (0 = not allowed)."""
    pb, ps = prev
    b, s = num
    if b == pb:
        if s > ps:
            return 1.0
        # "497A. [Omitted …]" is sometimes printed before "497." — allow it.
        return 0.8 if s < ps else 0.0
    if pb < b <= pb + MAX_SECTION_STEP:
        return 1.0
    if b > pb:
        return 0.3  # forward jump (missing / repealed sections)
    if b <= 2 and pb >= 10:
        # Numbering restarts (table of contents → body). Heavily penalised so a
        # short numbered list inside a section never wins over the real sequence.
        return RESTART_PENALTY
    return 0.0


ALT_READING_COST = 0.05  # small penalty for re-interpreting OCR'd digits


def _best_chain(
    nums: list[str],
    positions: list[int] | None = None,
    costs: list[float] | None = None,
    window: int = 400,
) -> list[int]:
    """Indices of the highest-scoring plausible heading sequence (DP)."""
    n = len(nums)
    pos = positions if positions is not None else list(range(n))
    cost = costs if costs is not None else [0.0] * n
    if n == 0:
        return []
    parsed = [_split_num(x) for x in nums]
    score = [1.0 - c for c in cost]
    back = [-1] * n
    for i in range(n):
        for j in range(max(0, i - window), i):
            if pos[j] == pos[i]:
                continue  # alternative readings of the same line
            step = _step_score(parsed[j], parsed[i])
            if step != 0.0 and score[j] + step - cost[i] > score[i]:
                score[i] = score[j] + step - cost[i]
                back[i] = j
    i = max(range(n), key=lambda k: (score[k], -k))
    chain: list[int] = []
    while i != -1:
        chain.append(i)
        i = back[i]
    return chain[::-1]


def _previous_line(text: str, pos: int) -> tuple[int, str]:
    end = text.rfind("\n", 0, pos)
    if end <= 0:
        return pos, ""
    start = text.rfind("\n", 0, end) + 1
    return start, text[start:end].strip()


def _side_note_before(text: str, line_start: int) -> tuple[int, str] | None:
    """A margin-note line (from OCR) just above the heading, if any."""
    pos = line_start
    for _ in range(3):  # look back over at most 3 lines (blank lines allowed)
        prev_start, prev = _previous_line(text, pos)
        if prev_start == pos:
            return None
        if prev.startswith(SIDE_NOTE_MARK):
            return prev_start, prev.replace(SIDE_NOTE_MARK, "").strip(" .")
        if prev:
            return None
        pos = prev_start
    return None


def _title_for(text: str, line_start: int, rest: str) -> tuple[int, str]:
    rest = rest.strip()
    note = _side_note_before(text, line_start)
    if note is not None and note[1]:
        return note
    m = _TITLE_DASH.match(rest)
    if m:
        return line_start, m.group("title").strip(" .")
    prev_start, prev = _previous_line(text, line_start)
    _, before_prev = _previous_line(text, prev_start) if prev else (0, "")
    if (
        prev
        and len(prev) <= 90
        and not prev.endswith((".", ";", ",", ":", "।", " or", " and", "-", "—"))
        and re.search(r"[A-Za-zঀ-৿]{3}", prev)
        and not KEYWORD_HEADING.match(prev)
        and not BARE_HEADING.match(prev)
        and not CHAPTER_RE.match(prev)
        and not PART_RE.match(prev)
        and not CHAPTER_RE.match(before_prev)
        and not PART_RE.match(before_prev)
    ):
        return prev_start, prev.strip(" .")
    first = re.split(r"(?<=[.।])\s", rest, maxsplit=1)[0]
    return line_start, first[:80].strip(" .")


def find_headings(text: str) -> list[_Heading]:
    """Detect section/article headings in document order."""
    matches: list[tuple[int, str, str, float]] = []  # start, number, rest, cost
    for m in KEYWORD_HEADING.finditer(text):
        matches.append((m.start(), m.group("num"), m.group("rest") or "", 0.0))
    taken = {start for start, *_ in matches}
    for m in BARE_HEADING.finditer(text):
        if m.start() not in taken:
            num = m.group("num")
            matches.append((m.start(), num, m.group("rest") or "", 0.0))
            # OCR may drop the "[" of an amended provision: "11[6." → "116.".
            lead = re.match(r"[0-9]+", num)
            if lead and len(lead.group(0)) >= 2:
                run, suffix = lead.group(0), num[len(lead.group(0)) :]
                for k in range(1, len(run)):
                    tail = run[-k:]
                    if not tail.startswith("0"):
                        matches.append((m.start(), tail + suffix, m.group("rest") or "", ALT_READING_COST))
    taken = {start for start, *_ in matches}
    for m in LOOSE_HEADING.finditer(text):
        if m.start() in taken:
            continue
        digits, suffix = m.group("digits"), m.group("suffix")
        for k in (1, 2, 3):
            tail = digits[-k:]
            if not tail.startswith("0"):
                matches.append((m.start(), tail + suffix, m.group("rest") or "", ALT_READING_COST))
    matches.sort(key=lambda item: (item[0], len(item[1])))

    nums = [bn_to_ascii_digits(raw) for _, raw, _, _ in matches]
    chain = _best_chain(nums, [item[0] for item in matches], [item[3] for item in matches])
    headings: list[_Heading] = []
    for idx in chain:
        line_start, _, rest, _ = matches[idx]
        start, title = _title_for(text, line_start, rest)
        if headings and start <= headings[-1].start:
            start = line_start
        headings.append(_Heading(start=start, num=nums[idx], title=title))
    return headings


def _labels(text: str, pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for m in pattern.finditer(text):
        label = m.group(0).strip()
        nxt_end = text.find("\n", m.end() + 1)
        nxt = text[m.end() + 1 : nxt_end if nxt_end != -1 else None].strip()
        if nxt and len(nxt) <= 80 and not BARE_HEADING.match(nxt) and not KEYWORD_HEADING.match(nxt):
            label = f"{label} — {nxt}"
        found.append((m.start(), label[:120]))
    return found


def _label_at(labels: list[tuple[int, str]], pos: int) -> str:
    starts = [s for s, _ in labels]
    idx = bisect.bisect_right(starts, pos) - 1
    return labels[idx][1] if idx >= 0 else ""


def make_chunk_id(source: str, index: int, content: str) -> str:
    digest = hashlib.sha256(f"{source}|{index}|{content}".encode()).hexdigest()
    return digest[:32]


def _join_pages(pages: list[Document]) -> tuple[str, list[int], list[int]]:
    parts: list[str] = []
    offsets: list[int] = []
    numbers: list[int] = []
    pos = 0
    for page in sorted(pages, key=lambda d: int(d.metadata.get("page", 0))):
        offsets.append(pos)
        numbers.append(int(page.metadata.get("page", 0)))
        parts.append(page.page_content)
        pos += len(page.page_content) + 2
    return "\n\n".join(parts), offsets, numbers


def chunk_document_pages(
    pages: list[Document], chunk_size: int = 1200, chunk_overlap: int = 150
) -> list[Document]:
    """Chunk all pages of ONE source PDF into section-level chunks."""
    if not pages:
        return []
    meta0 = pages[0].metadata
    source = str(meta0.get("source", meta0.get("filename", "unknown")))
    filename = str(meta0.get("filename", Path(source).name))
    identity = identify_document(filename, str(meta0.get("doc_title", "")))
    act = identity.act

    text, offsets, numbers = _join_pages(pages)
    bangla = bangla_ratio(text) >= 0.3
    unit = unit_label(act, bangla)

    def page_at(pos: int) -> int:
        idx = bisect.bisect_right(offsets, max(pos, 0)) - 1
        return numbers[max(idx, 0)]

    headings = find_headings(text)
    chapters = _labels(text, CHAPTER_RE)
    parts = _labels(text, PART_RE)

    spans: list[tuple[int, int, str, str]] = []  # start, end, num, title
    if headings:
        if headings[0].start > 0:
            spans.append((0, headings[0].start, "", ""))
        for i, h in enumerate(headings):
            end = headings[i + 1].start if i + 1 < len(headings) else len(text)
            spans.append((h.start, end, h.num, h.title))
    else:
        spans.append((0, len(text), "", ""))

    splitter = RecursiveCharacterTextSplitter(
        separators=LEGAL_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks: list[Document] = []
    for start, end, num, title in spans:
        raw = text[start:end]
        lead = len(raw) - len(raw.lstrip())
        body = raw.strip()
        if len(body) < MIN_SECTION_CHARS:
            continue
        body_start = start + lead
        if num:
            header = f"{act} — {unit} {num}" + (f": {title}" if title else "")
        else:
            header = f"{act} — preliminary text"
        if len(body) <= chunk_size:
            pieces = [(0, body)]
        else:
            pieces = [
                (int(d.metadata.get("start_index", 0)), d.page_content)
                for d in splitter.create_documents([body])
            ]
        chapter = _label_at(chapters, body_start)
        part = _label_at(parts, body_start)
        for offset, piece in pieces:
            abs_start = body_start + max(offset, 0)
            body_text = piece.replace(SIDE_NOTE_MARK, "").strip()
            content = f"{header}\n{body_text}"
            index = len(chunks)
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        "chunk_id": make_chunk_id(source, index, content),
                        "chunk_index": index,
                        "source": source,
                        "filename": filename,
                        "act": act,
                        "suspicious": identity.suspicious,
                        "suspicious_reason": identity.reason,
                        "unit": unit,
                        "section": num,
                        "section_title": title,
                        "chapter": chapter,
                        "part": part,
                        "page": page_at(abs_start),
                        "page_end": page_at(abs_start + len(piece) - 1),
                    },
                )
            )
    return chunks


def group_by_source(docs: Iterable[Document]) -> OrderedDict[str, list[Document]]:
    groups: OrderedDict[str, list[Document]] = OrderedDict()
    for doc in docs:
        key = str(doc.metadata.get("source", doc.metadata.get("filename", "unknown")))
        groups.setdefault(key, []).append(doc)
    return groups


def chunk_documents(docs: list[Document], chunk_size: int = 1200, chunk_overlap: int = 150) -> list[Document]:
    """Chunk page documents from any number of PDFs."""
    chunks: list[Document] = []
    for pages in group_by_source(docs).values():
        chunks.extend(chunk_document_pages(pages, chunk_size, chunk_overlap))
    return chunks
