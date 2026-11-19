from __future__ import annotations

import pytest

from legalbot import pdf_loader
from legalbot.pdf_loader import clean_text, load_pdfs


def test_loads_pages_recursively_and_skips_blank_and_broken(pdf_dir):
    docs = load_pdfs(pdf_dir, ocr_lang="eng")
    sources = {d.metadata["source"] for d in docs}
    assert sources == {"Penal code.pdf", "sub/Constituion.pdf"}  # blank + broken skipped
    penal = [d for d in docs if d.metadata["filename"] == "Penal code.pdf"]
    assert [d.metadata["page"] for d in penal] == [1, 2]
    assert penal[0].metadata["total_pages"] == 2
    assert "Punishment for murder" in penal[1].page_content


def test_no_pdfs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdfs(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_pdfs(tmp_path / "missing")


def test_clean_text():
    assert clean_text("a\x00b  \n\n\n\nc\x0c") == "ab\n\nc"


def test_ocr_used_for_scanned_pages(pdf_dir, monkeypatch):
    calls = []

    class FakeOcr:
        available = True

        def __init__(self, lang, cache_dir=None, workers=None):
            self.lang = lang

        def ocr_pages(self, pdf, page_numbers, file_hash, label):
            calls.append((label, list(page_numbers)))
            return {n: "OCR text recovered from a scanned page of the statute." for n in page_numbers}

        def warn_unavailable(self):
            raise AssertionError("should not be called")

    monkeypatch.setattr(pdf_loader, "_Ocr", FakeOcr)
    docs = load_pdfs(pdf_dir, ocr_lang="eng+ben")
    blank = [d for d in docs if d.metadata["filename"] == "blank.pdf"]
    assert len(blank) == 1 and "OCR text" in blank[0].page_content
    assert ("blank.pdf", [1]) in calls


def test_ocr_missing_tesseract_is_graceful(monkeypatch, pdf_dir, caplog):
    monkeypatch.setattr(pdf_loader, "find_tesseract", lambda: None)
    ocr = pdf_loader._Ocr("eng")
    assert ocr.available is False
    docs = load_pdfs(pdf_dir)
    assert "blank.pdf" not in {d.metadata["filename"] for d in docs}
    assert "Tesseract is not installed" in caplog.text


# ── layout-aware OCR reconstruction ─────────────────────────────────────────
def _ocr_data(lines):
    """Build a pytesseract image_to_data-style dict from (left, top, text) lines."""
    data = {
        k: [] for k in ("text", "conf", "block_num", "par_num", "line_num", "left", "top", "width", "height")
    }
    for block, (left, top, text) in enumerate(lines, 1):
        x = left
        for word in text.split():
            data["text"].append(word)
            data["conf"].append("90")
            data["block_num"].append(block)
            data["par_num"].append(1)
            data["line_num"].append(1)
            data["left"].append(x)
            data["top"].append(top)
            data["width"].append(len(word) * 18)
            data["height"].append(40)
            x += len(word) * 18 + 12
    data["text"].append(" ")  # empty token with conf -1 must be ignored
    data["conf"].append("-1")
    for k in ("block_num", "par_num", "line_num", "left", "top", "width", "height"):
        data[k].append(0)
    return data


def test_layout_text_reattaches_margin_notes_above_sections():
    from legalbot.pdf_loader import SIDE_NOTE_MARK, layout_text

    W = 2550
    data = _ocr_data(
        [
            (240, 400, "Punishment"),
            (640, 410, "302. Whoever commits murder shall be punished with death, or"),
            (240, 470, "for murder"),
            (640, 520, "imprisonment for life, and shall also be liable to fine."),
            (240, 900, "Culpable"),
            (640, 905, "303. Whoever being under sentence of imprisonment for life commits"),
            (240, 975, "homicide by"),
        ]
    )
    text = layout_text(data, W).splitlines()
    assert text[0] == SIDE_NOTE_MARK + "Punishment for murder"
    assert text[1].startswith("302. Whoever")
    assert text[2].startswith("imprisonment for life")
    i = text.index(SIDE_NOTE_MARK + "Culpable homicide by")
    assert text[i + 1].startswith("303.")


def test_layout_text_single_column_has_no_notes():
    from legalbot.pdf_loader import SIDE_NOTE_MARK, layout_text

    data = _ocr_data([(300, 100, "A plain single column page of text."), (300, 160, "Second line here.")])
    out = layout_text(data, 2550)
    assert SIDE_NOTE_MARK not in out
    assert out.splitlines() == ["A plain single column page of text.", "Second line here."]
    assert layout_text(_ocr_data([]), 2550) == ""


def test_strip_boilerplate_removes_repeated_headers_and_footers():
    from legalbot.pdf_loader import strip_boilerplate

    bodies = ["Whoever commits murder.", "Whoever commits theft.", "Whoever abets.", "Bail.", "Remand."]
    pages = [
        f"16/05/2026\nThe Penal Code, 1860\n{body}\nbdlaws.minlaw.gov.bd/act-print-11.html\n{i}/200"
        for i, body in enumerate(bodies, 1)
    ]
    cleaned = strip_boilerplate(pages)
    assert cleaned == bodies
    assert all("Penal Code" not in c and "bdlaws" not in c for c in cleaned)
    # a line repeated in the middle of pages is content, not boilerplate
    mid = [
        "Top unique A\n(1) Whoever\nend A",
        "Top unique B\n(1) Whoever\nend B",
        "Top C\n(1) Whoever\nend C",
    ]
    assert all("(1) Whoever" in c for c in strip_boilerplate(mid, edge_lines=1))
    assert strip_boilerplate([]) == []


def test_resolve_languages_drops_missing_packs():
    from legalbot.pdf_loader import resolve_languages

    class FakeTess:
        def __init__(self, langs):
            self.langs = langs

        def get_languages(self, config=""):
            return self.langs

    assert resolve_languages("eng+ben", FakeTess(["eng", "osd"])) == "eng"
    assert resolve_languages("eng+ben", FakeTess(["eng", "ben"])) == "eng+ben"
    assert resolve_languages("ben", FakeTess(["eng"])) == "eng"

    class Broken:
        def get_languages(self, config=""):
            raise RuntimeError

    assert resolve_languages("eng+ben", Broken()) == "eng+ben"


def test_ocr_cache_avoids_repeat_work(tmp_path, pdf_dir, monkeypatch):
    import pymupdf

    from legalbot.pdf_loader import _Ocr

    ocr = _Ocr("eng", cache_dir=tmp_path / "cache", workers=1)
    ocr.available = True
    calls = []

    def fake_image_data(png):
        calls.append(len(png))
        return {"width": 2550, "data": _ocr_data([(300, 100, "recognised text of the scanned page")])}

    monkeypatch.setattr(ocr, "image_data", fake_image_data)
    with pymupdf.open(str(pdf_dir / "blank.pdf")) as pdf:
        first = ocr.ocr_pages(pdf, [1], "hash123", "blank.pdf")
        second = ocr.ocr_pages(pdf, [1], "hash123", "blank.pdf")
    assert first == second == {1: "recognised text of the scanned page"}
    assert len(calls) == 1  # second run served from cache

    def failing(png):
        raise RuntimeError("tesseract crashed")

    monkeypatch.setattr(ocr, "image_data", failing)
    with pymupdf.open(str(pdf_dir / "blank.pdf")) as pdf:
        assert ocr.ocr_pages(pdf, [1], "otherhash", "blank.pdf") == {1: ""}
    # corrupt cache entry is ignored and re-OCR'd
    cache_file = next((tmp_path / "cache").glob("*.json"))
    cache_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(ocr, "image_data", fake_image_data)
    with pymupdf.open(str(pdf_dir / "blank.pdf")) as pdf:
        assert ocr.ocr_pages(pdf, [1], "hash123", "blank.pdf") == {1: "recognised text of the scanned page"}


def test_real_tesseract_layout_if_installed(tmp_path):
    """Runs only where Tesseract is installed: OCR a rendered two-column page."""
    import pymupdf

    from legalbot.pdf_loader import SIDE_NOTE_MARK, _Ocr, find_tesseract, load_pdf

    if find_tesseract() is None:
        pytest.skip("tesseract not installed")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 100), "Punishment", fontsize=10)
    page.insert_text((40, 113), "for murder", fontsize=10)
    page.insert_text((160, 100), "302. Whoever commits murder shall be punished with death,", fontsize=10)
    page.insert_text((160, 113), "or imprisonment for life, and shall also be liable to fine.", fontsize=10)
    pix = page.get_pixmap(dpi=200)
    scan = pymupdf.open()
    scan_page = scan.new_page(width=page.rect.width, height=page.rect.height)
    scan_page.insert_image(scan_page.rect, stream=pix.tobytes("png"))  # image-only page
    path = tmp_path / "Penal code.pdf"
    scan.save(str(path))
    docs = load_pdf(path, tmp_path, _Ocr("eng", workers=1))
    assert docs, "OCR produced no text"
    lines = docs[0].page_content.splitlines()
    assert lines[0].startswith(SIDE_NOTE_MARK) and "Punishment" in lines[0]
    assert any(line.startswith("302.") for line in lines)


def test_detect_title_from_running_header():
    from legalbot.pdf_loader import detect_title

    pages = [f"16/05/2026\nThe Penal Code, 1860\nBody {i}" for i in range(5)]
    assert detect_title(pages) == "The Penal Code, 1860"
    single = [
        "The Engineering and Technological University Ordinance,\n1961 (East Pakistan Ordinance)\nWHEREAS"
    ]
    assert detect_title(single).startswith("The Engineering and Technological University Ordinance")
    assert detect_title(["no title here"]) == ""
    assert detect_title([]) == ""


def test_margin_date_is_not_glued_to_section_title():
    from legalbot.pdf_loader import SIDE_NOTE_MARK, layout_text

    data = _ocr_data(
        [
            (90, 60, "16/05/2026"),
            (240, 120, "Safeguards"),
            (240, 185, "as to arrest"),
            (640, 130, "24[33. (1) No person who is arrested shall be detained in custody without"),
        ]
    )
    lines = layout_text(data, 2550).splitlines()
    assert SIDE_NOTE_MARK + "Safeguards as to arrest" in lines
    assert "16/05/2026" in lines  # kept as a normal line (removed later as boilerplate)
