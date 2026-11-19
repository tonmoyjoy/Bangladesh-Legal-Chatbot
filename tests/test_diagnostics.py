from __future__ import annotations

from legalbot import diagnostics, pdf_loader
from legalbot.config import load_settings
from legalbot.diagnostics import ERROR, WARNING, api_key_diagnostics, index_diagnostics, ocr_diagnostics
from legalbot.vectorstore import IndexStatus


def codes(found):
    return [(d.code, d.severity) for d in found]


def test_missing_api_key():
    found = api_key_diagnostics("")
    assert codes(found) == [("missing_api_key", ERROR)]
    assert "GROQ_API_KEY" in found[0].fix
    assert api_key_diagnostics("gsk_x") == []


def test_missing_pdfs_vs_missing_index(tmp_path, pdf_dir, monkeypatch):
    monkeypatch.setattr(diagnostics, "ocr_diagnostics", lambda s: [])
    empty = load_settings(env={"PDF_DIR": str(tmp_path / "none")})
    found = index_diagnostics(empty, IndexStatus(exists=False))
    assert codes(found) == [("missing_pdfs", ERROR)]
    assert "python ingest.py" in found[0].fix

    with_pdfs = load_settings(env={"PDF_DIR": str(pdf_dir)})
    found = index_diagnostics(with_pdfs, IndexStatus(exists=False))
    assert codes(found) == [("missing_index", ERROR)]
    assert found[0].fix == "python ingest.py"


def test_missing_index_with_scanned_pdfs_adds_ocr_checks(pdf_dir, monkeypatch):
    monkeypatch.setattr(pdf_loader, "find_tesseract", lambda: None)
    settings = load_settings(env={"PDF_DIR": str(pdf_dir)})  # pdf_dir contains an image-only page
    found = index_diagnostics(settings, IndexStatus(exists=False))
    assert ("missing_tesseract", ERROR) in codes(found)


def test_incompatible_embedding_model_and_unavailable_index():
    settings = load_settings(env={})
    found = index_diagnostics(settings, IndexStatus(exists=True, count=5, embedding_model="old/model"))
    assert codes(found) == [("incompatible_embedding_model", ERROR)]
    assert "python ingest.py --rebuild" in found[0].fix
    assert (
        index_diagnostics(
            settings, IndexStatus(exists=True, count=5, embedding_model=settings.embedding_model)
        )
        == []
    )
    assert codes(index_diagnostics(settings, IndexStatus(exists=False, error="boom"))) == [
        ("index_unavailable", ERROR)
    ]


class FakeTesseract:
    class pytesseract:
        tesseract_cmd = ""

    def __init__(self, langs):
        self.langs = langs

    def get_languages(self, config=""):
        return self.langs


def test_bengali_ocr_pack_missing_is_a_visible_warning(monkeypatch):
    import sys

    monkeypatch.setattr(pdf_loader, "find_tesseract", lambda: "/usr/bin/tesseract")
    monkeypatch.setitem(sys.modules, "pytesseract", FakeTesseract(["eng", "osd"]))
    found = ocr_diagnostics(load_settings(env={"OCR_LANG": "eng+ben"}))
    assert codes(found) == [("missing_bengali_ocr", WARNING)]
    assert "ben.traineddata" in found[0].message
    assert "tessdata" in found[0].fix

    monkeypatch.setitem(sys.modules, "pytesseract", FakeTesseract(["eng", "ben"]))
    assert ocr_diagnostics(load_settings(env={"OCR_LANG": "eng+ben"})) == []
    assert ocr_diagnostics(load_settings(env={"OCR_LANG": "eng"})) == []


def test_missing_tesseract(monkeypatch):
    monkeypatch.setattr(pdf_loader, "find_tesseract", lambda: None)
    found = ocr_diagnostics(load_settings(env={}))
    assert codes(found) == [("missing_tesseract", ERROR)]
    assert "UB-Mannheim" in found[0].fix
