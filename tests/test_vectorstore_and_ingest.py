from __future__ import annotations

from dataclasses import replace

import pytest

import ingest
from legalbot.config import load_settings
from legalbot.vectorstore import (
    IncompatibleIndexError,
    IndexNotFoundError,
    VectorStore,
    index_status,
    make_client,
)
from tests.conftest import FakeEmbedder


@pytest.fixture
def settings(tmp_path):
    return load_settings(
        env={"CHROMA_DIR": str(tmp_path / "db"), "PDF_DIR": str(tmp_path / "data"), "RERANKER_MODEL": ""}
    )


def test_upsert_query_and_read_back(settings, all_chunks, fake_embedder):
    client = make_client(settings)
    store = VectorStore.create(client, "law", fake_embedder)
    store.upsert(all_chunks)
    assert store.count() == len(all_chunks)

    hits = store.similarity_search("punishment for theft imprisonment three years", k=3)
    assert hits and hits[0][0].metadata["section"] == "379"
    assert hits[0][0].metadata["chunk_id"]

    docs = store.all_documents()
    assert {d.metadata["chunk_id"] for d in docs} == {c.metadata["chunk_id"] for c in all_chunks}


def test_reingest_is_idempotent_and_removes_stale(settings, all_chunks, fake_embedder):
    client = make_client(settings)
    store = VectorStore.create(client, "law", fake_embedder)
    store.upsert(all_chunks)
    store.upsert(all_chunks)  # same IDs → no duplicates
    assert store.count() == len(all_chunks)

    penal = [c for c in all_chunks if c.metadata["source"] == "Penal code.pdf"]
    kept = penal[:2]
    removed = store.remove_stale("Penal code.pdf", [c.metadata["chunk_id"] for c in kept])
    assert removed == len(penal) - 2
    assert store.count() == len(all_chunks) - removed


def test_rebuild_wipes_collection(settings, all_chunks, fake_embedder):
    client = make_client(settings)
    VectorStore.create(client, "law", fake_embedder).upsert(all_chunks)
    fresh = VectorStore.create(client, "law", fake_embedder, rebuild=True)
    assert fresh.count() == 0


def test_open_errors(settings, all_chunks, fake_embedder):
    client = make_client(settings)
    with pytest.raises(IndexNotFoundError):
        VectorStore.open(client, "missing", fake_embedder)
    VectorStore.create(client, "empty", fake_embedder)
    with pytest.raises(IndexNotFoundError):
        VectorStore.open(client, "empty", fake_embedder)
    VectorStore.create(client, "law", fake_embedder).upsert(all_chunks)
    with pytest.raises(IncompatibleIndexError):
        VectorStore.open(client, "law", FakeEmbedder(name="other-model"))
    with pytest.raises(IncompatibleIndexError):
        VectorStore.create(client, "law", FakeEmbedder(name="other-model"))
    assert VectorStore.open(client, "law", fake_embedder).count() == len(all_chunks)


def test_index_status(settings, all_chunks, fake_embedder):
    assert index_status(settings).exists is False
    client = make_client(settings)
    VectorStore.create(client, settings.collection_name, fake_embedder).upsert(all_chunks)
    status = index_status(settings)
    assert status.exists and status.count == len(all_chunks)
    assert status.embedding_model == "fake-embedder"
    assert index_status(replace(settings, collection_name="nope")).exists is False


# ── ingest CLI ──────────────────────────────────────────────────────────────
@pytest.fixture
def ingest_env(monkeypatch, tmp_path, pdf_dir):
    monkeypatch.setenv("PDF_DIR", str(pdf_dir))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "db"))
    monkeypatch.setenv("OCR_CACHE_DIR", str(tmp_path / "ocr-cache"))  # never write into the repo
    monkeypatch.setenv("CHROMA_MODE", "local")
    monkeypatch.setenv("COLLECTION_NAME", "law_test")
    monkeypatch.setenv("CHUNK_SIZE", "600")
    monkeypatch.setenv("CHUNK_OVERLAP", "60")
    embedder = FakeEmbedder(name="intfloat/multilingual-e5-base")
    monkeypatch.setenv("EMBEDDING_MODEL", embedder.name)
    monkeypatch.setattr(ingest, "build_embedder", lambda *a, **k: embedder)
    monkeypatch.setattr(ingest, "ocr_diagnostics", lambda settings: [])  # same result on any machine
    return load_settings()


def _count(settings):
    return index_status(settings).count


def test_ingest_cli_build_incremental_and_rebuild(ingest_env, capsys):
    assert ingest.main([]) == 0
    first = _count(ingest_env)
    assert first > 0
    assert ingest.main([]) == 0  # incremental run: same content → same size
    assert _count(ingest_env) == first
    assert ingest.main(["--rebuild"]) == 0
    assert _count(ingest_env) == first


def test_ingest_dry_run_writes_nothing(ingest_env, capsys):
    assert ingest.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Penal Code, 1860" in out and "Dry run" in out
    assert _count(ingest_env) == 0


def test_ingest_missing_pdfs(ingest_env, tmp_path):
    assert ingest.main(["--pdf-dir", str(tmp_path / "empty")]) == 1


def test_ingest_bad_config(monkeypatch):
    monkeypatch.setenv("CHROMA_MODE", "bogus")
    assert ingest.main([]) == 2


def test_legacy_index_without_model_metadata_is_rejected(settings, fake_embedder):
    client = make_client(settings)
    legacy = client.get_or_create_collection(name="law", embedding_function=None)
    legacy.add(ids=["x"], documents=["old chunk"], embeddings=[[0.1] * 8])
    with pytest.raises(IncompatibleIndexError, match="older version"):
        VectorStore.open(client, "law", fake_embedder)
    with pytest.raises(IncompatibleIndexError, match="--rebuild"):
        VectorStore.create(client, "law", fake_embedder)
    fresh = VectorStore.create(client, "law", fake_embedder, rebuild=True)
    assert fresh.count() == 0


def test_legacy_index_opens_with_legacy_model_and_matching_dimension(settings):
    client = make_client(settings)
    legacy_embedder = FakeEmbedder(name="sentence-transformers/all-MiniLM-L6-v2")
    dim = len(legacy_embedder.embed_query("probe"))
    legacy = client.get_or_create_collection(name="law", embedding_function=None)
    legacy.add(ids=["x"], documents=["old chunk"], embeddings=[[0.1] * dim])
    assert VectorStore.open(client, "law", legacy_embedder).count() == 1

    wrong_dim = client.get_or_create_collection(name="law2", embedding_function=None)
    wrong_dim.add(ids=["x"], documents=["old chunk"], embeddings=[[0.1] * (dim + 1)])
    with pytest.raises(IncompatibleIndexError, match="older version"):
        VectorStore.open(client, "law2", legacy_embedder)


# ── corpus integrity: the mislabelled "Muslim Family Law ordinance.pdf" ─────
WRONG_TITLE = "The Engineering and Technological University Ordinance, 1961"


@pytest.fixture
def mislabelled_pdf(pdf_dir):
    from tests.conftest import write_pdf

    pages = [
        f"{WRONG_TITLE}\n{i}. The Syndicate of the University shall consist of the Vice-Chancellor "
        "and members appointed under this Ordinance, who shall hold office for three years."
        for i in range(1, 4)
    ]
    write_pdf(pdf_dir / "Muslim Family Law ordinance.pdf", pages)
    return pdf_dir


def test_ingest_refuses_known_mismatched_statute(ingest_env, mislabelled_pdf, capsys):
    assert ingest.main([]) == 3
    out = capsys.readouterr().out
    assert "SUSPICIOUS: Muslim Family Law ordinance.pdf" in out
    assert "Engineering and Technological University Ordinance" in out
    assert "--allow-suspicious" in out
    assert _count(ingest_env) == 0  # nothing written


def test_dry_run_reports_suspicious_without_failing(ingest_env, mislabelled_pdf, capsys):
    assert ingest.main(["--dry-run"]) == 0
    assert "SUSPICIOUS" in capsys.readouterr().out


@pytest.mark.parametrize("how", ["flag", "env"])
def test_override_indexes_and_flags_suspicious_chunks(ingest_env, mislabelled_pdf, monkeypatch, how):
    from legalbot import services

    if how == "flag":
        assert ingest.main(["--allow-suspicious"]) == 0
    else:
        monkeypatch.setenv("ALLOW_SUSPICIOUS_DOCS", "true")
        assert ingest.main([]) == 0
    store = VectorStore.open(
        make_client(ingest_env),
        ingest_env.collection_name,
        FakeEmbedder(name="intfloat/multilingual-e5-base"),
    )
    docs = store.all_documents()
    flagged = {d.metadata["filename"] for d in docs if d.metadata.get("suspicious")}
    assert flagged == {"Muslim Family Law ordinance.pdf"}
    found = services.find_suspicious(docs)
    assert list(found) == ["Muslim Family Law ordinance.pdf"]
    assert "Engineering and Technological University Ordinance" in found["Muslim Family Law ordinance.pdf"]


def test_legacy_index_without_flag_is_still_detected():
    from langchain_core.documents import Document

    from legalbot import services

    legacy = [
        Document(
            page_content="x",
            metadata={
                "filename": "Muslim Family Law ordinance.pdf",
                "act": "Engineering and Technological University Ordinance, 1961",
            },
        ),
        Document(page_content="y", metadata={"filename": "Penal code.pdf", "act": "Penal Code, 1860"}),
    ]
    assert list(services.find_suspicious(legacy)) == ["Muslim Family Law ordinance.pdf"]


def test_ingest_stops_when_scanned_pdfs_need_missing_tesseract(ingest_env, monkeypatch, caplog):
    from legalbot.diagnostics import ERROR, Diagnostic

    monkeypatch.setattr(
        ingest,
        "ocr_diagnostics",
        lambda settings: [
            Diagnostic("missing_tesseract", ERROR, "Tesseract OCR is not installed", "install it")
        ],
    )
    assert ingest.main([]) == 1
    assert "Tesseract OCR is not installed" in caplog.text
    assert _count(ingest_env) == 0
