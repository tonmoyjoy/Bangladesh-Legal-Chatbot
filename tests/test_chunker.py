from __future__ import annotations

from legalbot.chunker import (
    act_name_for,
    chunk_document_pages,
    chunk_documents,
    find_headings,
    make_chunk_id,
    unit_label,
)
from tests.conftest import make_pages


def _by_section(chunks):
    return {c.metadata["section"]: c for c in chunks}


def test_act_names_from_filenames():
    assert act_name_for("Penal code.pdf") == "Penal Code, 1860"
    assert act_name_for("code of criminal procedur.pdf") == "Code of Criminal Procedure, 1898"
    assert act_name_for("Constituion.pdf") == "Constitution of Bangladesh"
    assert act_name_for("Muslim Family Law ordinance.pdf") == "Muslim Family Laws Ordinance, 1961"
    assert act_name_for("some_new-act.pdf") == "Some New Act"


def test_unit_labels():
    assert unit_label("Constitution of Bangladesh", bangla=False) == "Article"
    assert unit_label("Penal Code, 1860", bangla=False) == "Section"
    assert unit_label("Penal Code, 1860", bangla=True) == "ধারা"
    assert unit_label("Constitution of Bangladesh", bangla=True) == "অনুচ্ছেদ"


def test_sections_detected_with_titles_chapters_and_pages(penal_pages):
    chunks = chunk_document_pages(penal_pages, chunk_size=600, chunk_overlap=60)
    sections = _by_section(chunks)
    assert {"299", "300", "302", "378", "379"} <= set(sections)

    murder = sections["302"]
    assert murder.metadata["section_title"] == "Punishment for murder"
    assert murder.metadata["page"] == 2
    assert murder.metadata["act"] == "Penal Code, 1860"
    assert murder.metadata["chapter"].startswith("CHAPTER XVI")
    assert murder.page_content.startswith("Penal Code, 1860 — Section 302: Punishment for murder")
    assert "imprisonment for life" in murder.page_content

    theft = sections["378"]
    assert theft.metadata["section_title"] == "Theft"
    assert theft.metadata["chapter"].startswith("CHAPTER XVII")
    assert sections["379"].metadata["section_title"] == "Punishment for theft"


def test_numbered_list_inside_section_is_not_a_heading(penal_pages):
    chunks = chunk_document_pages(penal_pages, chunk_size=600, chunk_overlap=60)
    assert "1" not in _by_section(chunks)
    murder = _by_section(chunks)["302"]
    assert "numbered list item" in murder.page_content


def test_constitution_articles_and_part(constitution_pages):
    chunks = chunk_document_pages(constitution_pages, chunk_size=600, chunk_overlap=60)
    sections = _by_section(chunks)
    assert sections["27"].metadata["unit"] == "Article"
    assert sections["27"].metadata["part"].startswith("PART III")
    assert sections["33"].metadata["section_title"] == "Safeguards as to arrest and detention"
    # Part caption must not be mistaken for the article title.
    assert sections["27"].metadata["section_title"] != "FUNDAMENTAL RIGHTS"


def test_bangla_sections_normalised_to_ascii_numbers(bangla_pages):
    chunks = chunk_document_pages(bangla_pages, chunk_size=600, chunk_overlap=60)
    sections = _by_section(chunks)
    assert set(sections) >= {"302", "379"}
    assert sections["302"].metadata["unit"] == "ধারা"
    assert sections["302"].metadata["section_title"] == "খুনের শাস্তি"
    assert sections["302"].metadata["chapter"].startswith("অধ্যায়")


def test_table_of_contents_then_body():
    toc = "\n".join(f"{i}. Title {i}." for i in range(1, 30))
    body = "\n".join(
        f"{i}. Body of section {i}, long enough to be kept as a real section of this act."
        for i in range(1, 30)
    )
    headings = find_headings(toc + "\n\n" + body)
    nums = [h.num for h in headings]
    assert nums[-29:] == [str(i) for i in range(1, 30)]
    chunks = chunk_document_pages(make_pages([toc + "\n\n" + body], "Some Act.pdf"), 1200, 100)
    # TOC entries are shorter than MIN_SECTION_CHARS and are dropped.
    assert all("Body of section" in c.page_content for c in chunks if c.metadata["section"])
    assert len([c for c in chunks if c.metadata["section"]]) == 29


def test_lettered_sections_follow_their_base_number():
    text = (
        "304. Punishment for culpable homicide not amounting to murder, whoever commits it.\n"
        "304A. Causing death by negligence.—Whoever causes the death of any person by doing a rash act.\n"
        "304B. Dowry death.—Where the death of a woman is caused by burns within seven years.\n"
        "305. Abetment of suicide of child or insane person, if any person under eighteen."
    )
    nums = [h.num for h in find_headings(text)]
    assert nums == ["304", "304A", "304B", "305"]


def test_long_section_is_split_with_header_and_overlap():
    long_body = "302. Punishment for murder.—" + " ".join(
        f"Sentence number {i} explaining the law in detail." for i in range(120)
    )
    chunks = chunk_document_pages(make_pages([long_body], "Penal code.pdf"), chunk_size=400, chunk_overlap=80)
    section_chunks = [c for c in chunks if c.metadata["section"] == "302"]
    assert len(section_chunks) > 3
    assert all(c.page_content.startswith("Penal Code, 1860 — Section 302") for c in section_chunks)
    assert all(len(c.page_content) <= 400 + 80 for c in section_chunks)


def test_chunk_ids_are_deterministic_and_unique(all_chunks, penal_pages, constitution_pages, bangla_pages):
    again = chunk_documents(penal_pages + constitution_pages + bangla_pages, chunk_size=600, chunk_overlap=60)
    assert [c.metadata["chunk_id"] for c in all_chunks] == [c.metadata["chunk_id"] for c in again]
    ids = [c.metadata["chunk_id"] for c in all_chunks]
    assert len(ids) == len(set(ids))
    assert make_chunk_id("a", 0, "x") != make_chunk_id("a", 0, "y")


def test_document_without_headings_falls_back_to_plain_chunks():
    text = "This preamble has no numbered sections at all. " * 40
    chunks = chunk_document_pages(make_pages([text], "Notes.pdf"), chunk_size=500, chunk_overlap=50)
    assert chunks and all(c.metadata["section"] == "" for c in chunks)
    assert chunks[0].page_content.startswith("Notes — preliminary text")


def test_empty_input():
    assert chunk_documents([]) == []


def test_side_note_title_from_ocr_and_marker_stripped():
    from legalbot.pdf_loader import SIDE_NOTE_MARK

    text = (
        f"{SIDE_NOTE_MARK}Punishment of bribery\n"
        "171E. Whoever commits the offence of bribery shall be punished with imprisonment.\n\n"
        f"{SIDE_NOTE_MARK}Bribery\n"
        "171F. (1) Whoever gives a gratification to any person with the object of inducing him."
    )
    chunks = chunk_document_pages(make_pages([text], "Penal code.pdf"), 1200, 100)
    sections = _by_section(chunks)
    assert sections["171E"].metadata["section_title"] == "Punishment of bribery"
    assert sections["171F"].metadata["section_title"] == "Bribery"
    assert all(SIDE_NOTE_MARK not in c.page_content for c in chunks)
    assert "Punishment of bribery\n171E. Whoever" in sections["171E"].page_content


def test_ocr_footnote_bracket_variants():
    text = (
        "31. To enjoy the protection of the law is the inalienable right of every citizen.\n"
        "32. No person shall be deprived of life or personal liberty save in accordance with law.\n"
        "24733. (1) No person who is arrested shall be detained in custody without being informed.\n"
        "25[34. (1) All forms of forced labour are prohibited and any contravention shall be an offence.\n"
        "[35. (1) No person shall be convicted of any offence except for violation of a law in force."
    )
    nums = [h.num for h in find_headings(text)]
    assert nums == ["31", "32", "33", "34", "35"]


def test_act_name_prefers_document_title_and_warns_on_mismatch(caplog):
    assert act_name_for("Penal code.pdf", "The Penal Code, 1860") == "Penal Code, 1860"
    assert (
        act_name_for("x.pdf", "The Constitution of the People's Republic of Bangladesh")
        == "Constitution of Bangladesh"
    )
    wrong = act_name_for(
        "Muslim Family Law ordinance.pdf",
        "The Engineering and Technological University Ordinance, 1961 (East Pakistan Ordinance)",
    )
    assert wrong.startswith("Engineering and Technological University Ordinance, 1961")
    assert "Check that the right PDF was downloaded" in caplog.text


def test_dropped_bracket_digits_are_reinterpreted_but_literal_numbers_win():
    text = (
        "5. Nothing in this article shall prevent the State from making special provision.\n"
        "116. (1) Every citizen shall have the right to move freely throughout Bangladesh.\n"
        "7. (1) All powers in the Republic belong to the people, and their exercise.\n"
        "18. Whoever does something that is long enough to count as a real section here.\n"
        "19. Whoever wages war against Bangladesh shall be punished with death here."
    )
    # "116" is read as article 6 (dropped "["), but "18"/"19" keep their literal reading
    assert [h.num for h in find_headings(text)] == ["5", "6", "7", "18", "19"]
