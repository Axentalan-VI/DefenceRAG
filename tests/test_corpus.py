"""The metaData corpus must load cleanly into the reused Chunk type."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from defrag.corpus import DOCUMENTS, load_metadata_chunks  # noqa: E402

META = Path(__file__).resolve().parents[1] / "data/raw/metaData.csv"


@pytest.fixture(scope="module")
def chunks():
    return load_metadata_chunks(META)


def test_all_seven_documents_present(chunks):
    docs = {c.doc_id for c in chunks}
    assert docs == set(DOCUMENTS), f"unexpected doc set: {docs ^ set(DOCUMENTS)}"


def test_reasonable_chunk_count(chunks):
    assert 1500 <= len(chunks) <= 2000  # ~1668


def test_every_chunk_has_text_and_valid_source(chunks):
    for c in chunks:
        assert c.text.strip()
        assert c.doc_id in DOCUMENTS


def test_retrieval_text_includes_document_name(chunks):
    c = chunks[0]
    assert c.doc_id in c.retrieval_text
    assert c.text[:30] in c.retrieval_text


def test_chunk_ids_are_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
