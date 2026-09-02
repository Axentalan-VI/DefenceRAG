"""Generation pipeline mechanics, validated with a stub generator (no GPU)."""
import csv
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from defrag.corpus import DOCUMENTS, load_metadata_chunks  # noqa: E402
from defrag.generate import (  # noqa: E402
    answer_question, build_messages, build_submission, retrieve_in_doc,
)
from defrag.index import HashingEmbedder, HybridIndex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_SECTION = re.compile(r"^section-\d+$")


@pytest.fixture(scope="module")
def index():
    chunks = load_metadata_chunks(ROOT / "data/raw/metaData.csv")
    return HybridIndex.build(chunks, HashingEmbedder())


def _stub(messages):
    # Echo the first evidence line, to prove evidence reaches the generator.
    user = messages[-1]["content"]
    line = next((l for l in user.splitlines() if l.startswith("[1]")), "answer")
    return f"Per the cited document, {line[:80]}"


def test_retrieval_is_restricted_to_the_named_document(index):
    chunks = retrieve_in_doc(index, "Navy Regulations Part II escalation", "RegsNavyII.pdf")
    assert chunks and all(c.doc_id == "RegsNavyII.pdf" for c in chunks)


def test_messages_carry_system_document_and_evidence(index):
    chunks = retrieve_in_doc(index, "DPM 2025 Volume I approval authority", "DPM-2025-VOLUME-I.pdf")
    msgs = build_messages("who approves?", "DPM-2025-VOLUME-I.pdf", chunks)
    assert msgs[0]["role"] == "system"
    assert "DPM-2025-VOLUME-I.pdf" in msgs[1]["content"]
    assert "[1]" in msgs[1]["content"]


def test_answer_question_agrees_source_with_attribution(index):
    pred, src, sec = answer_question(index, "Under DPM 2025 Volume II, what escalates?", _stub)
    assert src == "DPM-2025-VOLUME-II.pdf"
    assert _SECTION.match(sec)
    assert pred.strip()


def test_full_submission_is_schema_valid(tmp_path, index):
    out = tmp_path / "sub.csv"
    n = build_submission(ROOT / "data/raw/test.csv", index, _stub, out)
    assert n == 140
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert list(rows[0].keys()) == ["id", "prediction", "pred_source", "pred_section"]
    assert [r["id"] for r in rows] == [str(i) for i in range(1, 141)]
    for r in rows:
        assert r["pred_source"] in DOCUMENTS
        assert _SECTION.match(r["pred_section"])
        assert r["prediction"].strip()
