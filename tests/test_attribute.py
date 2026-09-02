"""The question->source parser must be exact and unambiguous on the test set."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from defrag.attribute import parse_source, resolve_source  # noqa: E402
from defrag.corpus import DOCUMENTS  # noqa: E402

TEST_CSV = Path(__file__).resolve().parents[1] / "data/raw/test.csv"


@pytest.fixture(scope="module")
def questions():
    return list(csv.DictReader(TEST_CSV.open(encoding="utf-8")))


def test_every_test_question_maps_to_a_document(questions):
    unmatched = [r["id"] for r in questions if parse_source(r["question"]) is None]
    assert not unmatched, f"unmatched question ids: {unmatched}"


def test_all_seven_documents_are_used(questions):
    got = {parse_source(r["question"]) for r in questions}
    assert got == set(DOCUMENTS)


@pytest.mark.parametrize("q,expected", [
    ("Under DPM 2025 Volume I, what authority approves?", "DPM-2025-VOLUME-I.pdf"),
    ("According to DPM 2025 Volume II, what happens on excess?", "DPM-2025-VOLUME-II.pdf"),
    ("How does DFPDS Booklet 2024 support compliance?", "Delegation_of_Financial_Powers_Rules_2024_Booklet.pdf"),
    ("Navy Regulations Part I governance guidance", "RegsNavyI.pdf"),
    ("Which principle in Navy Regulations Part II ...", "RegsNavyII.pdf"),
    ("Navy Regulations Part III ...", "RegsNavyIII.pdf"),
    ("Navy Regulations Part IV ...", "RegsNavyIV.pdf"),
])
def test_specific_mentions(q, expected):
    assert parse_source(q) == expected


def test_volume_two_not_shadowed_by_volume_one():
    assert parse_source("DPM 2025 Volume II") == "DPM-2025-VOLUME-II.pdf"


def test_part_three_not_shadowed_by_part_one():
    assert parse_source("Navy Regulations Part III") == "RegsNavyIII.pdf"


def test_resolve_always_returns_valid_document():
    assert resolve_source("no mention here", retrieved_doc="RegsNavyII.pdf") == "RegsNavyII.pdf"
    assert resolve_source("no mention", retrieved_doc=None) in DOCUMENTS
