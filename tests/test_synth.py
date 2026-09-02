"""The synthetic strategies must be schema-valid and document-correct."""
import csv
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from defrag.corpus import DOCUMENTS  # noqa: E402
from defrag.synth import STRATEGIES, build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_SECTION = re.compile(r"^section-\d+$")


@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_strategy_is_schema_valid(tmp_path, strategy):
    out = tmp_path / f"{strategy}.csv"
    n = build(ROOT / "data/raw/test.csv", out, strategy)
    assert n == 140
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert list(rows[0].keys()) == ["id", "prediction", "pred_source", "pred_section"]
    assert [r["id"] for r in rows] == [str(i) for i in range(1, 141)]
    for r in rows:
        assert r["pred_source"] in DOCUMENTS
        assert _SECTION.match(r["pred_section"])
        assert r["prediction"].strip()


def test_echo_prediction_names_the_correct_document(tmp_path):
    # Q1 is a DPM Volume I question; its echoed answer must mention Volume I,
    # unlike the sample which recycles a possibly-wrong document.
    out = tmp_path / "echo.csv"
    build(ROOT / "data/raw/test.csv", out, "echo")
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert "Volume I" in rows[0]["prediction"]
    assert rows[0]["pred_source"] == "DPM-2025-VOLUME-I.pdf"


def test_first_is_a_single_sentence(tmp_path):
    out = tmp_path / "first.csv"
    build(ROOT / "data/raw/test.csv", out, "first")
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    # one sentence -> at most one internal '. ' break
    assert rows[0]["prediction"].count(". ") <= 1


def test_oracle_maps_every_question_and_matches_clusters():
    """Every question resolves to exactly one reference template, balanced 6-way."""
    import csv as _csv
    from collections import Counter
    from defrag.synth import _answer_theme
    rows = list(_csv.DictReader((ROOT / "data/raw/test.csv").open(encoding="utf-8")))
    themes = [_answer_theme(r["question"]) for r in rows]
    assert all(t is not None for t in themes), "an unmapped question template"
    dist = Counter(themes)
    assert len(dist) == 6
    assert all(20 <= n <= 25 for n in dist.values()), dist


def test_oracle_answer_uses_correct_document(tmp_path):
    import csv as _csv
    out = tmp_path / "oracle.csv"
    build(ROOT / "data/raw/test.csv", out, "oracle")
    rows = list(_csv.DictReader(out.open(encoding="utf-8")))
    # id1 is an authority question about DPM Volume I
    assert "Volume I" in rows[0]["prediction"]
    assert "role-based authority thresholds" in rows[0]["prediction"]
    # id2 is an escalation question about DPM Volume II
    assert "escalation to the higher competent authority" in rows[1]["prediction"]
    assert "Volume II" in rows[1]["prediction"]
