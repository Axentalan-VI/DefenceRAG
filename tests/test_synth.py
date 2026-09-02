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
