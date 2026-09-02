"""A built submission must match sample_submission's schema exactly."""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from defrag.corpus import DOCUMENTS  # noqa: E402
from defrag.submit import build_submission  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_SECTION = re.compile(r"^section-\d+$")


def test_submission_matches_sample_schema(tmp_path):
    out = tmp_path / "submission.csv"
    n = build_submission(ROOT / "data/raw/test.csv", ROOT / "data/raw/metaData.csv", out)
    assert n == 140

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    sample = list(csv.DictReader((ROOT / "data/raw/sample_submission.csv").open(encoding="utf-8")))

    assert list(rows[0].keys()) == list(sample[0].keys())
    assert [r["id"] for r in rows] == [r["id"] for r in sample]
    for r in rows:
        assert r["pred_source"] in DOCUMENTS
        assert _SECTION.match(r["pred_section"])
        assert r["prediction"].strip()
