"""Build submission.csv from test.csv.

Baseline pipeline (Day 1), no LLM yet:
  pred_source  -- rule-based question->document parser (140/140 on test).
  supporting   -- best retrieved chunk *within* that document.
  prediction   -- extractive snippet from the supporting chunk (placeholder for
                  the Gemma-generated grounded answer in Day 2).
  pred_section -- placeholder until the section-1..12 scheme is known (Step 1).

The submission columns and row order exactly match sample_submission.csv.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from .attribute import default_section, resolve_source
from .corpus import load_metadata_chunks
from .index import HashingEmbedder, HybridIndex

_SENT = re.compile(r"(?<=[.!?])\s+")


def _extractive_answer(text: str, max_chars: int = 320) -> str:
    """A clean 1-2 sentence snippet from a chunk, as a baseline prediction."""
    clean = re.sub(r"\s+", " ", text).strip()
    out = ""
    for sentence in _SENT.split(clean):
        if len(out) + len(sentence) + 1 > max_chars and out:
            break
        out = f"{out} {sentence}".strip()
    return out or clean[:max_chars]


def best_chunk_in_doc(index: HybridIndex, question: str, document: str, k: int = 12):
    """Top retrieved chunk whose doc matches `document`; fall back to global top."""
    hits = index.search(question, k=k)
    for hit in hits:
        if hit.chunk.doc_id == document:
            return hit.chunk
    return hits[0].chunk if hits else None


def build_submission(test_csv: Path, meta_csv: Path, out_csv: Path) -> int:
    chunks = load_metadata_chunks(meta_csv)
    index = HybridIndex.build(chunks, HashingEmbedder())

    rows = list(csv.DictReader(test_csv.open(encoding="utf-8")))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prediction", "pred_source", "pred_section"])
        for r in rows:
            q = r["question"]
            source = resolve_source(q)
            chunk = best_chunk_in_doc(index, q, source)
            prediction = _extractive_answer(chunk.text) if chunk else "No supporting passage found."
            section = default_section(chunk.section_path[0] if chunk and chunk.section_path else None)
            writer.writerow([r["id"], prediction, source, section])
    return len(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", type=Path, default=Path("data/raw/test.csv"))
    ap.add_argument("--meta", type=Path, default=Path("data/raw/metaData.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/submission.csv"))
    args = ap.parse_args()
    n = build_submission(args.test, args.meta, args.out)
    print(f"wrote {args.out} ({n} rows)")
