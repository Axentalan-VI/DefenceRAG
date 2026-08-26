"""Load the competition corpus (metaData.csv) into reusable Chunk objects.

The documents are already chunked by the organizers, so this maps each metaData
row onto the Chunk dataclass reused from the Covered project, which lets the
hybrid index (BM25 + EmbeddingGemma) work unchanged.

metaData.csv columns -> Chunk fields:
    chunk_id  -> chunk_id           (page-anchored id, e.g. ..._p0003)
    document  -> doc_id, doc_title  (the source PDF; this is pred_source)
    section   -> section_path       (a single descriptive heading)
    text      -> text               (the passage)
    topic     -> source             (one of: regulations, procurement,
                                     financial powers)

The Chunk.retrieval_text property prepends doc_title + section to the body, so
document and heading terms are searchable -- the same trick that fixed
definitional lookups in Covered.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .chunk import Chunk

# The seven source PDFs; pred_source must be one of these exact names.
DOCUMENTS = (
    "DPM-2025-VOLUME-I.pdf",
    "DPM-2025-VOLUME-II.pdf",
    "Delegation_of_Financial_Powers_Rules_2024_Booklet.pdf",
    "RegsNavyI.pdf",
    "RegsNavyII.pdf",
    "RegsNavyIII.pdf",
    "RegsNavyIV.pdf",
)


def load_metadata_chunks(path: Path) -> list[Chunk]:
    """Parse metaData.csv into Chunk objects (multi-line text handled by csv)."""
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            document = (row.get("document") or "").strip()
            section = (row.get("section") or "").strip()
            chunks.append(
                Chunk(
                    chunk_id=(row.get("chunk_id") or f"chunk-{len(chunks)}").strip(),
                    doc_id=document,
                    doc_title=document,
                    section_path=(section,) if section else (),
                    text=text,
                    url="",
                    source=(row.get("topic") or "").strip(),
                )
            )
    return chunks


if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", type=Path, default=Path("data/raw/metaData.csv"))
    args = ap.parse_args()

    chunks = load_metadata_chunks(args.meta)
    print(f"loaded {len(chunks):,} chunks")
    by_doc = Counter(c.doc_id for c in chunks)
    for doc in DOCUMENTS:
        print(f"  {by_doc.get(doc, 0):5d}  {doc}")
    unknown = set(by_doc) - set(DOCUMENTS)
    if unknown:
        print(f"  WARNING unknown documents: {unknown}")
    lens = [len(c.text) for c in chunks]
    lens.sort()
    print(f"text chars: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")
    print("sample retrieval_text:")
    print("  " + chunks[0].retrieval_text[:160].replace("\n", " "))
