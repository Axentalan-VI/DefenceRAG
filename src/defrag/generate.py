"""Grounded answer generation for the `prediction` column.

Unlike Covered, the answer is free text (source/section are separate submission
columns), so no inline [id] citations are needed -- the model just answers from
the retrieved passages of the question's document, concisely and factually.

The generator is injected (`generate_fn: messages -> str`), so the whole pipeline
is testable offline with a stub; on Kaggle it is `rag.make_transformers_generator`
wrapping a loaded Gemma-4. Retrieval is filtered to the question's `pred_source`
so the evidence and the attribution agree.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

from .attribute import default_section, resolve_source
from .chunk import Chunk
from .index import HybridIndex

Generator = Callable[[list[dict]], str]

SYSTEM = (
    "You are a precise assistant for Indian defence procurement and policy "
    "questions. Answer ONLY from the provided passages of the cited document. "
    "Be concise (1-3 sentences), factual, and policy-consistent. Do not invent "
    "clause numbers. If the passages do not settle the question, answer with the "
    "most relevant governing principle they do state."
)

USER_TEMPLATE = """Document: {document}

Passages:
{evidence}

Question: {question}

Grounded answer:"""

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def format_evidence(chunks: Sequence[Chunk], max_chars: int = 2400) -> str:
    """Concatenate passage texts (with their section heading), bounded in size."""
    parts, total = [], 0
    for i, c in enumerate(chunks, 1):
        head = c.section_path[0] if c.section_path else c.doc_title
        body = _clean(c.text)
        block = f"[{i}] ({head}) {body}"
        if total + len(block) > max_chars and parts:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def build_messages(question: str, document: str, chunks: Sequence[Chunk]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(
            document=document, evidence=format_evidence(chunks), question=question)},
    ]


def retrieve_in_doc(index: HybridIndex, question: str, document: str,
                    k: int = 6, pool: int = 20) -> list[Chunk]:
    """Top-k retrieved chunks restricted to `document` (fallback: global top-k)."""
    hits = index.search(question, k=pool)
    in_doc = [h.chunk for h in hits if h.chunk.doc_id == document]
    if in_doc:
        return in_doc[:k]
    return [h.chunk for h in hits[:k]]


def answer_question(index: HybridIndex, question: str, generate_fn: Generator,
                    source: str | None = None) -> tuple[str, str, str]:
    """Return (prediction, pred_source, pred_section) for one question."""
    document = source or resolve_source(question)
    chunks = retrieve_in_doc(index, question, document)
    raw = generate_fn(build_messages(question, document, chunks)) if chunks else ""
    prediction = _clean(raw) or "The passages do not directly address this question."
    section = default_section(chunks[0].section_path[0] if chunks and chunks[0].section_path else None)
    return prediction, document, section


def build_submission(test_csv: Path, index: HybridIndex, generate_fn: Generator,
                     out_csv: Path) -> int:
    import csv

    rows = list(csv.DictReader(test_csv.open(encoding="utf-8")))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prediction", "pred_source", "pred_section"])
        for r in rows:
            pred, src, sec = answer_question(index, r["question"], generate_fn)
            writer.writerow([r["id"], pred, src, sec])
    return len(rows)
