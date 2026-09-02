"""Source and section attribution for the submission.

`pred_source`: every one of the 140 test questions names its source document
explicitly ("Under DPM 2025 Volume I", "According to DPM 2025 Volume II",
"Navy Regulations Part III", "DFPDS Booklet 2024"), so a rule-based parser
recovers it with 140/140 coverage on the test set -- more reliable than
retrieval, which confuses Volume I/II and Navy Parts I-IV.

Pattern order matters: the more specific surface form must be tried first, so
"Volume II" is not captured by the "Volume I" rule and "Part III" is not
captured by "Part I".

`pred_section`: the `section-1..12` scheme is defined only on the Evaluation tab
and maps to nothing in the shipped data. Until it is known, `default_section`
returns a placeholder; swap in the real mapping once Step 1 resolves it.
"""
from __future__ import annotations

import re

from .corpus import DOCUMENTS

# (regex, canonical pdf) -- ordered most-specific first.
_SOURCE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), doc)
    for pat, doc in (
        (r"dpm\s*2025\s*volume\s*ii|dpm[^.]*vol[^.]*\bii\b|volume\s*2\b", "DPM-2025-VOLUME-II.pdf"),
        (r"dpm\s*2025\s*volume\s*i\b|dpm[^.]*vol[^.]*\bi\b|volume\s*1\b", "DPM-2025-VOLUME-I.pdf"),
        (r"dfpds|delegation\s+of\s+financial|financial\s+powers", "Delegation_of_Financial_Powers_Rules_2024_Booklet.pdf"),
        (r"navy\s*regulations?\s*part\s*iv|\bpart\s*4\b", "RegsNavyIV.pdf"),
        (r"navy\s*regulations?\s*part\s*iii|\bpart\s*3\b", "RegsNavyIII.pdf"),
        (r"navy\s*regulations?\s*part\s*ii|\bpart\s*2\b", "RegsNavyII.pdf"),
        (r"navy\s*regulations?\s*part\s*i\b|\bpart\s*1\b", "RegsNavyI.pdf"),
    )
)


def parse_source(question: str) -> str | None:
    """Return the source PDF named in the question, or None if none matches."""
    for pattern, doc in _SOURCE_PATTERNS:
        if pattern.search(question):
            return doc
    return None


def resolve_source(question: str, retrieved_doc: str | None = None) -> str:
    """pred_source: prefer the question's explicit mention, fall back to retrieval.

    Always returns a valid document name so every submission row is well-formed.
    """
    parsed = parse_source(question)
    if parsed is not None:
        return parsed
    if retrieved_doc in DOCUMENTS:
        return retrieved_doc
    return DOCUMENTS[0]


# TODO(step-1): replace with the real section-1..12 mapping from the Eval tab.
DEFAULT_SECTION = "section-1"


def default_section(chunk_section: str | None = None) -> str:
    return DEFAULT_SECTION
