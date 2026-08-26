"""Structure-aware chunking.

A citation has to resolve to a named section of a named document, not to a
floating character offset. If the heading path is lost during chunking, a
"grounded" answer becomes unverifiable - which is the precise failure this
project exists to prevent - so section identity travels with every chunk and
``tests/test_chunk.py`` guards it.

HealthCare.gov content arrives as HTML fragments whose ``<h2>``/``<h3>``
headings are the document's real section structure. Splitting on those headings
gives chunks whose boundaries a human would recognise, rather than arbitrary
windows.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from bs4 import BeautifulSoup

HEADING_TAGS = ("h1", "h2", "h3", "h4")

# Long sections are split at paragraph boundaries; these bound the pieces.
MAX_CHARS = 1_800
MIN_CHARS = 120


@dataclass(frozen=True)
class Chunk:
    """A retrievable, citable span."""

    chunk_id: str      # "<doc_id>#<section_index>" - stable across rebuilds
    doc_id: str
    doc_title: str
    section_path: tuple[str, ...]  # heading breadcrumb, outermost first
    text: str
    url: str
    source: str        # Source.key it came from

    @property
    def section_id(self) -> str:
        """Human-readable citation target."""
        if not self.section_path:
            return self.doc_title
        return f"{self.doc_title} § {' > '.join(self.section_path)}"

    @property
    def retrieval_text(self) -> str:
        """What the retrievers index: citation context prepended to the body.

        The body alone is not enough. A glossary entry for "Federal poverty
        level (FPL)" defines it as "A measure of income updated each year by
        HHS..." - the term being defined never appears in its own body, so both
        BM25 and the embedder were blind to it and the definitive chunk did not
        make top-5 for "what is the federal poverty level".

        Titles and heading paths carry the topic; bodies carry the detail.
        Indexing both is what makes definitional lookups work.
        """
        return f"{self.section_id}\n{self.text}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["section_path"] = list(self.section_path)
        d["section_id"] = self.section_id
        return d


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _hard_wrap(text: str) -> list[str]:
    """Last-resort split for a span with no sentence boundary in it.

    Flattened bullet lists ("...must cover the following: A, B, C, ...") arrive
    as one unbroken sentence and can run well past MAX_CHARS, so falling back to
    word boundaries is what keeps the guarantee absolute rather than typical.
    """
    words = text.split(" ")
    pieces: list[str] = []
    buffer = ""
    for word in words:
        if buffer and len(buffer) + len(word) + 1 > MAX_CHARS:
            pieces.append(buffer)
            buffer = word
        else:
            buffer = f"{buffer} {word}".strip()
    if buffer:
        pieces.append(buffer)
    return pieces


def _split_long(text: str) -> list[str]:
    """Split an over-long section at sentence, then word, boundaries.

    Every returned piece is <= MAX_CHARS. tests/test_chunk.py asserts this, as a
    chunk longer than the retriever's window silently loses its tail.
    """
    if len(text) <= MAX_CHARS:
        return [text]

    pieces: list[str] = []
    buffer = ""
    # Sentence-ish boundaries: keeps a citation's supporting span readable.
    for part in re.split(r"(?<=[.!?])\s+", text):
        if buffer and len(buffer) + len(part) + 1 > MAX_CHARS:
            pieces.append(buffer.strip())
            buffer = part
        else:
            buffer = f"{buffer} {part}".strip()
    if buffer.strip():
        pieces.append(buffer.strip())

    # A single sentence can still exceed the cap; wrap those.
    wrapped: list[str] = []
    for piece in pieces:
        wrapped.extend(_hard_wrap(piece) if len(piece) > MAX_CHARS else [piece])
    return wrapped


def chunk_html(
    html: str,
    *,
    doc_id: str,
    doc_title: str,
    url: str,
    source: str,
) -> list[Chunk]:
    """Split one HTML document into section-aware chunks.

    Text before any heading belongs to the document root and gets an empty
    section path - it is still citable, by document title.
    """
    soup = BeautifulSoup(html or "", "html.parser")

    # (heading_path, [text parts]) in document order.
    sections: list[tuple[tuple[str, ...], list[str]]] = [((), [])]

    # (heading_level, heading_text), outermost first. Levels are tracked
    # explicitly rather than used as path indices: real documents start at h2
    # and skip levels, so depth-in-path and tag-level are not the same number.
    stack: list[tuple[int, str]] = []

    for element in soup.find_all(HEADING_TAGS + ("p", "li", "td", "dd", "dt")):
        text = _clean(element.get_text(" "))
        if not text:
            continue

        if element.name in HEADING_TAGS:
            level = HEADING_TAGS.index(element.name)
            # A heading closes every heading at its own level or deeper.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            sections.append((tuple(t for _, t in stack), []))
        else:
            sections[-1][1].append(text)

    chunks: list[Chunk] = []
    for section_path, parts in sections:
        body = _clean(" ".join(parts))
        if len(body) < MIN_CHARS:
            # Too short to stand alone as evidence; a heading with no body is
            # navigation, not content.
            continue
        for piece in _split_long(body):
            if len(piece) < MIN_CHARS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{len(chunks)}",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section_path=section_path,
                    text=piece,
                    url=url,
                    source=source,
                )
            )
    return chunks


def _title_from(entry: dict, fallback: str) -> str:
    for key in ("title", "name", "term"):
        value = entry.get(key)
        if value:
            return _clean(str(value))
    return fallback


def _doc_id_from_url(url: str, index: int, prefix: str) -> str:
    slug = url.strip("/").replace("/", "-") if url else ""
    return slug or f"{prefix}-{index}"


def chunk_healthcare_gov(raw_file: Path, source_key: str) -> list[Chunk]:
    """Chunk a HealthCare.gov API payload (glossary.json or articles.json).

    Both share a shape: ``{"<collection>": [ {url, title, content(HTML)}, ... ]}``.
    """
    payload = json.loads(raw_file.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        # Single top-level collection key ("glossary" / "articles").
        entries = next(
            (v for v in payload.values() if isinstance(v, list)), []
        )
    else:
        entries = payload

    chunks: list[Chunk] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "")
        doc_id = _doc_id_from_url(url, index, source_key)
        title = _title_from(entry, fallback=doc_id)
        content = entry.get("content") or entry.get("body") or ""
        chunks.extend(
            chunk_html(
                str(content),
                doc_id=doc_id,
                doc_title=title,
                url=f"https://www.healthcare.gov{url}" if url.startswith("/") else url,
                source=source_key,
            )
        )
    return chunks


def write_chunks(chunks: list[Chunk], dest: Path) -> Path:
    """Write chunks as JSONL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return dest


def read_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            d = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=d["chunk_id"],
                    doc_id=d["doc_id"],
                    doc_title=d["doc_title"],
                    section_path=tuple(d["section_path"]),
                    text=d["text"],
                    url=d["url"],
                    source=d["source"],
                )
            )
    return chunks
