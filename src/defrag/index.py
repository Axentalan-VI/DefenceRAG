"""Hybrid retrieval: BM25 (lexical) + EmbeddingGemma (dense), fused with RRF.

Both halves earn their place. Coverage questions mix rare exact tokens that
lexical search nails - plan codes, statute numbers, "Form 1095-A" - with
consumer paraphrase that only dense search resolves ("can I stay on my parents'
plan" -> "dependent coverage to age 26"). Either alone loses a class of query.

Fusion is Reciprocal Rank Fusion: it combines *ranks*, not scores, so a BM25
score and a cosine similarity never have to be forced onto a common scale.

The index is built once by scripts/build_index.py and published as a Kaggle
Dataset, so retrieval needs no network when judges run the demo notebook.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .chunk import Chunk, read_chunks

RRF_K = 60  # standard damping constant; larger = flatter rank weighting


def check_embeddings(vectors: np.ndarray) -> np.ndarray:
    """Reject degenerate embeddings at the point they are produced.

    A NaN vector normalises, saves and sorts without complaint, so a silently
    broken embedder yields a broken index that still looks healthy - and BM25
    hides it by carrying the results on its own. Failing loudly here is what
    turns that into a five-second diagnosis instead of a quality mystery.
    """
    if vectors.size == 0:
        return vectors

    if not np.isfinite(vectors).all():
        bad = int((~np.isfinite(vectors)).any(axis=1).sum())
        raise ValueError(
            f"embedder produced {bad}/{len(vectors)} non-finite vectors. "
            "Gemma overflows in fp16 - use fp32 or bf16, not float16."
        )

    norms = np.linalg.norm(vectors, axis=1)
    if (norms < 1e-6).any():
        raise ValueError(
            f"embedder produced {int((norms < 1e-6).sum())} zero-length vectors"
        )
    return vectors


class Embedder(Protocol):
    """Anything that turns text into unit-norm vectors."""

    dim: int

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic, model-free embedder for tests and CI.

    Not semantically meaningful - it exists so the index, fusion and retrieval
    logic can be tested without downloading a model or touching a GPU.
    """

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def _bucket(self, token: str) -> int:
        # zlib.crc32, not the builtin hash(): Python salts string hashing per
        # process, so hash() would put a token in a different bucket on every
        # run and make anything built on this embedder intermittently flaky.
        import zlib

        return zlib.crc32(f"{self.seed}:{token}".encode()) % self.dim

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                out[row, self._bucket(token)] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)


class EmbeddingGemmaEmbedder:
    """EmbeddingGemma via sentence-transformers.

    EmbeddingGemma is trained with asymmetric task prefixes: queries and
    documents are embedded differently. Using the wrong side measurably degrades
    retrieval, so the distinction is threaded through ``is_query`` rather than
    left to the caller to remember.
    """

    def __init__(self, model_path: str, device: str | None = None, fp16: bool = False):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path, device=device)

        # fp16 defaults OFF, and that is not a performance choice.
        #
        # Gemma models are trained in bf16 and carry activation magnitudes that
        # exceed fp16's ~65504 ceiling. Running EmbeddingGemma in fp16 does not
        # error - it silently returns all-NaN vectors. A NaN embedding still
        # normalises, still saves, and still sorts, so a fully broken index
        # looks exactly like a working one until retrieval quality is measured.
        # This bit us: an all-NaN index still "passed" 4/5 smoke probes, because
        # BM25 was carrying the whole result set.
        #
        # bf16 is the safe half-precision for Gemma. On Turing (T4) it has no
        # tensor-core support and falls back to fp32 compute - slower, but
        # correct, which is the right trade for a value you cannot eyeball.
        if fp16:
            self.model = self.model.half()

        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        encode_fn = (
            self.model.encode_query if is_query else self.model.encode_document
        )
        vectors = encode_fn(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return check_embeddings(np.asarray(vectors, dtype=np.float32))


@dataclass
class Hit:
    """One retrieved chunk, with why it surfaced."""

    chunk: Chunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None

    def as_evidence(self) -> str:
        """Render for a model prompt, citation attached to the span itself."""
        return f"[{self.chunk.chunk_id}] {self.chunk.section_id}\n{self.chunk.text}"


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


class HybridIndex:
    """BM25 + dense over a fixed chunk set."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder | None = None):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunk/embedding mismatch: {len(chunks)} vs {embeddings.shape[0]}"
            )
        self.chunks = chunks
        # Also guards a stale index loaded from disk, not just a fresh build.
        self.embeddings = check_embeddings(embeddings)
        self.embedder = embedder
        # BM25Okapi divides by corpus size at construction, so it cannot be
        # built over an empty corpus. search() short-circuits that case anyway.
        # retrieval_text, not text: the title/heading carries the topic.
        self.bm25 = (
            BM25Okapi([_tokenize(c.retrieval_text) for c in chunks]) if chunks else None
        )

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedder: Embedder,
        batch_size: int = 64,
        progress: bool = False,
    ) -> "HybridIndex":
        import time

        vectors = []
        started = time.time()
        for start in range(0, len(chunks), batch_size):
            batch = [c.retrieval_text for c in chunks[start : start + batch_size]]
            vectors.append(embedder.encode(batch, is_query=False))
            if progress:
                done = min(start + batch_size, len(chunks))
                elapsed = time.time() - started
                rate = done / max(elapsed, 1e-6)
                eta = (len(chunks) - done) / max(rate, 1e-6)
                print(
                    f"  {done:>6,}/{len(chunks):,}  {rate:5.1f} chunks/s  eta {eta/60:4.1f} min",
                    end="\r", flush=True,
                )
        if progress:
            print()
        embeddings = (
            np.vstack(vectors) if vectors else np.zeros((0, embedder.dim), dtype=np.float32)
        )
        return cls(chunks, embeddings, embedder)

    def search(self, query: str, k: int = 8, candidates: int = 50) -> list[Hit]:
        """Retrieve top-k chunks by reciprocal-rank fusion of both retrievers."""
        if not self.chunks:
            return []

        pool = min(candidates, len(self.chunks))

        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_order = np.argsort(bm25_scores)[::-1][:pool]

        query_vector = self.embedder.encode([query], is_query=True)[0]
        dense_scores = self.embeddings @ query_vector
        dense_order = np.argsort(dense_scores)[::-1][:pool]

        bm25_rank = {int(idx): r for r, idx in enumerate(bm25_order)}
        dense_rank = {int(idx): r for r, idx in enumerate(dense_order)}

        fused: dict[int, float] = {}
        for ranks in (bm25_rank, dense_rank):
            for idx, rank in ranks.items():
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            Hit(
                chunk=self.chunks[idx],
                score=score,
                bm25_rank=bm25_rank.get(idx),
                dense_rank=dense_rank.get(idx),
            )
            for idx, score in top
        ]

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        (directory / "meta.json").write_text(
            json.dumps(
                {"n_chunks": len(self.chunks), "dim": int(self.embeddings.shape[1])},
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> "HybridIndex":
        chunks = read_chunks(directory / "chunks.jsonl")
        embeddings = np.load(directory / "embeddings.npy")
        return cls(chunks, embeddings, embedder)
