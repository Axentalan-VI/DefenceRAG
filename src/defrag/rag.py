"""Single-shot grounded answering with citations.

This is Phase 1's baseline and, later, the ``naive-rag`` arm of the evaluation:
retrieve once, answer once, cite. The agentic loop in ``covered.agent`` has to
beat this to justify its existence, so it is built honestly rather than as a
strawman.

The one non-negotiable is the **citation contract**: the model may only cite
chunk ids that were actually placed in its context, and every citation is
checked against the retrieved set before an answer is returned. A model that
invents a plausible-looking id is producing exactly the unverifiable output this
project exists to prevent, so invented ids are stripped rather than trusted.

Generation is injected as a callable, so the whole pipeline is testable without
a GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .index import Hit, HybridIndex

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"

CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-./#]+)\]")

SYSTEM_PROMPT = """You answer questions about US health coverage using ONLY the \
evidence provided.

Rules, in order of priority:
1. Every factual claim must be supported by the evidence given. If the evidence \
does not settle the question, reply with exactly {abstain} and nothing else.
2. Cite the evidence id in square brackets immediately after each claim it \
supports, like [doc-name#3]. Only cite ids that appear in the evidence below.
3. Never rely on knowledge outside the evidence, even if you are confident.
4. Do not give medical or legal advice. Report what the documents say.

Being unable to answer is an acceptable outcome. Guessing is not.""".format(
    abstain=ABSTAIN_TOKEN
)

USER_TEMPLATE = """Evidence:
{evidence}

Question: {question}

Answer using only the evidence above, citing ids in brackets."""


@dataclass
class GroundedAnswer:
    """An answer plus everything needed to check it."""

    question: str
    text: str
    citations: list[str]                     # validated chunk ids
    hits: list[Hit] = field(default_factory=list)
    abstained: bool = False
    dropped_citations: list[str] = field(default_factory=list)  # hallucinated ids
    raw: str = ""

    @property
    def is_grounded(self) -> bool:
        """An answer is grounded if it abstained, or cited real evidence."""
        return self.abstained or bool(self.citations)

    def cited_sections(self) -> list[str]:
        """Human-readable citation targets, for display and for judging."""
        by_id = {h.chunk.chunk_id: h.chunk for h in self.hits}
        return [by_id[c].section_id for c in self.citations if c in by_id]


def format_evidence(hits: Sequence[Hit]) -> str:
    return "\n\n".join(hit.as_evidence() for hit in hits)


def build_messages(question: str, hits: Sequence[Hit]) -> list[dict]:
    """Gemma 4 supports a native `system` role - use it rather than prepending."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                evidence=format_evidence(hits), question=question
            ),
        },
    ]


def validate_citations(text: str, hits: Sequence[Hit]) -> tuple[list[str], list[str]]:
    """Split cited ids into (real, invented), preserving first-seen order."""
    allowed = {hit.chunk.chunk_id for hit in hits}
    kept: list[str] = []
    dropped: list[str] = []
    for cited in CITATION_RE.findall(text):
        bucket = kept if cited in allowed else dropped
        if cited not in bucket:
            bucket.append(cited)
    return kept, dropped


def strip_invented_citations(text: str, invented: Sequence[str]) -> str:
    """Remove citation markers the evidence does not support."""
    for bad in invented:
        text = text.replace(f"[{bad}]", "")
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def answer(
    question: str,
    index: HybridIndex,
    generate: Callable[[list[dict]], str],
    *,
    k: int = 8,
) -> GroundedAnswer:
    """Retrieve, generate, then validate citations before returning."""
    hits = index.search(question, k=k)

    if not hits:
        # Nothing retrieved: abstain rather than let the model answer unaided.
        return GroundedAnswer(
            question=question, text=ABSTAIN_TOKEN, citations=[],
            hits=[], abstained=True, raw="",
        )

    raw = generate(build_messages(question, hits))
    text = (raw or "").strip()

    if ABSTAIN_TOKEN in text:
        return GroundedAnswer(
            question=question, text=ABSTAIN_TOKEN, citations=[],
            hits=hits, abstained=True, raw=raw,
        )

    kept, dropped = validate_citations(text, hits)
    if dropped:
        text = strip_invented_citations(text, dropped)

    # An answer that cites nothing real is indistinguishable from a guess.
    if not kept:
        return GroundedAnswer(
            question=question, text=ABSTAIN_TOKEN, citations=[], hits=hits,
            abstained=True, dropped_citations=dropped, raw=raw,
        )

    return GroundedAnswer(
        question=question, text=text, citations=kept, hits=hits,
        abstained=False, dropped_citations=dropped, raw=raw,
    )


def make_transformers_generator(model, tokenizer, max_new_tokens: int = 512):
    """Wrap a loaded Gemma 4 model as the ``generate`` callable."""
    import torch

    def generate(messages: list[dict]) -> str:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # grounded answering wants determinism
            )
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )

    return generate
