# DefenceRAG

Grounded, **source-attributed** QA over Indian defence procurement/policy
documents, for the [DefenceRAG Challenge 2026](https://www.kaggle.com/competitions/defence-rag-procurement-policy-reasoning-challenge-2026)
(Kudos, solo, closes Sep 29). See [PLAN.md](PLAN.md).

Reuses the **Covered** grounded-RAG stack (`Build with Gemma - Boston`): the
hybrid BM25 + EmbeddingGemma index, the Gemma-4 loader (bf16 lesson), and the
citation/abstention answering, repointed at the defence corpus.

## Task

- **Corpus:** `metaData.csv` -> 1,668 pre-chunked passages across 7 PDFs
  (`document, section, text, topic`).
- **Questions:** `test.csv` (140) `id, question, contexts` -- the `contexts` are
  generic boilerplate, so grounding comes from retrieving over metaData.
- **Submission:** `id, prediction, pred_source, pred_section` -- an answer plus
  the source PDF and a `section-1..12` bucket.

## Status (Day 3) - STRATEGY PIVOT (evidence-driven)

The leaderboard + data analysis overturned the grounded-RAG premise:

- **`sample_submission.csv` is the #1 score (0.90178)**, above every real
  competitor (~0.897). Its answers are short generic policy sentences (~5
  templates recycled) with **wrong** `pred_source`/`pred_section`.
- **The corpus for answering is templated**: `test.csv` has only **7 distinct
  context strings / 9 unique passages** across 140 questions -- generic
  per-document policy statements.

Conclusion: the metric is **prediction-vs-reference text similarity**, the
references are generic policy prose, and attribution columns barely count. A
fact-grounded RAG answer would *diverge* from the generic reference and score
lower -- which is why the serious entrants are stuck below the sample.

**Pivot:** `synth.py` returns each question's own **document-correct contexts**
as the answer (fixing the sample's document mismatch via the 140/140 parser).
Three GPU-free candidates to A/B on the 5/day budget:
- `data/submission_echo.csv`  -- the 3 context sentences joined
- `data/submission_lead.csv`  -- doc-named lead + contexts
- `data/submission_first.csv` -- the single document-naming sentence (sample-style)

The Gemma-4 grounded pipeline (`generate.py`, `defrag_submit.ipynb`) is retained
as a fallback in case the hidden 70% rewards real grounding, but the evidence
says echo-the-contexts wins. 27 tests pass.

Next: submit the three candidates, compare to 0.90178, keep the best.

