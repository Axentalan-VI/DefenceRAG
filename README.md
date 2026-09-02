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

## Status (Day 1)

- **`pred_source` solved rule-based: 140/140.** Every test question names its
  source ("DPM 2025 Volume II", "Navy Regulations Part III", "DFPDS Booklet
  2024"); `attribute.parse_source` recovers it exactly, beating retrieval (which
  confuses Vol I/II and Navy Parts I-IV).
- **Baseline submission builds and validates** against `sample_submission.csv`
  (140 rows, exact schema): `submit.build_submission`. Predictions are extractive
  snippets for now; `pred_section` is a placeholder.
- **18 tests pass** (corpus, attribution incl. the 140/140 guard, submission shape).

Next: Day 2 swaps the extractive baseline for Gemma-4 grounded generation on
Kaggle. Still blocked on Step 1 (metric + `section-1..12`) for tuning.
