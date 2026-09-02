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

## Status (Day 2)

- **Grounded generation pipeline** (`generate.py`): retrieval restricted to the
  question's `pred_source`, a defence-specific grounded prompt, injectable
  generator so the loop is tested offline with a stub (22 tests pass).
- **Kaggle submission notebook** (`notebooks/defrag_submit.ipynb`): rule-based
  source + BM25-within-document + **Gemma-4 E4B** generation (bf16, `disable_mmap`,
  transformers>=5.15 upgrade cell). Writes `/kaggle/working/submission.csv`. The
  inlined parser was verified identical to the tested module (140/140).
- E4B is the default generator (mounts reliably vs the 12B's partial-mount issue).

Next: run the notebook on Kaggle for a real answer-quality submission; resolve
Step 1 (metric + `section-1..12`) to tune `prediction` and fix `pred_section`.
