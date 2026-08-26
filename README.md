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

## Status (Day 0)

- Repo + vendored Covered modules (`src/defrag/{chunk,index,models,rag,corpus}.py`).
- `corpus.load_metadata_chunks` maps metaData -> the reused `Chunk` type; **5
  tests pass**, all 7 documents present.
- Hybrid index builds over 1,668 chunks; retrieval smoke on the real questions
  lands the right document family for most.
- **Finding:** questions usually name their source explicitly ("Under DPM 2025
  Volume I", "Navy Regulations Part II", "DFPDS Booklet 2024") -> `pred_source` is
  largely recoverable by a question->document parser, likely beating retrieval for
  attribution.

## Blocking (Step 1)

The **metric** (`metric_template 4b2689`) and the **`section-1..12` scheme** are
defined only on the Evaluation tab (JS-rendered, unfetchable here). Read them
in-browser on the joined competition page before tuning generation; the plan has
working defaults and a leaderboard-probe fallback.
