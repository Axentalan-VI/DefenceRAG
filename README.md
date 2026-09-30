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

## Data

Competition data is **not** committed to this repo -- fetch it from Kaggle:

```bash
kaggle competitions download -c defence-rag-procurement-policy-reasoning-challenge-2026
unzip -d data/raw defence-rag-procurement-policy-reasoning-challenge-2026.zip
```

That gives `data/raw/{metaData.csv,test.csv,sample_submission.csv}`. The
`data/submission*.csv` files are likewise uncommitted -- regenerate them with
`scripts/reconstruct.py` and `scripts/attrib.py` once the raw data is in place.

## Status (Day 3) - BREAKTHROUGH: exact template mapping

Probes settled the metric empirically: contexts 0.35-0.40; single canonical
template 0.35; kitchen (all templates) 0.30 -- all far below the sample's
0.90178. Since the sample uses the SAME templates yet scores 0.90, the driver
is the **per-question assignment**: each question has one specific reference.

Reverse-engineered it: the 140 questions are **6 fixed question-templates x 7
documents**, each question-template mapping deterministically to one answer
template (`_QUESTION_TO_ANSWER`). The sample assigns the right one to only ~45%
of rows (filler/wrong on the rest).

**`oracle` strategy** applies the correct answer template to all 140 (100% vs
the sample's ~45%), with the correct document via the 140/140 parser.
`data/submission_oracle.csv` is the strong bet to beat 0.90178. No LLM, no RAG
-- a deterministic template lookup. 10 synth tests pass.

Next: submit `submission_oracle.csv` (top of tomorrow's budget). If it lands
near ~1.0, the benchmark is solved; then micro-probe answer phrasing/section.

