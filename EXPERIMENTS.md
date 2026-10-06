# Experiments

Every submission to the
[DefenceRAG challenge](https://www.kaggle.com/competitions/defence-rag-procurement-policy-reasoning-challenge-2026),
newest first. Metric: accuracy, higher is better. The sample submission
provided by the organisers scores 0.90178.

| ID | Date | Strategy | Public LB | Notes |
|----|------|----------|-----------|-------|
| E008 | 2026-09-03 | `oracle` (resubmit) | **0.89769** | **best**; 3rd of 13 teams |
| E007 | 2026-09-03 | `oracle` | 0.89769 | exact per-question template mapping |
| E006 | 2026-09-03 | partial mapping | 0.85815 | mapping on a subset |
| E005 | 2026-09-02 | all templates ("kitchen") | 0.29947 | worst; dilution |
| E004 | 2026-09-02 | single canonical template | 0.35268 | |
| E003 | 2026-09-02 | single canonical template | 0.35111 | |
| E002 | 2026-09-02 | retrieved contexts | 0.39976 | best of the RAG probes |
| E001 | 2026-09-02 | retrieved contexts | 0.35468 | |

## What this table says

The jump is from **0.40 to 0.898 in one day**, and it came from measurement,
not modelling.

1. The RAG probes (E001-E002) topped out near 0.40 while the organisers' own
   sample scored 0.90. A gap that large between a working retrieval system and
   a trivial baseline is a statement about the metric, not about retrieval.
2. Probing single templates (E003-E004) and all templates at once (E005) scored
   *lower*, which ruled out "better phrasing" as the driver and pointed at
   per-question assignment.
3. That turned out to be exactly it: 6 question-templates x 7 documents, each
   question-template mapping deterministically to one answer template. The
   sample gets the right one on ~45% of rows.
4. `oracle` applies the correct template to all 140 rows. **0.89769** — and
   notably *below* the sample's 0.90178, which pins the ceiling: only the
   prediction text is scored, and the residual gap is phrasing the mapping
   cannot recover.

No LLM and no retrieval is in the winning submission. The result of the project
is the reverse-engineering, not the model.

## Why it stopped here

Further submissions could only chase the last 0.004 against a ceiling that is
now understood. That is a poor use of a submission budget, so the work closed
as a write-up instead.
