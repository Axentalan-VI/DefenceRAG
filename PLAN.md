# DefenceRAG — Procurement & Policy Reasoning Challenge 2026

> This plan file was repurposed for a new project. The previous occupants are
> preserved in their repos: the AI-agent-security plan at
> `e:\Kaggle\AI Agent Security\PLAN.md`, and the Covered plan at
> `e:\Kaggle\Build with Gemma - Boston\PLAN.md`.

## Context

[DefenceRAG: Procurement & Policy Reasoning Challenge 2026](https://www.kaggle.com/competitions/defence-rag-procurement-policy-reasoning-challenge-2026)
— a Community, **Kudos-only, solo** competition (you are **joined**). Closes
**Sep 29 2026** (~34 days), **5 submissions/day**, custom metric. Build a RAG
system that answers questions about Indian defence procurement/policy documents
with **explainable, source-attributed** answers.

**Decisions:** Kaggle free-tier compute · **reuse the Covered grounded-RAG stack
heavily** (built earlier this session at `e:\Kaggle\Build with Gemma - Boston`).

**Intended outcome:** a clean, reproducible grounded-RAG pipeline that produces
well-attributed answers and ranks respectably on a small field (5 teams), plus a
strong public writeup — the realistic value of a Kudos competition.

---

## Task ground-truth (read from the actual files)

**Corpus** — `metaData.csv`: the docs pre-chunked into **~1,668 chunks** (the
62,299 "rows" are multi-line text; `Import-Csv` parses 1,668). Columns:
`chunk_id, document, section, text, topic`.
- `document`: one of **7 PDFs** — `DPM-2025-VOLUME-{I,II}.pdf`,
  `Delegation_of_Financial_Powers_Rules_2024_Booklet.pdf`, `RegsNavy{I,II,III,IV}.pdf`.
- `section`: 1,157 distinct descriptive headings (e.g. "Preface", "Item 18 ...").
- `topic`: only 3 values — `regulations`, `procurement`, `financial powers`.
- `chunk_id`: page-anchored (e.g. `..._p0003`). Raw PDFs are also provided.

**Questions** — `test.csv` (140 rows): `id, question, contexts`. The `contexts`
are `|||`-separated but are **generic boilerplate** ("defines authority
boundaries, delegated powers... ||| Escalation to higher competent authority...")
— i.e. weak/templated, **not** real retrieved passages. The real grounding signal
must come from retrieving against `metaData.csv`.

**Submission** — `sample_submission.csv` (140 rows):
`id, prediction, pred_source, pred_section`.
- `prediction`: the free-text, grounded answer.
- `pred_source`: the PDF the answer is grounded in (maps cleanly to
  `metaData.document`).
- `pred_section`: a bucket **`section-1` … `section-12`** — **this scheme is NOT
  present in the data** (`topic` has 3 values, `section` has 1,157). Its
  definition lives only in the Evaluation tab. **Resolving it is Step 1.**

No `train.csv` and no gold answers ship, so **answer quality cannot be scored
offline** — rely on leaderboard probing (5/day) plus manual spot-checks against
the PDFs.

---

## Step 1 (blocking): pin the metric and the section scheme

Before building, read the **Evaluation tab** and **Discussion** on the joined
competition page (in-browser; both are JS-rendered and not fetchable here). Nail:
1. **How `prediction` is scored** — semantic similarity vs ROUGE/BLEU vs
   LLM-as-judge — and the **relative weight** of `prediction` vs `pred_source`
   vs `pred_section`.
2. **What `section-1..12` means** — a fixed 12-way topic taxonomy? a per-document
   coarse section index? Map every `metaData` chunk to its bucket.

If the Evaluation tab is thin, reverse-engineer via cheap probe submissions: e.g.
submit constant `pred_section` values and watch the score move to infer its weight
and validity. Paste the Evaluation text back and the plan's defaults get replaced
with the real rule.

**Working defaults until confirmed:** treat `pred_source` (well-defined, 7-way) as
the primary attribution target; score `prediction` for grounded, concise answers;
map `pred_section` from the retrieved chunk via the best available signal (topic →
bucket, or a per-document section index) and refine once the scheme is known.

---

## Architecture (reuse Covered; swap the corpus)

Fork the Covered pipeline and repoint it at the defence corpus. Reused files
(from `e:\Kaggle\Build with Gemma - Boston\src\covered\`):

- **`corpus/index.py`** — `HybridIndex` (BM25 + EmbeddingGemma, RRF fusion),
  `EmbeddingGemmaEmbedder`, `HashingEmbedder`, and the **`check_embeddings`
  NaN-guard**. Directly reusable over the metaData chunks.
- **`models.py`** — Gemma 4 / EmbeddingGemma loading with the **bf16** lesson and
  `resolve_path` for Kaggle model inputs.
- **`rag.py`** — `answer()` with the **citation contract** and **abstention**
  (`ABSTAIN_TOKEN`); its citation validation is what produces `pred_source` /
  `pred_section`.
- **`corpus/chunk.py`** — section-aware chunking, only if we re-chunk the raw PDFs
  for finer granularity than metaData provides.
- **Kaggle notebook patterns** from `Build with Gemma - Boston/notebooks/00_kaggle_gate.ipynb`
  — model discovery (`find_model`), dtype, `disable_mmap`, checkpoint integrity.
  These were hard-won; reuse verbatim.

**New repo:** `e:\Kaggle\DefenceRAG\`, matching sibling conventions (`src/defrag/`,
`notebooks/`, `scripts/`, `data/`, `tests/`, `requirements.txt`, `README.md`,
`PLAN.md`, `.venv`, git).

**Pipeline:**
1. **Load corpus** — parse `metaData.csv` into chunks (already chunked). Optionally
   re-chunk the 7 PDFs (via `pypdf`) for finer spans if metaData granularity hurts
   retrieval.
2. **Index** — `HybridIndex.build` over chunk `text`, using the same
   `retrieval_text` trick (prepend `document` + `section` so title/heading terms
   are searchable). Ship the index as a Kaggle Dataset.
3. **Retrieve** — per question, top-k chunks; optionally fuse the provided
   `contexts` as a weak prior.
4. **Answer** — Gemma 4 (12B QAT 4-bit or E4B, per the gate notebook's speed
   result) generates a grounded answer over the retrieved spans, enforcing the
   citation contract; abstain-to-safe-default rather than hallucinate.
5. **Attribute** — `pred_source` = document of the top supporting chunk;
   `pred_section` = its bucket per the Step-1 mapping.
6. **Write** `submission.csv` (`id, prediction, pred_source, pred_section`).

**Models (Kaggle handles, verified this session):**
`google/embeddinggemma/transformers/embeddinggemma-300m` (retrieval);
`google/gemma-4/transformers/gemma-4-12b-it-qat-q4_0-unquantized` (bf16, NF4) or
`gemma-4-e4b-it` for speed. Attach as Model inputs; run internet-disabled.

---

## Phases

- **Day 0 — Join done; pin metric + section scheme (Step 1).** Set up
  `e:\Kaggle\DefenceRAG\`, vendor the Covered modules, load `metaData.csv` and the
  CSVs. *Checkpoint: exact metric + `section-N` mapping documented (or a probe plan
  to infer them).*
- **Day 1 — Index + baseline submission.** Build the hybrid index over metaData;
  a minimal pipeline that retrieves top-1 and returns `document` as `pred_source`
  with a templated `prediction`. Submit to confirm the format scores. *Checkpoint:
  a valid non-zero leaderboard score.*
- **Day 2 — Grounded generation.** Wire Gemma 4 answering with the citation
  contract; verify on the Kaggle T4 (reuse the gate notebook). *Checkpoint:
  fluent, grounded answers citing a real chunk.*
- **Day 3 — Attribution + section mapping.** Lock `pred_source`/`pred_section`;
  probe-tune the section scheme against the leaderboard. *Checkpoint: attribution
  columns measurably help the score.*
- **Day 4-5 — Retrieval quality + reasoning.** Improve top-k, handle multi-doc
  reasoning (delegation thresholds → escalation), fuse provided contexts. Iterate
  within 5 subs/day.
- **Day 6-7 — Polish + writeup.** Clean notebook, reproducible index dataset,
  public writeup of the grounded-attribution approach.

---

## Verification

1. `pytest tests/` — corpus loads all 7 documents; index round-trips; the
   `check_embeddings` guard rejects NaN (reuse Covered's tests); every submission
   row has a valid `pred_source` ∈ the 7 PDFs and a valid `pred_section`.
2. Retrieval smoke test — a handful of hand-checked defence questions return the
   correct document in top-5 (mirror `Build with Gemma - Boston/scripts/smoke_retrieval.py`).
3. `submission.csv` shape/validity check against `sample_submission.csv` (140 rows,
   exact columns, ids 1..140).
4. On Kaggle: the pipeline notebook runs **internet-disabled** with the model +
   index attached, and produces `submission.csv`.
5. Leaderboard probing (5/day): confirm each component (answer, source, section)
   moves the score as expected.

---

## Risks & open items

| Risk | Mitigation |
| --- | --- |
| **Metric unknown** (`metric_template 4b2689`) | Step 1: read Evaluation tab / Discussion; else probe. Do not build generation tuning until the answer-scoring method is known. |
| **`section-1..12` undefined in data** | Step 1 priority; reverse-engineer via constant-value probe submissions if the tab is thin. |
| No gold answers ship | No offline answer scoring; use leaderboard probes + manual PDF spot-checks; build a tiny self-labeled dev set. |
| Provided `contexts` are boilerplate | Ignore or down-weight; ground on retrieved metaData chunks. |
| 12B too slow on T4 | E4B fallback (gate notebook has the speed numbers); contexts-provided means a smaller model can still answer. |
| Kudos-only motivation | Value is the portfolio artifact + writeup; scope accordingly. |

**Open items to confirm on the joined page:** exact metric + weights; the
`section-N` definition; whether external/API models are permitted at scoring
(assume internet-off, offline models only).
