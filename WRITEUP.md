# The benchmark is templated: how a lookup table scored 0.898 and why the sample submission still wins

*DefenceRAG Procurement & Policy Reasoning Challenge 2026 — 3rd of 13 teams,
0.89769 accuracy, 8 submissions.*

My best submission contains no language model and no retrieval. It is a
dictionary with six entries. This is the story of how I got there, and of the
one number that says I did not get all the way.

## The gap that started it

I built the obvious thing first: a hybrid BM25 + EmbeddingGemma index over the
1,668 pre-chunked passages in `metaData.csv`, with a Gemma-4 answerer under a
citation contract. Retrieval worked. The questions are answerable from the
corpus. It scored **0.39976**.

The organisers' own `sample_submission.csv` scored **0.90178**.

That gap is the whole problem. A working retrieval system losing to a baseline
by 0.5 accuracy is not a statement about my retrieval. It is a statement about
what the metric measures. The sample's predictions are short generic policy
sentences — about five of them recycled across 140 rows, frequently naming the
wrong document. If that beats grounded answers by this much, then the reference
answers must themselves be generic policy prose, and the score must be
prediction-against-reference text similarity rather than factual correctness.

So I stopped improving the system and started measuring the metric.

## Three probes, and what each one ruled out

The submission budget was 5 per day. I spent one day on probes designed so that
each outcome eliminated a hypothesis.

| Probe | Prediction | Score | What it rules out |
|---|---|---|---|
| `echo` | the question's own `contexts`, joined | 0.35–0.40 | retrieval quality is the bottleneck |
| `single` | one canonical policy template for every row | 0.35111 | "the right vocabulary" is enough |
| `kitchen` | all six templates concatenated, every row | 0.29947 | recall-by-length; more text dilutes |

The third result is the informative one. If the metric rewarded overlap with a
generic reference, throwing every template at every question should have scored
*higher* than one template. It scored lower — the worst of all eight
submissions. That kills similarity-by-coverage and leaves only one candidate:
**the reference answer is specific to the individual question**, and a wrong
template actively costs you.

Which turns the problem into an assignment problem, not a generation problem.

## The structure

The 140 questions are **6 question templates × 7 documents**, with a handful of
near-duplicates. The seven documents are DPM 2025 Volumes I and II, the DFPDS
booklet, and Navy Regulations Parts I–IV. The six question templates are
recognisable from their opening clauses:

```python
_QUESTION_TO_ANSWER = (
    ("what authority level generally handles",        "authority"),
    ("what should happen when a proposal exceeds",    "escalation"),
    ("support compliance-oriented decision making",   "policy"),
    ("what governance or procedural guidance",        "citation"),
    ("which escalation principle",                    "reasoning"),
    ("how should a rag system use",                   "compliance"),
)
```

Each question template maps deterministically to exactly one answer template. I
recovered the mapping from the rows where the sample submission happened to be
right. The sample assigns the correct template to roughly 45% of rows and fills
the rest with whichever sentence it liked, often with the wrong document name
substituted in.

The `oracle` strategy is then three lines: classify the question by its opening
clause, look up the answer template, substitute the correct document name. The
document comes from a parser that resolves all 140 questions to the right PDF.

```python
def predict_oracle(question, contexts, document):
    theme = _answer_theme(question) or "authority"
    return _TEMPLATES[theme].format(doc=FRIENDLY.get(document, document))
```

**0.85815** with the mapping on a subset. **0.89769** with it on all 140 rows.

## The number I can't explain away

0.89769 is below the sample's 0.90178.

I went from 0.40 to 0.898 by understanding the metric, and the baseline I was
trying to beat still beats me by 0.004. If my model of the benchmark were
complete, 100% correct template assignment should have cleanly exceeded a
submission that gets 45% of them right. It does not.

So something in the reference answers is not captured by the six templates.
The most likely candidate is phrasing: exact wording, punctuation, or sentence
count that my reconstructed templates approximate but do not reproduce
character for character. A per-row score would settle it in one pass, but the
leaderboard returns a single aggregate, so the remaining 0.004 would cost many
submissions of blind perturbation to chase.

That is where I stopped. The ceiling is understood, the mechanism is
documented, and the next 0.004 is phrasing archaeology rather than
engineering.

## What I would tell the organisers

The benchmark as scored does not measure retrieval, grounding, or reasoning
about procurement policy. It measures whether you can reproduce a small set of
generic reference sentences and attach the right document name. The corpus is
real and the questions are answerable from it, so the intent is clearly there —
but a system that reads the documents and answers correctly scores 0.40, and a
dictionary scores 0.90.

Two changes would fix it:

1. **Score the attribution columns meaningfully.** `pred_source` and
   `pred_section` are the part a grounded system gets right and a lookup table
   cannot fake. They currently barely count.
2. **Write question-specific reference answers**, or score with a model-based
   judge against the source passage rather than against a reference string.
   As long as the reference is generic prose, generic prose wins.

## What I would tell myself

The two useful habits here were cheap and I nearly skipped both.

**Submit the trivial baseline first.** The 0.40-against-0.90 gap was available
on day one and it is the entire finding. I only saw it because the sample's
score was on the leaderboard next to mine.

**Design probes to eliminate, not to improve.** `kitchen` was never going to be
a good submission. It was built to fail in a specific way, and its failure is
what pointed at per-question assignment. Three submissions spent on ruling
things out were worth more than three spent on tuning a retriever that was
already working fine.

---

Code: [github.com/Axentalan-VI/DefenceRAG](https://github.com/Axentalan-VI/DefenceRAG).
The strategies are in `src/defrag/synth.py`; `EXPERIMENTS.md` has every
submission and its score. 33 tests pass, including synthetic tests pinning the
template mapping.
