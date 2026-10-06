# Progress

## Current status

Finished, **3rd of 13 teams**, best score **0.89769** accuracy from 8
submissions. The competition closed on 2026-09-29.

The deliverable is the finding rather than the system: the benchmark's 140
questions are 6 fixed question-templates across 7 documents, each mapping
deterministically to one answer template, and only the prediction text is
scored. A deterministic lookup reaches the benchmark's ceiling (~0.90); no LLM
or retrieval is involved. 33 tests pass.

## Last session (2026-10-06)

- Recovered the submission history into `EXPERIMENTS.md`.
- Rewrote the README's status section, which still *predicted* an outcome that
  had already happened — it was waiting for a score it had got weeks earlier.

## Open issues

- **The write-up does not exist yet.** This is the most interesting result on
  the account and there is nothing to point a reader at beyond the README.
- The repo's branch is `master` while every other repo here uses `main`.

## Next steps (prioritized)

1. **Write the solution post.** Structure: the 0.40-vs-0.90 gap, what that gap
   implies about the metric, the three probes that narrowed it, the template
   mapping, and the ceiling at ~0.90. Publish it and link it from the README
   and the portfolio card.
2. Rename the branch to `main` for consistency (needs the GitHub default
   branch changed at the same time).

## Decisions & rationale

- Stopped at 8 submissions rather than chasing the last 0.004, because the
  ceiling is understood and the remaining gap is answer phrasing the mapping
  cannot recover (2026-09-03).
- The winning submission deliberately contains no LLM: once the metric is
  understood, a generative model can only add variance (2026-09-03).
