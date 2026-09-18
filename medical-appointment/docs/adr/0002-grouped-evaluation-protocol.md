# 2. Evaluation protocol: grouped 20/40/40, bootstrap CI, Validation never tunes

Date: 2026-09-18

## Status

Accepted

## Context

Two thresholds decide every answer: a Relevance threshold that rejects Off-Topic
Questions, and an Entailment judgement that rejects Hard Negatives. Both must be
calibrated, and calibrating them wrongly is the failure mode that would cost the
single evaluation attempt without ever looking like a bug.

The supplied data appears to be 390 questions, which sounds ample. It is not.
Ten Questions about one Conversation share one transcript, one set of Chunks and
one speaker pair, so they are not independent observations. The real sample size
is **39 Conversations**.

Three sources bear on this and none covers it completely.

Our evaluation notes prescribe a fixed 20/40/40 train/dev/test split over ~100
labelled traces, hold-out never inspected until done, TPR and TNR both measured
because measuring one hides systematic bias, and a bootstrap confidence interval
over resampled test predictions. The notes assume independent traces and say
nothing about grouping.

Huyen (Ch4) contributes the contamination principle and the rule that components
are evaluated separately because a system-level score hides where failures
occur. Ch4 says nothing about splitting.

Standard practice contributes grouped splitting, which neither source states.

Separately, the competition permits unlimited Validation runs against 19
Conversations. This is a trap. Repeatedly tuning against a set that returns a
score is contamination by Huyen's own definition, and would leave us overfitted
to the validation set by the time the single evaluation attempt arrives.

## Decision

**Split 20/40/40 train/dev/test, grouped at Conversation level** — 8/16/15
Conversations, assignment fixed by seed and recorded. All Questions of a
Conversation live in the same fold. The test fold is not inspected
until tuning is finished.

Grouping is the part no source states and it is the part that matters most here.
Splitting a Conversation's Questions across folds leaks its transcript and its
Chunks into both sides, and the resulting numbers would be optimistic in a way
no amount of care downstream could detect.

Chosen over grouped k-fold cross-validation, which was the earlier candidate.
Cross-validation would extract more signal from 39 groups and give a variance
estimate, but the notes prescribe a fixed hold-out and pair it with bootstrap
confidence intervals, which supplies the variance estimate by another route. We
follow the source rather than improvise around it.

**Thresholds chosen on the dev fold, for robustness under bootstrap
resampling rather than by argmax.** The argmax over a handful of groups is
substantially noise; the value whose accuracy is stable across resampled dev
predictions is preferred. The test fold never informs a threshold: it is scored
exactly once, after tuning ends, and reported with a bootstrap confidence
interval. (Amended 2026-09-18: the original wording placed the bootstrap over
the test fold, which would have tuned on the hold-out.)

**TPR and TNR reported separately at every stage.** This is not general good
practice here, it is specific: the shipped baseline scores 1.000 on Positives
and 0.000 on everything else, and a single blended number cannot see that.

**Components tuned against component metrics, never the blended score.**
Retriever on recall@k, Chunker on oracle-selected tIoU, Relevance threshold on
Off-Topic accuracy, Entailment on Hard-Negative accuracy with Positive accuracy
held fixed. The 0.4/0.6 weighted score is a report, not a tuning signal.

**Competition Validation runs never feed back into threshold choice.** They
confirm deployment, protocol shape and latency. Thresholds are set locally and
are not adjusted in response to a Validation score.

**Usefulness thresholds are gates, fixed before building.** Two, per Huyen Ch4's
rule that the bar is defined before the work rather than after the first
measurement:

- **recall@5 >= 0.95** for the retriever. Anything not retrieved is unreachable
  by every judge downstream; a 5% structural miss is already ~0.03 off the final
  score. (Amended 2026-09-19: not met by BM25 alone, and the gate is restated
  below rather than lowered.)
- **worst-case per-Conversation latency <= 40 s** on the deployment VM against
  the longest supplied audio. One Conversation over 60 s loses its Questions and
  eats the whole-attempt budget of every Conversation after it.
- **oracle-selected Chunk tIoU >= 0.75** for the Chunker — the best tIoU
  achievable by picking the single best Chunk per annotated Evidence Span. This
  is the ceiling on 0.6 of the score. Below it the Chunker is the problem, and
  no work on Relevance or Entailment can recover it.

Failing a gate sends work back to that component. It does not send work forward.

If a gate proves unreachable, it is **renegotiated explicitly and amended into
this ADR with the reason**. It is never adjusted in passing. A gate that is
quietly lowered when it is inconvenient is worse than no gate, because it still
reads as a passed check.

**Error analysis precedes evaluator construction.** All 39 Conversations are
transcribed once and cached, but only the train and dev folds (24
Conversations) are read against their annotated Evidence Spans to build a
taxonomy of failure modes before any threshold is tuned. The test fold is
transcribed, not read. (Amended 2026-09-18: reading all 39 contradicted the
untouched hold-out above.) Both sources
name the alternative as a pitfall — optimizing retrieval parameters before error
analysis on what is actually failing.

## Amendment 2026-09-19: recall@5 is not a gate BM25 alone can pass

Measured by `python -m scripts.retrieval_metrics` on the tuned Chunk ladder
(lengths 1–48 normalized tokens, stride 0.2 of each), against the cached
`large-v3` transcripts. A Chunk counts as having found an annotated Evidence
Span when their temporal IoU is at least 0.5 — it overlaps the annotation more
than the two of them miss each other.

| Fold  | Spans | recall@1 | recall@5 | recall@10 | oracle tIoU |
|-------|-------|----------|----------|-----------|-------------|
| train | 47    | 0.213    | 0.489    | 0.574     | 0.839       |
| dev   | 75    | 0.333    | 0.533    | 0.707     | 0.835       |

**The Chunker's gate passes.** Oracle-selected tIoU is 0.835 on dev against a
gate of 0.75, so the ceiling on 0.6 of the score is not the Chunker's to raise
and work moves on from it. Finer ladders were measured up to 2,600 Chunks per
Conversation and oracle tIoU plateaus near 0.85, which is where the ASR's word
boundaries and the two mis-annotated Positives put it.

**The retriever's gate fails, by a lot.** 0.533 on dev against 0.95. Three
things were measured before recording this rather than after: BM25's length
normalization `b` swept from 0 to 1 (moves recall@5 by under 0.03 and 0.4 is
already near the best of it), the Chunk ladder swept from 138 to 2,600 Chunks
per Conversation (recall@5 between 0.52 and 0.71 on dev, always far short), and
non-maximum suppression over the ranked Chunks by time (recall@5 up 0.05, but
recall@50 down from 0.91 to 0.77 — it buys the near ranks by capping the deep
ones).

The shortfall is not a defect in the index. It is the corpus. ADR-0001's
amendment already recorded that many Hard Negatives share little vocabulary
with the passage that refutes them, and the error analysis counts 14 of 122
Positives whose evidence is a paraphrase with no content word in common and 4
more whose drug name the ASR mis-spelled. Those 18 are lexically unreachable at
any k, which caps BM25's recall near 0.85 before ranking is considered at all.
BM25 also has no way to prefer the Chunk whose *boundaries* are right among the
dozens overlapping the same correct passage, which is what separates recall@5
from the 0.91 the same ranking reaches by k=100: the evidence is found, and
ranked deep.

**The gate is not lowered.** Both components it exists to protect are still
ahead: the cross-encoder reranker reads Question and Chunk jointly over the top
candidates, which is exactly the boundary discrimination BM25 lacks, and dense
retrieval plus fusion is what the 18 lexically unreachable Positives need.
recall@5 is re-measured after each, and the gate is met there or the shortfall
is amended again with what the reranker and the dense half actually bought.

What this does change is the depth the system carries forward. `retrieve_bm25`
returns 10 ranked Chunks per Question rather than 5, because recall@10 is 0.707
on dev against recall@5's 0.533 and the reranker is the component that turns a
deeper candidate list into a better rank-1. The gate stays stated at 5.

## Consequences

The effective test set is roughly 16 Conversations, or 160 Questions. Small.
Bootstrap intervals will be wide and must be reported as such rather than
rounded away.

The discipline around Validation is a commitment, not a mechanism. Nothing
enforces it. It is recorded here because it is the rule most likely to be broken
quietly the night before a deadline, and breaking it would invalidate every
number above it.

Transcripts are cached to `transcripts/` (already git-ignored) so the ASR cost is
paid once and every tuning iteration runs offline. The cache belongs to the
evaluation harness and must never enter the request path, or the dev loop's
caching silently becomes production behaviour.
