# 4. Cut the Conversation into sentences, not a ladder of overlapping windows

Date: 2026-09-19

## Status

Accepted. Supersedes the Chunk scheme ADR-0002 gated, and retires the
retrieval-mode comparison ADR-0001 set up.

## Context

The Chunker cut every Conversation twelve ways: windows of 1, 2, 3, 4, 6, 8,
11, 15, 20, 27, 36 and 48 normalized tokens, each stepping a fifth of its own
length. It produced about 2 100 candidates for a two-minute Conversation, a
BM25 index ranked them, and a cross-encoder rescored the top ten.

The scheme was chosen against an oracle: the best tIoU any Chunk of a
Conversation reaches, which it scored 0.862 on dev against ADR-0002's gate of
0.75. What it was never measured against is the tIoU actually achieved, and the
two are far apart. Over train and dev, 122 annotated Evidence Spans:

| stage | mean tIoU |
| --- | --- |
| best Chunk in the Conversation (oracle) | 0.862 |
| best Chunk in BM25's top ten | 0.605 |
| the Chunk the cross-encoder cited | 0.431 |
| the same, after padding and snapping | 0.463 |

Half the achievable evidence score was being lost between the Chunker and the
citation, on the half of the score that carries 0.6 of the weight.

The obvious reading is that the candidate pool is too shallow, and it is wrong.
BM25's pool ceiling does keep rising with depth — 0.605 at ten, 0.759 at fifty,
0.812 at five hundred — but handing the cross-encoder the deeper pool makes the
citation *worse*, not better: 0.467 at ten, 0.462 at fifty, 0.459 at a hundred.

The cause is that a ladder asks a Relevance model to make a decision it has no
signal for. Twelve overlapping windows of the same stretch are the same subject
matter with different boundaries, and a cross-encoder scores what a passage is
about. Among a dozen views of the utterance that answers the Question it has no
reason to prefer the one whose edges match the annotation, and the annotation is
what tIoU is measured against. Deepening the pool adds more views of the same
stretch, so it adds noise to a decision that was already being made blind.

Three measurements say the same thing from different directions. Cutting at
sentences instead — one Chunk per spoken sentence, no overlap, about fifty per
Conversation — lowers the oracle to 0.697 and raises the citation to 0.508.
Adding two- and three-sentence Chunks beside the single ones raises the oracle
again, to 0.782, and drops the citation to 0.440. And asking the cross-encoder
to choose between just four options — the cited sentence, and it extended by a
neighbour on either side or both — scores 0.438 against 0.508 for not asking.

Every time the model is asked about extent rather than aboutness, it gets worse.

## Decision

A Chunk is one spoken sentence. The Chunker splits the Conversation's Words on
terminal punctuation and cuts an unpunctuated run at `chunk_max_words`, which is
a robustness guard against a sparsely punctuated transcript and fires nowhere in
the supplied data.

Three things follow and are done rather than deferred:

- **No retriever.** Fifty sentences is a batch the cross-encoder scores in one
  pass, and a BM25 prefilter can only remove the right sentence: at depths 8,
  16, 24 and 32 it scores 0.461, 0.477, 0.487 and 0.488 against 0.507 for
  scoring all of them. `medapp/retrieval.py` and `medapp/dense.py` are deleted
  with the retrieval-mode setting, the fusion settings and their harnesses.
- **No span padding.** Padding existed to repair the arbitrary edges a fixed
  window cuts. A sentence's edges are real ones, and the padding sweep over the
  sentence Chunks chooses zero at both ends. `medapp/span_refiner.py` is deleted
  and `Verdict` carries the Chunk it cited rather than a span beside a ranking.
- **The knobs are re-fitted.** The Relevance gate was 0.4, a bar set against
  one-to-four-token windows that score high on any Question sharing their
  tokens. A sentence scores lower on the same match, and holding the old bar
  against it rejected 33 of 122 Positives. Re-swept jointly over train and dev
  by `python -m scripts.tune`, the gate is 0.1, the Entailment threshold 0.0002
  and the Entailment depth 1.

## Consequences

Measured end to end on the folds, against the ladder system on the same cached
transcripts:

| fold | ladder | sentences |
| --- | --- | --- |
| train | 0.560 | 0.586 |
| dev | 0.633 | 0.657 |
| train+dev | 0.606 | 0.630 [0.586, 0.670] |

Both halves move on dev: accuracy 0.894 either way, mean tIoU 0.459 to 0.500.

The ceiling falls from 0.862 to 0.697, and that is the cost. It is the right
trade only for as long as nothing can select within a redundant candidate set —
the ladder is the better scheme the moment something can judge boundaries rather
than subject matter. An extractive-QA span model was the obvious candidate and
was measured: `deepset/roberta-base-squad2` over the whole transcript scores
0.417 mean tIoU, and 0.455 when its span is snapped out to the sentences it
overlaps, against 0.508 for the cross-encoder on sentences. It undershoots,
because SQuAD answers are minimal phrases and an annotated Evidence Span is a
clause — a median of 2.88 s against a median sentence of 1.52 s. Their union
oracle is 0.592, so the two are complementary and neither dominates; what is
missing is a selector between them.

Judging one Question costs 311 ms on average and 689 ms worst on the dev
machine's CPU, against 180 ms and 308 ms with the ten-candidate shortlist. Ten
Questions at the worst case is 6.9 s against the 12 s the answering reserve
keeps back, so the budget still holds with margin, and the deployment GPU has
more of it.

`bm25s` and `PyStemmer` leave the dependency set.
