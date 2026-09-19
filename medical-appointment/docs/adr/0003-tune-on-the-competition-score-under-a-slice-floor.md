# 3. Tune on the competition score, under a per-type slice floor

Date: 2026-09-19

## Status

Accepted

Refines ADR-0002, which fixed the split and the resampling. This fixes what is
resampled *for*.

## Context

Every knob that decides a Verdict had been chosen against a different proxy.
The Relevance threshold was chosen on Off-Topic accuracy, the Entailment
threshold on Hard-Negative accuracy with Positive accuracy held at a floor, the
span padding on mean tIoU over already-retrieved spans, and the Entailment
depth was not chosen at all — it was fixed at one Chunk by an argument rather
than a measurement.

None of those is the number the attempt is graded on, which is

    Score = 0.4 x Accuracy + 0.6 x mean tIoU

and the gap is not presentational. The tIoU average is taken over every
annotated Positive whether it was answered yes or not. A Positive the system
rejects therefore scores zero on *both* halves: its marginal cost is
`0.4/n + 0.6 x (the tIoU it would have earned)/n`, against `0.4/n` for a Hard
Negative wrongly accepted. On the dev fold that ratio is about two to one. An
accuracy-only sweep prices them equally, so it lands systematically tighter
than the score wants — which is exactly what had happened.

The knobs also do not decompose. Judging more Chunks lets a Hard Negative find
one that entails it, which a higher Entailment threshold then has to reject;
padding that suits a Chunk of one token ruins a Chunk of forty-eight. Sweeping
them one at a time measures each against the others' stale values.

But tuning on the aggregate alone has its own failure, and it appeared on the
first run. The highest-scoring candidates all bought mean tIoU by lowering the
Entailment threshold: Hard-Negative accuracy fell from 0.855 to 0.790 while the
score rose from 0.618 to 0.628. That is Simpson's-paradox-shaped — the
aggregate improved while the slice the case is actually about got materially
worse — and the README is explicit that hard negatives are most of the no's and
that the evaluation set is a different draw of Conversations. A system that
leans on padding arithmetic rather than on judgement travels worse to a draw it
has not seen.

## Decision

Knobs are chosen against the competition score, computed by `harness.score` and
swept jointly by `harness.answer_sweep`, subject to three rules:

1. **The lower bound, not the point estimate.** As ADR-0002. Lower bounds
   within `TIE_TOLERANCE` (0.005) count as tied, because a bound read off 2000
   resamples of 16 Conversations carries more Monte Carlo noise than its third
   decimal and reading it finer is choosing on the draw.
2. **Ties break on missed Positives, fewest first.** It is the only failure that
   costs both halves, and the one this fold measures least well.
3. **A slice floor.** No candidate may drop any Question type's accuracy more
   than `SLICE_FLOOR` (0.03) below the reference configuration. The reference is
   pinned to the pre-tuning baseline in `scripts.tune`, never to whatever
   Settings currently hold: a floor that moved with the defaults would ratchet,
   permitting a type to walk down to nothing in steps that each looked
   acceptable.

The sweep runs in two passes — decision knobs first at a fixed padding, then the
padding grid over the best few — because padding cannot reorder the decision
knobs by more than the scale factor it applies to the tIoU half, and the full
product runs past a million candidates.

`harness.answer_sweep.outcome` replays the shipped Answerer over a recording
rather than reimplementing it, and a test asserts the two agree Verdict for
Verdict. A tuning rule that drifts from the served rule chooses knobs for a
system nobody runs.

## Consequences

On dev, 16 Conversations and 160 Questions: score 0.618 to 0.626
(90% interval [0.578, 0.674]), Positive accuracy 0.867 to 0.893, missed
Positives 10 to 8, mean tIoU 0.443 to 0.451, Off-Topic 1.000 throughout,
Hard-Negative 0.855 to 0.839 — inside the floor. Both halves move, which is
what the floor is for: the unconstrained winner moved one half by giving up the
other.

Two changes to the system came out of it rather than out of an argument. The
Entailment judge now reads the top two Chunks and the Verdict cites the one that
entailed, which is CONTEXT.md's rule that the answer and the Evidence Span are
one decision, finally made true in the code. And the span padding is now fixed
seconds plus a fraction of the cited Chunk's own duration, because annotated
spans run from 0.16 s to 14.2 s and one offset cannot serve both ends of that.

The component sweeps of tickets 08, 09 and 11 remain, as diagnostics. They say
which judgement moved; they no longer say whether the move was worth making.

The risk this accepts is that 16 Conversations is a small fold to choose from
and the grid is large. The lower bound, the tie band and the slice floor are all
there to blunt that, and the test fold is still unread.
