# 1. Build BM25 first; adopt hybrid retrieval on evidence, not on reputation

Date: 2026-09-18

## Status

Accepted

## Context

An Evidence Span is found by retrieving the Chunk that answers a Question, so
retrieval quality bounds both halves of the score.

Hybrid retrieval — sparse and dense signals fused, then reranked by a
cross-encoder — is the established production pattern, and pure dense retrieval
alone is known to underperform it. That is the destination.

What the literature establishes is that hybrid is *usually* better. It does not
establish by how much on any particular corpus, and the mix that wins is
corpus-dependent: it turns on how much of the discriminating information is
lexical versus semantic.

This corpus is unusual in that respect. 142 of the 390 supplied Questions are
Hard Negatives, and a Hard Negative differs from the truth by one token — "100
mg" against "200 mg", "two weeks" against "six weeks". The discriminating
information is almost entirely lexical. Dense embeddings place those two strings
close together by design; BM25 treats them as different terms. So the case for
the dense half of a hybrid is weaker here than the general result implies, and
the case for the sparse half is stronger.

Amendment 2026-09-18: a reading of the supplied Hard Negatives shows the
one-token swap is only one of their shapes. Many assert something plausible the
Conversation never established or explicitly ruled out ("Should the dose be
increased?", "Were abnormal sounds heard over the lungs?"), and those share
little vocabulary with the passage that refutes them. The lexical case is
therefore weaker than the paragraph above implies, and the dense half has a
real chance of winning.

That is a reason to measure, not a reason to conclude. It cuts both ways: dense
retrieval may still recover paraphrased Questions that share no vocabulary with
the passage, and those exist too. We do not know the balance, and on this corpus
it is cheap to find out.

## Decision

Build BM25 retrieval first and establish it as the measured baseline. Add dense
retrieval and fusion as the next increment, and keep it if it wins.

The gate is **Hard-Negative accuracy**, evaluated alongside recall@k rather than
after it. Recall@k alone can improve while the thing we actually need gets
worse: fusion can promote a lexically near-identical but factually wrong Chunk
above the correct one, raising recall of *something relevant* while destroying
the distinction 142 Questions depend on.

The cross-encoder reranker is built regardless of which retriever wins. It
addresses a different failure: reading Question and Chunk jointly is what
separates "100 mg" from "200 mg", and no bag-of-words score or bi-encoder
similarity can do that at any k.

## Consequences

Either outcome ships with a number behind it. If hybrid wins we have Huyen's
pattern and the evidence for it; if BM25 wins we have a simpler system and the
evidence for that. Neither is a guess, and neither is a shortcut.

Sequencing this way also makes the dense half debuggable. A BM25 baseline that
already works is the reference that tells us whether a fusion change helped,
which is not available if both are built at once.

Fusion method is measured on the same gate rather than assumed. Reciprocal Rank
Fusion is the starting point, not the answer.

If dense retrieval loses, the embedder never loads in the request path, and its
memory and latency go to a larger reranker or ASR model instead.

### Outcome 2026-09-19: BM25 keeps the gate; the embedder does not ship

Measured by `python -m scripts.retrieval_modes` on the dev fold — 16
Conversations, 160 Questions, 75 annotated Evidence Spans. Every mode is
answered end to end through the shipped Answerer's own path (retrieve, rerank,
rewrite the Question as a Claim, judge Entailment at the Settings thresholds),
so the Hard-Negative number is the one the system scores rather than a proxy
for it. `BAAI/bge-small-en-v1.5` is the embedder; fusion is Reciprocal Rank
Fusion over the top 50 of each ranking at k=60.

| Mode   | Hard-neg | 90% interval   | recall@5 | 90% interval   | Positive | Off-topic | Accuracy |
|--------|----------|----------------|----------|----------------|----------|-----------|----------|
| bm25   | 0.855    | [0.787, 0.918] | 0.667    | [0.571, 0.767] | 0.867    | 1.000     | 0.881    |
| dense  | 0.790    | [0.703, 0.875] | 0.733    | [0.653, 0.811] | 0.907    | 1.000     | 0.875    |
| hybrid | 0.823    | [0.745, 0.894] | 0.707    | [0.622, 0.795] | 0.880    | 1.000     | 0.875    |

**BM25 stands.** Both candidates raise recall@5 and lower Hard-Negative
accuracy, which is precisely the trade this ADR named before any of it was
built: recall of *something relevant* improves while the distinction 142
Questions depend on gets worse. Dense buys 0.066 of recall and gives up 0.065
of the gate; fusion splits the difference in both directions. Had the gate been
recall alone, dense would have been adopted and the shipped accuracy would have
fallen from 0.881 to 0.875.

The mechanism is visible in the Positive column. Dense retrieval answers *more*
Positives correctly (0.907 against 0.867) — those are the paraphrased Questions
that share no content word with their evidence, which is exactly what the dense
half was expected to rescue. It pays for them by putting a Hard Negative's
near-miss in front of the reranker often enough to cost more than it gained.
The embedding does what the literature says; this corpus is the case where that
is the wrong thing to do.

Oracle-selected tIoU is 0.835 under all three modes, as it must be — it is the
best any Chunk achieves and no retriever makes new Chunks. It is reported per
mode as a check that the measurement is measuring what it claims.

**Cost, had it won.** Embedding one Conversation's Chunks costs about 1.48 s on
the Mac's CPU on average and 3.05 s on the longest Conversation of the fold —
the pass scales with the Chunk count, which runs from 1,228 to 4,376 — paid
once per request rather than per Question, against 20 ms to build the BM25
index. A whole Conversation is then 4.3 s of answering against the 15 s the
budget gives that half, where BM25 is 2.1 s. So the dense half is affordable
and that is not why it was rejected; it was rejected on the gate.

These are means over the fold and the index figures are stable between runs.
The per-Question *worst* times are not: a single-sample maximum over 160
Questions swung by a factor of two between runs on a shared laptop CPU, which
is why the script reports them without judging the budget off them. ADR-0002
states the latency gate on the deployment VM, and that is where it is checked.

**What this decides.** `MEDAPP_RETRIEVAL_MODE` defaults to `bm25` and the
embedder is never constructed in the request path — `build_index_factory` only
loads one for a mode that names it, and a mode that names one without it is a
startup failure rather than a silent fall back to BM25. All three modes remain
constructible and measurable, because the decision is a measurement that can be
re-run rather than code that was deleted.
