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
