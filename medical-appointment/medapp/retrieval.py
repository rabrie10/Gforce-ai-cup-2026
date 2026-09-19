"""Ranked retrieval over one Conversation's Chunks, built per request.

The corpus is a single Conversation — a few thousand overlapping Chunks — so
there is no index to persist and nothing to share between requests: building it
is a numpy pass over the Chunks that cost more to transcribe than to index.

Both sides go through the same normalizer before they are matched, because a
Question writes what the Conversation speaks: "41 mmol/mol" against *"41
millimoles per mole"*. Stemming on top of that is what makes "tablet" and
"tablets" one term. Nothing else is done to the text — in particular the
normalizer's compound tokens are left whole, so "135/88" stays one term rather
than becoming the two unremarkable numbers either side of the slash.

Which retriever the Answerer gets is Settings' to say. ADR-0001 builds BM25
first as the measured baseline and adopts the dense half only where fusing the
two wins on Hard-Negative accuracy as well as on recall, so all three
arrangements are constructible and the one that lost costs the request path
nothing — in particular the embedder is never loaded for a mode that does not
name it.
"""

from collections.abc import Callable, Sequence
from typing import Protocol

import bm25s
import Stemmer

from medapp.config import Settings, checked_retrieval_mode
from medapp.config import settings as default_settings
from medapp.dense import DenseIndex, Embedder
from medapp.normalizer import normalize_text
from medapp.types import Chunk

# bm25s's English list. Words every clinical Question carries and no Chunk is
# distinguished by; "no" and "not" are among them, which retrieval cannot use
# anyway — telling "no pus" from "pus" is the Entailment judge's job, and a
# term that appears in half the Chunks does not rank them.
STOPWORDS = frozenset(bm25s.stopwords.STOPWORDS_EN)


class Retriever(Protocol):
    """Ranks one Conversation's Chunks against a Question."""

    def rank(self, question: str, limit: int) -> tuple[Chunk, ...]:
        """The Chunks that best match one Question, best first."""
        ...


class Bm25Index:
    """The ranked-retrieval half of the Answerer, over one Conversation.

    Raises:
        ValueError: If the Conversation produced no Chunks, which leaves
            nothing to retrieve.
    """

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError(
                "The Conversation produced no Chunks, so there is nothing to "
                "retrieve from."
            )

        self._chunks = tuple(chunks)
        # PyStemmer objects hold decoding state and are not safe to share
        # across threads; one per index costs a few microseconds.
        self._stemmer = Stemmer.Stemmer("english")
        self._index = bm25s.BM25()
        self._index.index(
            [self._terms(chunk.text.split()) for chunk in self._chunks],
            show_progress=False,
        )

    def rank(self, question: str, limit: int) -> tuple[Chunk, ...]:
        """The Chunks that best match one Question, best first.

        Args:
            question: The Question as it was asked, normalized here so that it
                and the transcript reduce to the same tokens.
            limit: How many Chunks to return. Fewer come back when the
                Conversation holds fewer.

        Returns:
            The ranked Chunks, or nothing at all if the Question carries no
            term the index can match on — there is no passage to read an answer
            from in that case, and an arbitrary one would be worse than none.

        Raises:
            ValueError: If ``limit`` is not positive.
        """
        if limit < 1:
            raise ValueError(f"A ranking holds at least one Chunk: got {limit!r}.")

        terms = self._terms(normalize_text(question))

        if not terms:
            return ()

        indices, _ = self._index.retrieve(
            [terms], k=min(limit, len(self._chunks)), show_progress=False
        )

        return tuple(self._chunks[index] for index in indices[0])

    def _terms(self, tokens: Sequence[str]) -> list[str]:
        """Normalized tokens reduced to the terms the index matches on."""
        return [
            term
            for term in self._stemmer.stemWords(
                [token for token in tokens if token not in STOPWORDS]
            )
            if term
        ]


class FusedIndex:
    """Reciprocal Rank Fusion over two or more rankings of the same Chunks.

    Each retriever's ranking votes for a Chunk with ``1 / (k + rank)``, and the
    votes are added. What that buys over adding the scores themselves is that
    nothing has to be calibrated: a BM25 score and a cosine similarity are not
    on one scale and no weighting makes them so, where their *ranks* are
    directly comparable. The constant k sets how steeply a top rank outweighs a
    deep one.

    A Chunk only one retriever found still scores, which is the point — the
    Positives ADR-0002 measured as lexically unreachable are exactly the ones
    BM25 never ranks at all.

    Raises:
        ValueError: If there is only one ranking, which is not a fusion.
    """

    def __init__(
        self, retrievers: Sequence[Retriever], settings: Settings | None = None
    ) -> None:
        """Fix the retrievers fused and how deep each is read.

        Args:
            retrievers: The rankings to fuse, in the order ties are broken by.
            settings: The resolved environment. Defaults to the process-wide
                Settings.
        """
        if len(retrievers) < 2:
            raise ValueError(
                f"Fusion combines two or more rankings: got {len(retrievers)}."
            )

        self._retrievers = tuple(retrievers)
        self._settings = settings or default_settings

    def rank(self, question: str, limit: int) -> tuple[Chunk, ...]:
        """The Chunks the retrievers agree on, best first.

        Args:
            question: The Question as it was asked; each retriever normalizes
                it for itself.
            limit: How many Chunks to return.

        Returns:
            The fused ranking. Empty when no retriever ranked anything, which
            is a Question with no passage to read an answer from.

        Raises:
            ValueError: If ``limit`` is not positive.
        """
        if limit < 1:
            raise ValueError(f"A ranking holds at least one Chunk: got {limit!r}.")

        constant = self._settings.fusion_rank_constant
        # Never shallower than what was asked for: fusing two top-50s cannot
        # return 100 Chunks, and a caller that wants a deeper candidate list
        # than the fusion depth would silently get a short one.
        depth = max(limit, self._settings.fusion_depth)
        votes: dict[Chunk, float] = {}
        first_seen: dict[Chunk, int] = {}

        for retriever in self._retrievers:
            for rank, chunk in enumerate(retriever.rank(question, depth), start=1):
                votes[chunk] = votes.get(chunk, 0.0) + 1 / (constant + rank)
                first_seen.setdefault(chunk, len(first_seen))

        return tuple(
            sorted(votes, key=lambda chunk: (-votes[chunk], first_seen[chunk]))
        )[:limit]


def build_index_factory(
    settings: Settings | None = None, embedder: Embedder | None = None
) -> Callable[[Sequence[Chunk]], Retriever]:
    """How the Answerer builds its per-request index, as Settings name it.

    Args:
        settings: The resolved environment. Defaults to the process-wide
            Settings.
        embedder: The bi-encoder, already loaded, for the modes that rank with
            one. Held across Conversations: only the embeddings are
            per-request.

    Raises:
        ValueError: If the mode names no retriever, or ranks with an embedder
            and none was supplied. A startup failure rather than a quiet fall
            back to BM25, which would report the wrong mode's numbers under the
            right mode's name.
    """
    settings = settings or default_settings
    # Checked rather than trusted: ``Settings.model_copy(update=...)`` does not
    # validate, and a sweep that builds one Settings per mode reaches here with
    # whatever string it was handed. A typo falling through to a mode that does
    # build would be measured, and then recorded under the name that was typed.
    mode = checked_retrieval_mode(settings.retrieval_mode)

    if mode == "bm25":
        return Bm25Index

    if embedder is None:
        raise ValueError(
            f"The {mode!r} retrieval mode ranks with an embedder and none was supplied."
        )

    if mode == "dense":
        return lambda chunks: DenseIndex(chunks, embedder, settings)

    if mode == "hybrid":
        return lambda chunks: FusedIndex(
            (Bm25Index(chunks), DenseIndex(chunks, embedder, settings)), settings
        )

    # Named in the Literal and built by nothing here, which is a mode that was
    # added without a retriever rather than a typo.
    raise ValueError(f"No retriever implements the {mode!r} retrieval mode.")
