"""BM25 over one Conversation's Chunks, built per request and thrown away.

The corpus is a single Conversation — a few thousand overlapping Chunks — so
there is no index to persist and nothing to share between requests: building it
is a numpy pass over the Chunks that cost more to transcribe than to index.

Both sides go through the same normalizer before they are matched, because a
Question writes what the Conversation speaks: "41 mmol/mol" against *"41
millimoles per mole"*. Stemming on top of that is what makes "tablet" and
"tablets" one term. Nothing else is done to the text — in particular the
normalizer's compound tokens are left whole, so "135/88" stays one term rather
than becoming the two unremarkable numbers either side of the slash.
"""

from collections.abc import Sequence

import bm25s
import Stemmer

from medapp.normalizer import normalize_text
from medapp.types import Chunk

# bm25s's English list. Words every clinical Question carries and no Chunk is
# distinguished by; "no" and "not" are among them, which retrieval cannot use
# anyway — telling "no pus" from "pus" is the Entailment judge's job, and a
# term that appears in half the Chunks does not rank them.
STOPWORDS = frozenset(bm25s.stopwords.STOPWORDS_EN)


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
