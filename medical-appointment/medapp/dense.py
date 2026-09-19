"""The dense half of retrieval: one bi-encoder pass over a Conversation.

BM25 matches a Question to a Chunk by the terms they share, which is why
ADR-0002 measured 18 of 122 Positives as lexically unreachable at any k — their
evidence is a paraphrase with no content word in common, or a drug name the ASR
mis-spelled. A bi-encoder places the Question and that passage near each other
anyway, because it matches on meaning rather than on terms, and those Positives
are what it is here to rescue.

It is the wrong tool for the other half of the problem, and ADR-0001 says so:
"100 mg" and "200 mg" are near-identical to an embedding by design, so dense
retrieval on its own promotes a Hard Negative's near-miss as readily as the
truth. That is why the gate on this component is Hard-Negative accuracy
alongside recall, and why fusion with BM25 rather than replacement is what gets
measured.

The Conversation is embedded once per request rather than once per Question.
Its ten Questions share one set of Chunks, and embedding two thousand of them
ten times over would spend the answering budget on the same arithmetic.

Both sides reach the model through the same normalizer that feeds BM25, so a
Question written in symbols and the speech that spells it out are embedded as
one written form rather than two.
"""

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from medapp.config import Settings, resolved_torch_device
from medapp.config import settings as default_settings
from medapp.normalizer import normalize_text
from medapp.types import Chunk


class Embedder(Protocol):
    """The part of a sentence-transformer this module uses."""

    # The texts are positional-only because the library calls that parameter
    # something else, and the options are spelled out rather than collected as
    # keywords because the library's own signature is overloaded on them.
    def encode(
        self,
        sentences: Sequence[str],
        /,
        *,
        batch_size: int = ...,
        normalize_embeddings: bool = ...,
        show_progress_bar: bool = ...,
    ) -> Any:
        """Embed each text.

        Returns:
            One vector per text. Narrowed to a float array by the caller: the
            library's own return type is a union over torch and numpy.
        """
        ...


class DenseIndex:
    """Ranks one Conversation's Chunks by embedding similarity.

    Raises:
        ValueError: If the Conversation produced no Chunks, which leaves
            nothing to retrieve from.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        settings: Settings | None = None,
    ) -> None:
        """Embed the whole Conversation, once.

        Args:
            chunks: Every Chunk of the Conversation.
            embedder: The bi-encoder, already loaded. It is held across
                Conversations; only the embeddings here are per-request.
            settings: The resolved environment. Defaults to the process-wide
                Settings.
        """
        if not chunks:
            raise ValueError(
                "The Conversation produced no Chunks, so there is nothing to "
                "retrieve from."
            )

        self._chunks = tuple(chunks)
        self._embedder = embedder
        self._settings = settings or default_settings
        # Normalized to unit length, so the similarity between a Question and
        # a Chunk is one dot product and the ranking is one matrix multiply.
        self._embeddings = self._encode([chunk.text for chunk in self._chunks])

    def rank(self, question: str, limit: int) -> tuple[Chunk, ...]:
        """The Chunks whose meaning is closest to one Question, best first.

        Args:
            question: The Question as it was asked, normalized here into the
                written form the Chunks already carry.
            limit: How many Chunks to return. Fewer come back when the
                Conversation holds fewer.

        Returns:
            The ranked Chunks, or nothing at all when the Question normalizes
            to nothing — there is no Question left to match on, and BM25
            returns nothing in the same case, so fusing them must not invent a
            ranking here.

        Raises:
            ValueError: If ``limit`` is not positive.
        """
        if limit < 1:
            raise ValueError(f"A ranking holds at least one Chunk: got {limit!r}.")

        written = " ".join(normalize_text(question))

        if not written:
            return ()

        similarity = self._embeddings @ self._encode([written])[0]
        depth = min(limit, len(self._chunks))
        # Partition to the depth asked for and sort only that, because the
        # index runs to a few thousand Chunks and only the head is read.
        head = np.argpartition(-similarity, depth - 1)[:depth]

        return tuple(
            self._chunks[index] for index in head[np.argsort(-similarity[head])]
        )

    def _encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed text into unit vectors, in the batch Settings resolved."""
        return np.asarray(
            self._embedder.encode(
                texts,
                batch_size=self._settings.dense_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


def warm_up(embedder: Embedder, settings: Settings | None = None) -> None:
    """Embed a synthetic batch so no request pays the first forward pass.

    A full batch rather than one text, for the reason the reranker's warm-up
    records: the first pass allocates the kernels for the batch shape, and a
    Conversation is embedded a batch at a time. Warming at one text leaves the
    first real batch to pay for its own allocation inside a request.
    """
    settings = settings or default_settings

    embedder.encode(
        ["take 100 mg daily for 2 weeks"] * settings.dense_batch_size,
        batch_size=settings.dense_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def load_model(settings: Settings | None = None) -> Embedder:
    """Load the configured bi-encoder from the local model cache.

    Read offline for the same reason the reranker and the Entailment judge are:
    the deployment container has no route to the hub, and a download inside a
    request would spend the whole per-request budget before a Question was
    answered. Fill the cache with ``python -m scripts.fetch_models``.

    Raises:
        OSError: If the configured model is not in the cache.
    """
    # Imported here rather than at module scope: loading sentence-transformers
    # pulls torch in, which costs seconds, and a process running BM25 alone
    # should not pay for it.
    from sentence_transformers import SentenceTransformer

    settings = settings or default_settings

    model: Embedder = SentenceTransformer(
        settings.dense_model,
        device=resolved_torch_device(settings.torch_device),
        cache_folder=str(settings.model_cache_dir),
        local_files_only=True,
    )

    return model
