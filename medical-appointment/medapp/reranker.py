from collections.abc import Sequence
from typing import Protocol

from medapp.config import Settings, resolved_torch_device
from medapp.config import settings as default_settings
from medapp.normalizer import normalize_text
from medapp.types import Chunk, ScoredChunk


class ScoringModel(Protocol):
    """The part of a cross-encoder this module uses."""

    def predict(
        self, inputs: Sequence[tuple[str, str]], **options: object
    ) -> Sequence[float]: ...


class Reranker(Protocol):
    """Scores how far a Chunk is about what a Question asks about."""

    def rerank(self, question: str, chunks: Sequence[Chunk]) -> tuple[ScoredChunk, ...]:
        """Rescore retrieved Chunks against one Question, best first."""
        ...


class CrossEncoderReranker:
    """Rescores Chunks with a cross-encoder loaded from the local cache."""

    def __init__(
        self, settings: Settings | None = None, model: ScoringModel | None = None
    ) -> None:
        """Load the scoring model, unless one is supplied.

        Args:
            settings: The resolved environment. Defaults to the process-wide
                Settings.
            model: An already-constructed cross-encoder. The argument exists so
                tests and the dev-loop scripts can hold one model across many
                Conversations rather than reloading weights per call.
        """
        self._settings = settings or default_settings
        self._model = model if model is not None else load_model(self._settings)

    def rerank(self, question: str, chunks: Sequence[Chunk]) -> tuple[ScoredChunk, ...]:
        """Score every candidate against one Question and order them by it.

        Args:
            question: The Question as it was asked; normalized here into the
                written form the Chunks already carry.
            chunks: The retrieved candidates, in whatever order they arrived.

        Returns:
            Every candidate with its Relevance, best first. Ties keep the order
            they arrived in, which is the retriever's.
        """
        if not chunks:
            return ()

        written = " ".join(normalize_text(question))
        scores = self._model.predict(
            [(written, chunk.text) for chunk in chunks],
            batch_size=self._settings.rerank_batch_size,
            show_progress_bar=False,
        )

        scored = [
            ScoredChunk(chunk=chunk, relevance=float(score))
            for chunk, score in zip(chunks, scores, strict=True)
        ]

        return tuple(sorted(scored, key=lambda candidate: -candidate.relevance))

    def warm_up(self) -> None:
        """Score a synthetic batch so no request pays the first forward pass.

        Weights are memory-mapped and the first pass allocates the kernels for
        the batch shape, both of which would otherwise land inside a request
        that has seconds to spare. The batch is a whole Question's candidate
        list rather than one pair, because the allocation is per shape: warming
        at one pair leaves the first real batch to pay for its own, which was
        measured at seconds rather than milliseconds.
        """
        self._model.predict(
            [("is the dose 100 mg", "take 100 mg daily for two weeks")]
            * self._settings.retrieval_candidates,
            batch_size=self._settings.rerank_batch_size,
            show_progress_bar=False,
        )


def load_model(settings: Settings | None = None) -> ScoringModel:
    """Load the configured cross-encoder from the local model cache.

    The hub is not reachable from the deployment container and must not be
    reached from the request path in any case, so the weights are read offline:
    a model that was never pre-fetched is a startup failure rather than a
    download inside a request. Fill the cache with
    ``python -m scripts.fetch_models``.

    Raises:
        OSError: If the configured model is not in the cache.
    """
    # Imported here rather than at module scope: loading sentence-transformers
    # pulls torch in, which costs seconds, and the harness modules that only
    # need the Chunker should not pay for it.
    from sentence_transformers import CrossEncoder

    settings = settings or default_settings

    model: ScoringModel = CrossEncoder(
        settings.rerank_model,
        device=resolved_torch_device(settings.torch_device),
        cache_folder=str(settings.model_cache_dir),
        local_files_only=True,
        max_length=settings.rerank_max_tokens,
    )

    return model
