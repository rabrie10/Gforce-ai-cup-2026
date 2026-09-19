"""The Entailment judgement: does the Chunk establish what the Claim asserts?

Relevance answers one failure and cannot answer the other. A Hard Negative's
best Chunk scores very high on Relevance precisely because it is lexically
near-identical to the truth — the right drug at the wrong dose is *about* the
drug — so the threshold that would reject it would reject the Positives it
exists to keep. What separates them is whether the passage actually establishes
the Claim, which is the judgement a natural-language-inference model makes.

The model returns three probabilities over one premise-hypothesis pair, and
CONTEXT.md is explicit about how they are read: yes requires entailment;
nothing less. A Hard Negative's Chunk either contradicts the Claim — the wrong
dose is stated — or is silent on it, and a plausible detail that was never
agreed is neutral rather than contradicted. Both are a no, so only the
entailment probability is thresholded and the other two are carried for the
error analysis rather than for the decision.

The premise is the Chunk in the normalizer's written form and the hypothesis is
the Claim reduced to the same form, so that "41 mmol/mol" written and *"41
millimoles per mole"* spoken reach the model as one claim rather than two — the
same reduction the reranker applies, for the same reason.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from medapp.config import Settings, resolved_torch_device
from medapp.config import settings as default_settings
from medapp.normalizer import normalize_text
from medapp.types import Chunk

# The three judgements an NLI model makes, as the model's own config names
# them. Read from the loaded model rather than assumed: the label order differs
# between checkpoints, and a silently transposed order would threshold
# contradiction as though it were entailment.
CONTRADICTION = "contradiction"
ENTAILMENT = "entailment"
NEUTRAL = "neutral"
NLI_LABELS = frozenset({CONTRADICTION, ENTAILMENT, NEUTRAL})


@dataclass(frozen=True, slots=True)
class Judgement:
    """What the Entailment judge made of one Chunk against one Claim.

    Attributes:
        entailment: The probability the Chunk establishes the Claim. This alone
            decides the answer.
        neutral: The probability the Chunk is silent on it — a plausible detail
            that was never agreed, which is a no.
        contradiction: The probability the Chunk refutes it, which is also a
            no. Kept apart from neutral because the two are different failures
            and the error analysis reads them separately.
    """

    chunk: Chunk
    entailment: float
    neutral: float
    contradiction: float


class NliModel(Protocol):
    """The part of an NLI cross-encoder this module uses."""

    config: object

    def predict(
        self, inputs: Sequence[tuple[str, str]], **options: object
    ) -> Sequence[Sequence[float]]: ...


class EntailmentJudge(Protocol):
    """Judges whether a Chunk establishes what a Claim asserts."""

    def judge(self, claim: str, chunks: Sequence[Chunk]) -> tuple[Judgement, ...]:
        """Judge each Chunk against one Claim, in the order they arrived."""
        ...


class NliEntailmentJudge:
    """Judges Entailment with an NLI cross-encoder from the local cache."""

    def __init__(
        self, settings: Settings | None = None, model: NliModel | None = None
    ) -> None:
        """Load the NLI model, unless one is supplied.

        Args:
            settings: The resolved environment. Defaults to the process-wide
                Settings.
            model: An already-constructed NLI cross-encoder. The argument
                exists so tests and the dev-loop scripts can hold one model
                across many Conversations rather than reloading weights per
                call.

        Raises:
            ValueError: If the model does not label its outputs with the three
                NLI classes, which would leave the entailment probability
                unidentifiable.
        """
        self._settings = settings or default_settings
        self._model = model if model is not None else load_model(self._settings)
        self._labels = _label_positions(self._model)

    def judge(self, claim: str, chunks: Sequence[Chunk]) -> tuple[Judgement, ...]:
        """Judge every Chunk against one Claim.

        Args:
            claim: The Claim as the rewriter wrote it; normalized here into the
                written form the Chunks already carry.
            chunks: The Chunks to judge, in whatever order they arrived.

        Returns:
            One Judgement per Chunk, in the order the Chunks arrived. The
            ordering is the caller's: Relevance chose which Chunk to judge, and
            reordering by entailment here would silently substitute a different
            Chunk for the one the Evidence Span is read from.
        """
        if not chunks:
            return ()

        written = " ".join(normalize_text(claim))
        scores = self._model.predict(
            [(chunk.text, written) for chunk in chunks],
            batch_size=self._settings.nli_batch_size,
            apply_softmax=True,
            show_progress_bar=False,
        )

        return tuple(
            Judgement(
                chunk=chunk,
                entailment=float(row[self._labels[ENTAILMENT]]),
                neutral=float(row[self._labels[NEUTRAL]]),
                contradiction=float(row[self._labels[CONTRADICTION]]),
            )
            for chunk, row in zip(chunks, scores, strict=True)
        )

    def warm_up(self) -> None:
        """Judge one synthetic pair so no request pays the first forward pass."""
        self._model.predict(
            [("take 100 mg daily for 2 weeks", "the patient does take 100 mg")],
            batch_size=1,
            apply_softmax=True,
            show_progress_bar=False,
        )


def _label_positions(model: NliModel) -> dict[str, int]:
    """Which output column carries which judgement, from the model's own config.

    Raises:
        ValueError: If the model does not label all three NLI classes.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    positions = {
        str(label).lower(): int(index) for index, label in dict(id2label).items()
    }

    if not positions.keys() >= NLI_LABELS:
        raise ValueError(
            f"The configured NLI model labels its outputs {sorted(positions)}, "
            f"not {sorted(NLI_LABELS)}, so the entailment probability cannot be "
            "identified. MEDAPP_NLI_MODEL must name a natural-language-"
            "inference cross-encoder."
        )

    return positions


def load_model(settings: Settings | None = None) -> NliModel:
    """Load the configured NLI cross-encoder from the local model cache.

    Read offline for the same reason the reranker is: the deployment container
    has no route to the hub, and a download inside a request would spend the
    whole per-request budget before a Question was answered. Fill the cache
    with ``python -m scripts.fetch_models``.

    Raises:
        OSError: If the configured model is not in the cache.
    """
    # Imported here rather than at module scope: loading sentence-transformers
    # pulls torch in, which costs seconds, and the harness modules that only
    # need the Chunker should not pay for it.
    from sentence_transformers import CrossEncoder

    settings = settings or default_settings

    model: NliModel = CrossEncoder(
        settings.nli_model,
        device=resolved_torch_device(settings.torch_device),
        cache_folder=str(settings.model_cache_dir),
        local_files_only=True,
        max_length=settings.nli_max_tokens,
    )

    return model
