from dataclasses import dataclass

from harness import transcript_cache
from harness.folds import FoldName, conversations_in_fold
from medapp.chunker import SentenceScheme, chunk_conversation
from medapp.reranker import Reranker
from medapp.types import Chunk
from utils import Span, gold_evidence, temporal_iou

# The depths recall is reported at. 1 is what the Evidence Span is read from
# today, 5 is the gate, and 10 is how deep the reranker will be handed.
RECALL_DEPTHS: tuple[int, ...] = (1, 5, 10)

# What counts as having retrieved the evidence: a Chunk that overlaps the
# annotated span more than the two of them miss each other. Deliberately looser
# than the Chunker's 0.75 gate, because the two measure different things —
# whether the ranking found the right place, not whether the boundaries are
# already good enough to return.
HIT_TIOU = 0.5


@dataclass(frozen=True, slots=True)
class SpanMeasurement:
    """What one annotated Evidence Span got from the Chunker and the retriever.

    Attributes:
        oracle_tiou: The best tIoU any Chunk of the Conversation achieves,
            whether or not the retriever ranked it anywhere.
        ranked_tiou: The best tIoU within the top k, per depth — what is
            actually reachable once ranking has had its say.
    """

    question_id: str
    transcript_id: str
    oracle_tiou: float
    ranked_tiou: dict[int, float]


@dataclass(frozen=True, slots=True)
class FoldMeasurement:
    """Every annotated span of one fold and the indexes they were measured on.

    Attributes:
        chunk_counts: How many Chunks each Conversation of the fold produced,
            which is the candidate count the gates are bought at.
    """

    spans: tuple[SpanMeasurement, ...]
    chunk_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """One fold measured against both gates.

    Attributes:
        reranked: Whether the ranking measured is the reranker's or BM25's.
        chunks_per_conversation: The mean candidate count per Conversation, which
            is the cost the gates are bought at.
        recall: Fraction of spans found, per depth.
        oracle_tiou: Mean best-achievable tIoU over every Chunk — the Chunker's
            gate.
        ranked_tiou: Mean best tIoU within the top k, per depth.
    """

    fold: FoldName
    reranked: bool
    conversations: int
    spans: int
    chunks_per_conversation: float
    recall: dict[int, float]
    oracle_tiou: float
    ranked_tiou: dict[int, float]


def measure_span(
    chunks: tuple[Chunk, ...], ranked: tuple[Chunk, ...], annotated: Span
) -> tuple[float, dict[int, float]]:
    """The oracle and per-depth tIoU one annotated span achieves.

    Args:
        chunks: Every Chunk of the Conversation.
        ranked: The same Chunks as the retriever ordered them, best first.
        annotated: The annotated Evidence Span.

    Returns:
        The best tIoU over all Chunks, and the best tIoU within each depth in
        :data:`RECALL_DEPTHS`.
    """
    oracle = max((temporal_iou(annotated, chunk.span) for chunk in chunks), default=0.0)

    ranked_tious = [temporal_iou(annotated, chunk.span) for chunk in ranked]
    at_depth = {
        depth: max(ranked_tious[:depth], default=0.0) for depth in RECALL_DEPTHS
    }

    return oracle, at_depth


def measure_fold(
    fold: FoldName, scheme: SentenceScheme, reranker: Reranker
) -> FoldMeasurement:
    """Measure every annotated Evidence Span of one fold.

    The Chunks are built once per Conversation and every Question
    of it is ranked against them, which is exactly what happens inside one
    request.

    Args:
        fold: Which fold to read.
        scheme: The Chunk granularities to measure.
        reranker: Ranks the Chunks before they are measured.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
    """
    spans: list[SpanMeasurement] = []
    chunk_counts: list[int] = []
    depth = max(RECALL_DEPTHS)

    for transcript_id, rows in conversations_in_fold(fold):
        chunks = chunk_conversation(transcript_cache.read(transcript_id), scheme)
        chunk_counts.append(len(chunks))

        for row in rows:
            annotated = gold_evidence(row)

            if annotated is None:
                continue

            ranked = tuple(
                candidate.chunk
                for candidate in reranker.rerank(row["question"], chunks)[:depth]
            )

            oracle, at_depth = measure_span(chunks, ranked, annotated)
            spans.append(
                SpanMeasurement(
                    question_id=row["question_id"],
                    transcript_id=transcript_id,
                    oracle_tiou=oracle,
                    ranked_tiou=at_depth,
                )
            )

    return FoldMeasurement(spans=tuple(spans), chunk_counts=tuple(chunk_counts))


def report(
    fold: FoldName, scheme: SentenceScheme, reranker: Reranker
) -> RetrievalReport:
    """Summarise one fold against both gates.

    Raises:
        FileNotFoundError: If a Conversation of the fold is not cached.
        ValueError: If the fold holds no annotated Evidence Spans, which would
            leave both gates measured over nothing.
    """
    measurement = measure_fold(fold, scheme, reranker)

    if not measurement.spans:
        raise ValueError(
            f"The {fold} fold holds no annotated Evidence Spans, so there is "
            "nothing to measure retrieval against."
        )

    return RetrievalReport(
        fold=fold,
        reranked=reranker is not None,
        conversations=len(measurement.chunk_counts),
        spans=len(measurement.spans),
        chunks_per_conversation=_mean(
            [float(count) for count in measurement.chunk_counts]
        ),
        recall={
            depth: _mean(
                [
                    float(span.ranked_tiou[depth] >= HIT_TIOU)
                    for span in measurement.spans
                ]
            )
            for depth in RECALL_DEPTHS
        },
        oracle_tiou=_mean([span.oracle_tiou for span in measurement.spans]),
        ranked_tiou={
            depth: _mean([span.ranked_tiou[depth] for span in measurement.spans])
            for depth in RECALL_DEPTHS
        },
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
