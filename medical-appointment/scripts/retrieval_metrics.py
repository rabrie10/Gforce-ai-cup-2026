"""Report the Chunker's and the retriever's gates, per fold.

ADR-0002 fixes both before the work: recall@5 >= 0.95 for the retriever, and
oracle-selected Chunk tIoU >= 0.75 for the Chunker. A gate failing here sends
work back to that component; it does not send work forward.

    python -m scripts.retrieval_metrics
    python -m scripts.retrieval_metrics --rerank
    python -m scripts.retrieval_metrics --lengths 1,2,3,5,8,13,21,34 --stride 0.25

``--rerank`` measures the ranking the endpoint returns rather than BM25's own,
which is what ADR-0002 requires the recall gate to be re-read at once a
component that reorders the candidates ships.

The ``--lengths`` and ``--stride`` options exist so the ladder can be tuned on
the dev fold against oracle tIoU without editing Settings; the values that win
become the Settings defaults.

Train and dev only. The test fold is transcribed but not read until tuning ends.
"""

import argparse

from harness.folds import FoldName
from harness.retrieval_metrics import RECALL_DEPTHS, RetrievalReport, report
from medapp.chunker import ChunkScheme
from medapp.config import settings
from medapp.reranker import CrossEncoderReranker

READABLE_FOLDS: tuple[FoldName, ...] = ("train", "dev")

# ADR-0002's usefulness thresholds, fixed before the work rather than after the
# first measurement.
RECALL_GATE_DEPTH = 5
RECALL_GATE = 0.95
ORACLE_TIOU_GATE = 0.75


def format_header() -> str:
    """The column names, one per reported depth."""
    recall_columns = "".join(f"{f'recall@{depth}':>11}" for depth in RECALL_DEPTHS)
    ranked_columns = "".join(f"{f'tIoU@{depth}':>11}" for depth in RECALL_DEPTHS)

    return (
        f"  {'fold':<6}{'convs':>7}{'spans':>7}{'chunks':>9}"
        f"{recall_columns}{'oracle':>9}{ranked_columns}"
    )


def format_report(measurement: RetrievalReport) -> str:
    """One fold as a table row, with a mark against each failed gate."""
    recalls = "".join(
        f"{measurement.recall[depth]:>10.3f}"
        + _gate_mark(depth == RECALL_GATE_DEPTH, measurement.recall[depth], RECALL_GATE)
        for depth in RECALL_DEPTHS
    )
    ranked = "".join(
        f"{measurement.ranked_tiou[depth]:>11.3f}" for depth in RECALL_DEPTHS
    )

    return (
        f"  {measurement.fold:<6}{measurement.conversations:>7}"
        f"{measurement.spans:>7}{measurement.chunks_per_conversation:>9.0f}"
        f"{recalls}"
        f"{measurement.oracle_tiou:>8.3f}"
        f"{_gate_mark(True, measurement.oracle_tiou, ORACLE_TIOU_GATE)}"
        f"{ranked}"
    )


def _gate_mark(gated: bool, value: float, gate: float) -> str:
    """A single character saying whether a gated column passed."""
    if not gated:
        return " "

    return " " if value >= gate else "!"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        help="Comma-separated Chunk lengths in normalized tokens.",
    )
    parser.add_argument(
        "--stride",
        type=float,
        help="How far apart the Chunks of one length start, as a fraction of it.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Measure the reranked ranking rather than BM25's own.",
    )
    arguments = parser.parse_args()

    scheme = ChunkScheme(
        word_lengths=(
            tuple(int(length) for length in arguments.lengths.split(","))
            if arguments.lengths
            else settings.chunk_word_lengths
        ),
        stride_fraction=arguments.stride or settings.chunk_stride_fraction,
    )

    reranker = None

    if arguments.rerank:
        reranker = CrossEncoderReranker(settings)
        reranker.warm_up()

    print(
        f"Chunks at lengths {list(scheme.word_lengths)}, "
        f"stride {scheme.stride_fraction:g} of each, ranked by "
        f"{'BM25 then ' + settings.rerank_model if reranker else 'BM25'}."
    )
    print(format_header())

    for fold in READABLE_FOLDS:
        print(format_report(report(fold, scheme, reranker)))

    print(
        f"\nGates (ADR-0002): recall@{RECALL_GATE_DEPTH} >= {RECALL_GATE:g}, "
        f"oracle tIoU >= {ORACLE_TIOU_GATE:g}; ! marks a gate that failed.\n"
        "oracle is the best tIoU any Chunk reaches, tIoU@k the best within the "
        "top k.\nThe test fold is transcribed but not read until tuning ends."
    )


if __name__ == "__main__":
    main()
