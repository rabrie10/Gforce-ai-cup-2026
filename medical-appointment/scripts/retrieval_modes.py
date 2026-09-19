"""Measure BM25, dense and fused retrieval on the dev fold, and apply the gate.

    python -m scripts.retrieval_modes
    python -m scripts.retrieval_modes --fold train --modes bm25,hybrid

ADR-0001 fixed the gate before any of this was built: the dense half is adopted
only where it wins on **Hard-Negative accuracy** as well as recall@5, because
recall alone can rise while the distinction 142 Questions depend on falls. So
every mode is answered end to end through the shipped Answerer's path and both
halves are printed side by side, with bootstrap intervals at Conversation
level.

Whichever mode the gate adopts becomes ``MEDAPP_RETRIEVAL_MODE``'s default in
Settings, with these numbers recorded beside it and in ADR-0001.

The embedder is loaded from the local cache, so fill it first — and with the
shipped ``bm25`` default the fetcher leaves it out, because the request path
does not read it::

    python -m scripts.fetch_models --all

Train and dev only. The test fold is transcribed but not read until tuning ends.
"""

import argparse

from harness.folds import FoldName
from harness.retrieval_modes import ModeComparison, ModeReport, compare
from medapp.claims import ClaimRewriter
from medapp.config import RetrievalMode, checked_retrieval_mode, settings
from medapp.dense import load_model as load_embedder
from medapp.dense import warm_up as warm_embedder
from medapp.entailment import NliEntailmentJudge
from medapp.reranker import CrossEncoderReranker

READABLE_FOLDS: tuple[FoldName, ...] = ("train", "dev")
ALL_MODES: tuple[RetrievalMode, ...] = ("bm25", "dense", "hybrid")

# The spec's latency budget gives all ten Questions of a Conversation 15 s
# worst case, and the index is built once inside that.
QUESTIONS_PER_CONVERSATION = 10
ANSWERING_BUDGET_SECONDS = 15.0


def format_answer_header() -> str:
    """The column names of the answers table."""
    return (
        f"  {'mode':<8}{'hard-neg':>9}{'90% interval':>18}{'positive':>10}"
        f"{'90% interval':>18}{'off-topic':>11}{'accuracy':>10}"
        f"{'90% interval':>18}"
    )


def format_answers(report: ModeReport, chosen: bool) -> str:
    """One mode's answers as a table row, with the adopted one marked.

    Hard-Negative accuracy is the gate; Positive accuracy is the same number as
    the true positive rate and is what the gate must not be bought from.
    """
    return (
        f"{'*' if chosen else ' '} {report.mode:<8}"
        f"{report.hard_negative_accuracy:>7.3f}"
        f"{_format(report.hard_negative_interval):>20}"
        f"{report.positive_accuracy:>10.3f}"
        f"{_format(report.positive_interval):>18}"
        f"{report.off_topic_accuracy:>11.3f}{report.accuracy:>10.3f}"
        f"{_format(report.accuracy_interval):>18}"
    )


def format_retrieval_header() -> str:
    """The column names of the retrieval table."""
    return (
        f"  {'mode':<8}{'recall@5':>10}{'90% interval':>18}"
        f"{'oracle tIoU':>13}{'90% interval':>18}"
    )


def format_retrieval(report: ModeReport, chosen: bool) -> str:
    """One mode's retrieval as a table row.

    Oracle tIoU is the Chunker's ceiling and no retriever moves it, so the
    three rows agreeing is the check that this is measuring what it claims.
    """
    return (
        f"{'*' if chosen else ' '} {report.mode:<8}"
        f"{report.recall_at_5:>10.3f}{_format(report.recall_interval):>18}"
        f"{report.oracle_tiou:>13.3f}{_format(report.oracle_interval):>18}"
    )


def _format(interval: tuple[float, float]) -> str:
    """One bootstrap interval as a table cell."""
    low, high = interval

    return f"[{low:.3f}, {high:.3f}]"


def format_cost(report: ModeReport) -> str:
    """What one mode costs a request.

    Reported rather than judged against the budget. The index build is stable
    between runs and is the dense half's real cost; the worst *judging* time is
    a single-sample maximum over 160 Questions and swung by a factor of two
    between runs on a shared laptop CPU, which is machine noise rather than a
    property of the mode. ADR-0002 states the latency gate on the deployment VM
    for that reason, and it is checked there.
    """
    conversation = (
        report.index_seconds + report.mean_judging_seconds * QUESTIONS_PER_CONVERSATION
    )

    return (
        f"  {report.mode:<8} index {report.index_seconds * 1000:>6.0f} ms per "
        f"Conversation ({report.worst_index_seconds * 1000:.0f} ms worst), "
        f"judging {report.mean_judging_seconds * 1000:>4.0f} ms per Question "
        f"({report.worst_judging_seconds * 1000:.0f} ms worst); a Conversation "
        f"is {conversation:.1f} s of answering on these means, against "
        f"{ANSWERING_BUDGET_SECONDS:g} s"
    )


def format_comparison(comparison: ModeComparison) -> str:
    """The table, what the gate adopted, and what each mode costs."""
    baseline = comparison.baseline
    lines = [
        f"{comparison.fold}: {baseline.conversations} Conversations, "
        f"{baseline.questions} Questions, {baseline.spans} annotated Evidence "
        "Spans",
        "",
        "Answers:",
        format_answer_header(),
    ]
    lines += [
        format_answers(report, report is comparison.chosen)
        for report in comparison.reports
    ]
    lines += ["", "Retrieval:", format_retrieval_header()]
    lines += [
        format_retrieval(report, report is comparison.chosen)
        for report in comparison.reports
    ]
    lines += ["", "Cost per request:"]
    lines += [format_cost(report) for report in comparison.reports]

    chosen = comparison.chosen
    lines += [
        "",
        f"  adopted: {chosen.mode}"
        + (
            " — the baseline stands, so the embedder does not load in the "
            "request path"
            if chosen is baseline
            else f" — recall@5 {chosen.recall_at_5:.3f} against "
            f"{baseline.recall_at_5:.3f} and Hard-Negative accuracy "
            f"{chosen.hard_negative_accuracy:.3f} "
            f"{_format(chosen.hard_negative_interval)} against "
            f"{baseline.hard_negative_accuracy:.3f}"
        ),
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold",
        choices=READABLE_FOLDS,
        default="dev",
        help="Which fold to measure. The mode is chosen on dev.",
    )
    parser.add_argument(
        "--modes",
        help="Comma-separated modes to measure. Defaults to all three; BM25 "
        "is always among them, as the baseline the gate adopts against.",
    )
    arguments = parser.parse_args()

    # Checked at the boundary it arrives as free text, the way --fold is by
    # argparse: a typo that reached the factory would be measured as whichever
    # mode it fell through to and printed under the name that was typed.
    modes: tuple[RetrievalMode, ...] = (
        tuple(
            checked_retrieval_mode(mode.strip()) for mode in arguments.modes.split(",")
        )
        if arguments.modes
        else ALL_MODES
    )

    print(
        f"Ranking with {settings.dense_model} against BM25, reranking the top "
        f"{settings.retrieval_candidates} with {settings.rerank_model} and "
        f"judging with {settings.nli_model} on {settings.device}."
    )

    # Every model is warmed before the clock starts, exactly as the served
    # process warms them at import.
    reranker = CrossEncoderReranker(settings)
    reranker.warm_up()

    judge = NliEntailmentJudge(settings)
    judge.warm_up()

    embedder = load_embedder(settings) if set(modes) - {"bm25"} else None
    if embedder is not None:
        warm_embedder(embedder, settings)

    print(
        format_comparison(
            compare(
                fold=arguments.fold,
                settings=settings,
                reranker=reranker,
                rewriter=ClaimRewriter(),
                judge=judge,
                embedder=embedder,
                modes=modes,
            )
        )
    )

    print(
        "\nThe gate is Hard-Negative accuracy alongside recall@5, not recall "
        "alone: fusion can\npromote a lexically near-identical but factually "
        "wrong Chunk and raise one while\nlowering the other. A candidate is "
        "adopted only where recall@5 beats the baseline and\nthe "
        "Hard-Negative gain survives resampling; ties go to BM25, which is "
        "the simpler system.\nThe test fold is transcribed but not read until "
        "tuning ends."
    )


if __name__ == "__main__":
    main()
