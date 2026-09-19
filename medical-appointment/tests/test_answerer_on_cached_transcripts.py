"""The shipped Answerer against a real Conversation, real weights included.

Every other test of the Answerer stubs its judges, which checks the wiring and
nothing about whether the judgements work. This one runs the shipped
configuration over a cached transcript, one Question of each kind: an Off-Topic
Question has to come back no with nulls, a Positive has to come back yes
pointing somewhere that overlaps the annotation, and the two kinds of Hard
Negative — a value swapped for a wrong one, and a plausible detail the
Conversation never established — have to come back no while the Positive they
are near-identical to still comes back yes.

It needs the transcript cache and the model cache, neither of which is in the
repository, so it is skipped where they are absent. Fill them with
``python -m scripts.transcribe_cache`` and ``python -m scripts.fetch_models``.
The fold-wide numbers are measured by
``python -m scripts.entailment_threshold``; what is under test here is that one
Conversation answered under the shipped Settings does what the sweep says it
does.
"""

import pytest

from harness import transcript_cache
from medapp.answerer import build_answerer
from medapp.config import settings
from utils import load_sample_questions, temporal_iou

# One dev Conversation, with a Question of each kind. A single Conversation
# rather than the fold: the fold is a measurement, not a test.
TRANSCRIPT_ID = "sample_17"
OFF_TOPIC = "sample_17_off_topic_q01"
POSITIVE = "sample_17_yes_q02"
# "Was the prescribed dose 200 mg daily?" against the Positive above, which is
# "Should the daily dose be 100 mg?". The same sentence with the number swapped.
VALUE_SWAP = "sample_17_hard_no_q02"
# "Has the patient been told that the condition needs no treatment?" — a
# plausible thing to have been told, which this Conversation never establishes.
NEVER_ESTABLISHED = "sample_17_hard_no_q06"

pytestmark = pytest.mark.skipif(
    not transcript_cache.is_cached(TRANSCRIPT_ID)
    or not (settings.model_cache_dir / "models--BAAI--bge-reranker-v2-m3").exists()
    or not (
        settings.model_cache_dir / "models--cross-encoder--nli-deberta-v3-base"
    ).exists(),
    reason="Needs the transcript cache, the reranker and the NLI weights.",
)


def rows() -> dict[str, dict[str, str]]:
    return {row["question_id"]: row for row in load_sample_questions()}


@pytest.fixture(scope="module")
def verdicts() -> dict[str, object]:
    """Every Question of the Conversation, answered in one pass as served."""
    asked = [
        row for row in load_sample_questions() if row["transcript_id"] == TRANSCRIPT_ID
    ]
    answerer = build_answerer(settings)

    return dict(
        zip(
            [row["question_id"] for row in asked],
            answerer.answer(
                transcript_cache.read(TRANSCRIPT_ID),
                [row["question"] for row in asked],
            ),
            strict=True,
        )
    )


def test_an_off_topic_question_is_answered_no_with_nothing_to_point_at(verdicts):
    verdict = verdicts[OFF_TOPIC]

    assert verdict.answer is False
    assert verdict.evidence is None


def test_a_positive_question_is_answered_yes_from_the_annotated_passage(verdicts):
    row = rows()[POSITIVE]
    verdict = verdicts[POSITIVE]
    annotated = (float(row["evidence_start"]), float(row["evidence_end"]))

    assert verdict.answer is True
    assert temporal_iou(annotated, verdict.evidence) > 0


def test_a_hard_negative_with_a_swapped_value_yields_no(verdicts):
    """Its best Chunk is the passage that refutes it, which Relevance cannot
    see and Entailment can."""
    verdict = verdicts[VALUE_SWAP]

    assert verdict.answer is False
    assert verdict.evidence is None


def test_a_hard_negative_the_conversation_never_established_yields_no(verdicts):
    """Neutral, not contradicted — and still a no."""
    verdict = verdicts[NEVER_ESTABLISHED]

    assert verdict.answer is False
    assert verdict.evidence is None


def test_the_positive_the_value_swap_was_built_from_still_yields_yes(verdicts):
    """The pair is the whole point: the two differ by one number, and the
    threshold that rejects one has to keep the other."""
    row = rows()[POSITIVE]
    verdict = verdicts[POSITIVE]
    annotated = (float(row["evidence_start"]), float(row["evidence_end"]))

    assert verdict.answer is True
    assert temporal_iou(annotated, verdict.evidence) > 0
