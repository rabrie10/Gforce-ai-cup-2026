"""The fidelity check: what it counts as a fault, and what it forgives.

The check is what stands between a rewriter regression and a threshold that
absorbs it, so the check itself is tested against rewriters that fail one
property at a time. The rewriter under measurement is stubbed for that; the
real one is measured over the folds by ``python -m scripts.claim_fidelity``.
"""

import pytest

from harness.claim_fidelity import check
from medapp.claims import Claim, ClaimRewriter, ClaimShape, load_parser


class _Rewrites:
    """A rewriter that returns one fixed Claim, whatever it is asked."""

    def __init__(self, text: str, shape: ClaimShape = "auxiliary_fronted") -> None:
        self._claim = Claim(text=text, shape=shape)

    def rewrite(self, question: str) -> Claim:
        return self._claim


@pytest.fixture(scope="module")
def parser():
    return load_parser()


def faults(question: str, claim: str, parser) -> tuple[str, ...]:
    return check("q1", question, _Rewrites(claim), parser).faults


def test_a_faithful_rewrite_has_no_faults(parser):
    assert (
        check(
            "q1",
            "Did the doctor prescribe erythromycin?",
            ClaimRewriter(parser),
            parser,
        ).faults
        == ()
    )


def test_a_claim_that_is_still_a_question_is_a_fault(parser):
    assert "still a question" in faults(
        "Did the doctor prescribe erythromycin?",
        "Did the doctor prescribe erythromycin?",
        parser,
    )


def test_a_claim_with_the_auxiliary_still_in_front_is_a_fault(parser):
    assert "auxiliary still fronted" in faults(
        "Did the doctor prescribe erythromycin?",
        "did the doctor prescribe erythromycin",
        parser,
    )


def test_a_claim_that_lost_a_word_of_the_question_is_a_fault(parser):
    """A Hard Negative turns on one number, so a lost word is a lost answer."""
    assert "lost 200" in faults(
        "Was the prescribed dose 200 mg daily?",
        "the prescribed dose was mg daily",
        parser,
    )


def test_the_tag_a_tag_question_drops_is_not_counted_as_lost(parser):
    assert (
        faults(
            "The lipid profile came back normal, didn't it?",
            "The lipid profile came back normal",
            parser,
        )
        == ()
    )


def test_the_scoping_phrase_the_rules_drop_is_not_counted_as_lost(parser):
    assert (
        faults(
            "At any point does the patient talk about their pet?",
            "the patient does talk about their pet",
            parser,
        )
        == ()
    )


def test_a_claim_that_does_not_parse_as_a_clause_is_a_fault(parser):
    assert "no subject" in faults(
        "Does the patient complain of dryness?",
        "of dryness on the hands",
        parser,
    )


def test_an_auxiliary_moved_into_the_middle_of_a_word_is_a_fault(parser):
    """ "mmol/mol" is one word however the tokenizer cuts it."""
    assert "auxiliary moved inside a word" in faults(
        "Was the HbA1c 42 mmol/mol?", "the HbA1c 42 mmol/ was mol", parser
    )


def test_a_predicate_beginning_with_a_coordinator_is_a_fault(parser):
    assert "predicate begins with a coordinator" in faults(
        "Are redness and swelling absent?", "redness are and swelling absent", parser
    )


def test_a_mis_split_the_claim_still_reads_around_is_not_caught(parser):
    """Recorded, not aspirational: word order is preserved by construction, so
    re-parsing the Claim reports whatever split it was handed. ADR-0002 carries
    the residual this cannot see."""
    assert (
        faults("Is the patient feeling well?", "the patient feeling is well", parser)
        == ()
    )
