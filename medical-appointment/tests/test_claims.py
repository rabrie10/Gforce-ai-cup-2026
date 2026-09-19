"""The Claim rewriter, over every Question shape train and dev hold.

The Questions below are taken verbatim from the supplied data, one per shape
the two folds contain: auxiliary-fronted questions with each auxiliary that
appears, do-support, existentials, tag questions in each of their spellings,
and the two kinds of phrase that front an auxiliary-fronted question — one the
Claim keeps, one it drops.

The parser is real. It is what locates the subject, so stubbing it would leave
nothing under test; the model is a dependency rather than a download, so it is
there wherever the tests run. The fold-wide fidelity is measured by
``python -m scripts.claim_fidelity``, not asserted here.
"""

import pytest

from medapp.claims import ClaimRewriter, load_parser


@pytest.fixture(scope="module")
def rewriter() -> ClaimRewriter:
    return ClaimRewriter(load_parser())


@pytest.mark.parametrize(
    ("question", "claim"),
    [
        # Copula, every number and tense that appears.
        (
            "Is the heart examination without abnormal findings?",
            "the heart examination is without abnormal findings",
        ),
        ("Are the findings unremarkable?", "the findings are unremarkable"),
        ("Was the TSH normal?", "the TSH was normal"),
        (
            "Were the tonsils covered in coatings?",
            "the tonsils were covered in coatings",
        ),
        # A subject that carries a prepositional phrase, and one that carries a
        # coordinate: both belong in front of the auxiliary.
        (
            "Are four daily doses of one million IU of penicillin planned?",
            "four daily doses of one million IU of penicillin are planned",
        ),
        (
            "Were the lungs and heart normal on auscultation?",
            "the lungs and heart were normal on auscultation",
        ),
        # A coordinate the parser read as something other than a noun still
        # belongs to the subject: no predicate begins with "and".
        ("Are redness and swelling absent?", "redness and swelling are absent"),
        # A measurement is the predicate of the copula that introduces it, and
        # "mmol/mol" is one word however the tokenizer cuts it.
        ("Was the HbA1c 42 mmol/mol?", "the HbA1c was 42 mmol/mol"),
        # An adverb between the subject and the verb belongs with the
        # predicate, which is where the fronted auxiliary moves in front of it.
        (
            "Is the patient still being treated for asthma?",
            "the patient is still being treated for asthma",
        ),
        ("Was the vaccine actually given?", "the vaccine was actually given"),
        # An adjective behind an indefinite pronoun is part of the subject.
        (
            "Was anything abnormal picked up at the examination?",
            "anything abnormal was picked up at the examination",
        ),
        # Existentials, affirmative and negative.
        (
            "Is there any mention of attending a concert?",
            "there is any mention of attending a concert",
        ),
        (
            "Are there no signs of complications?",
            "there are no signs of complications",
        ),
        # Do-support: the auxiliary moves rather than the verb being
        # re-conjugated, which is emphatic "do" and the same assertion.
        (
            "Did the doctor prescribe erythromycin?",
            "the doctor did prescribe erythromycin",
        ),
        (
            "Does the plan include routine follow-up?",
            "the plan does include routine follow-up",
        ),
        ("Do the elements resemble molluscs?", "the elements do resemble molluscs"),
        # The subject boundary the same word sits on either side of.
        (
            "Did the doctor plan continued follow-up?",
            "the doctor did plan continued follow-up",
        ),
        # Perfect, modal and progressive auxiliaries.
        (
            "Has the patient reported tiredness and thirst?",
            "the patient has reported tiredness and thirst",
        ),
        ("Will the treatment last two weeks?", "the treatment will last two weeks"),
        (
            "Should the daily dose be 100 mg?",
            "the daily dose should be 100 mg",
        ),
        (
            "Can the patient leave without any additional treatment?",
            "the patient can leave without any additional treatment",
        ),
        (
            "Are the speakers discussing the weather?",
            "the speakers are discussing the weather",
        ),
        # Stacked auxiliaries belong with the predicate, not the subject.
        (
            "Is the check-up being done because the patient has symptoms "
            "from the heart?",
            "the check-up is being done because the patient has symptoms "
            "from the heart",
        ),
        # An adverbial fronted before the auxiliary is part of what the
        # Question asserts, and stays.
        (
            "Alongside asthma, is the patient treated for diabetes?",
            "Alongside asthma, the patient is treated for diabetes",
        ),
    ],
)
def test_an_auxiliary_fronted_question_becomes_the_claim_it_asserts(
    rewriter, question, claim
):
    rewritten = rewriter.rewrite(question)

    assert rewritten.text == claim
    assert rewritten.shape == "auxiliary_fronted"


@pytest.mark.parametrize(
    ("question", "claim"),
    [
        (
            "The HbA1c came out at 47 mmol/mol, right?",
            "The HbA1c came out at 47 mmol/mol",
        ),
        (
            "The visit was a preventive health check, correct?",
            "The visit was a preventive health check",
        ),
        (
            "A moisturizing cream was prescribed, wasn't it?",
            "A moisturizing cream was prescribed",
        ),
        (
            "The lipid profile came back normal, didn't it?",
            "The lipid profile came back normal",
        ),
    ],
)
def test_a_tag_question_is_the_declarative_in_front_of_the_tag(
    rewriter, question, claim
):
    rewritten = rewriter.rewrite(question)

    assert rewritten.text == claim
    assert rewritten.shape == "tag_question"


@pytest.mark.parametrize(
    ("question", "claim"),
    [
        (
            "At any point does the patient talk about their pet?",
            "the patient does talk about their pet",
        ),
        (
            "According to the conversation, was a specific type of sport "
            "played by the patient discussed?",
            "a specific type of sport played by the patient was discussed",
        ),
    ],
)
def test_a_phrase_that_scopes_the_question_over_the_recording_is_dropped(
    rewriter, question, claim
):
    """It scopes the asking, not the claim, and leaves the Claim ungrammatical."""
    assert rewriter.rewrite(question).text == claim


def test_a_question_already_in_the_asserting_form_is_left_as_one(rewriter):
    rewritten = rewriter.rewrite("The patient understands the treatment well.")

    assert rewritten.text == "The patient understands the treatment well."
    assert rewritten.shape == "declarative"


def test_the_value_a_hard_negative_turns_on_survives_the_rewrite(rewriter):
    """A Hard Negative differs from the truth by one number, so losing one
    loses the answer."""
    assert (
        rewriter.rewrite("Was the prescribed dose 200 mg daily?").text
        == "the prescribed dose was 200 mg daily"
    )


def test_negation_survives_the_rewrite(rewriter):
    """ "No new symptoms" and "new symptoms" are opposite answers."""
    assert (
        rewriter.rewrite("Did the patient report no new symptoms?").text
        == "the patient did report no new symptoms"
    )


def test_a_misparsed_do_support_question_drops_the_auxiliary(rewriter):
    """Fronted "do" is always an auxiliary, so a parse that made it the verb
    cannot be trusted to say where the subject ends."""
    assert (
        rewriter.rewrite("Does the patient complain of dryness on the hands?").text
        == "the patient complain of dryness on the hands"
    )


@pytest.mark.parametrize(
    ("question", "claim"),
    [
        (
            "Is the patient complaining of chest symptoms?",
            "the patient complaining is of chest symptoms",
        ),
        ("Is the patient feeling well?", "the patient feeling is well"),
    ],
)
def test_a_subject_the_parser_ran_past_is_recorded_rather_than_guessed_at(
    rewriter, question, claim
):
    """The parser reads these gerunds as nouns and hands back a subject that
    has swallowed the predicate. Nothing in the parse distinguishes them from
    "Is the heart examination without abnormal findings?", where the same
    shape is right, so the rewrite follows the parse and ADR-0002 records the
    residual. This test pins what it currently produces, so a parser or rule
    change that fixes it is noticed rather than absorbed."""
    assert rewriter.rewrite(question).text == claim
