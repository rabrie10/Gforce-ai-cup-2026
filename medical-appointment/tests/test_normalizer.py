"""A Question and the speech that answers it must reduce to the same tokens.

The cases are the forms the error analysis found in train and dev (taxonomy P2
and P6), plus the alignment the Chunker depends on: a token merged from several
Words still has to name the Words it covers, because that is what an Evidence
Span is returned as.
"""

import pytest

from medapp.normalizer import NormalizedToken, normalize_text, normalize_words
from medapp.types import Word


def _spoken(text: str, start: float = 0.0, step: float = 0.5) -> tuple[Word, ...]:
    """The words of ``text`` as the Transcriber would hand them over."""
    return tuple(
        Word(
            text=word,
            start=start + index * step,
            end=start + (index + 1) * step,
            probability=0.9,
        )
        for index, word in enumerate(text.split())
    )


def _texts(tokens: tuple[NormalizedToken, ...]) -> tuple[str, ...]:
    return tuple(token.text for token in tokens)


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("100 mg", "one hundred milligrams"),
        ("2 weeks", "two weeks"),
        ("155/98", "155 over 98"),
        ("0.3", "point three"),
        ("0.3", "zero point three"),
        ("41 mmol/mol", "41 millimoles per mole"),
        ("7.2 mmol/L", "7.2 millimoles per liter"),
        ("HbA1c", "hemoglobin A1c"),
        ("BMI", "body mass index"),
        ("ECG", "electrocardiogram"),
        ("COVID-19", "coronavirus"),
        ("135/88", "one hundred thirty five over eighty eight"),
        ("20 mg", "twenty milligrams"),
        ("500 mcg", "five hundred micrograms"),
        ("120/80 mmHg", "120 over 80 millimeters of mercury"),
        ("62 bpm", "62 beats per minute"),
        ("2.5 mg", "two and a half milligrams"),
    ],
)
def test_written_and_spoken_forms_reduce_to_the_same_tokens(
    written: str, spoken: str
) -> None:
    assert normalize_text(written) == _texts(normalize_words(_spoken(spoken)))


def test_a_question_and_the_span_that_answers_it_agree() -> None:
    question = "Was the prescribed dose 100 mg daily for 2 weeks?"
    span = _spoken("one hundred milligrams daily for two weeks")

    assert normalize_text(question)[-6:] == _texts(normalize_words(span))


def test_casing_and_punctuation_are_stripped_on_both_sides() -> None:
    assert normalize_text("Your blood pressure is 135/88.") == normalize_text(
        "your blood pressure is 135/88"
    )
    assert _texts(normalize_words(_spoken("Overall, your diabetes looks stable."))) == (
        "overall",
        "your",
        "diabetes",
        "looks",
        "stable",
    )


def test_a_hard_negative_value_swap_stays_distinguishable() -> None:
    hard_negative = normalize_text("Was the prescribed dose 200 mg daily?")
    spoken = _texts(normalize_words(_spoken("100 milligrams daily")))

    assert "200" in hard_negative
    assert "100" in spoken
    assert not set(hard_negative) & {"100"}


def test_a_merged_token_carries_every_word_it_was_read_from() -> None:
    words = _spoken("one hundred milligrams")

    tokens = normalize_words(words)

    assert _texts(tokens) == ("100", "mg")
    assert tokens[0].words == words[:2]
    assert tokens[0].span == (0.0, 1.0)
    assert tokens[1].words == (words[2],)
    assert tokens[1].span == (1.0, 1.5)


def test_a_reading_carries_the_words_of_both_numbers_and_the_word_between() -> None:
    words = _spoken("155 over 98")

    (token,) = normalize_words(words)

    assert token.text == "155/98"
    assert token.words == words
    assert token.span == (0.0, 1.5)


def test_every_token_maps_back_to_words_in_time_order() -> None:
    words = _spoken("your blood pressure is one thirty five over eighty eight today")

    tokens = normalize_words(words)

    covered = [word for token in tokens for word in token.words]
    assert covered == list(words)
    assert all(token.words for token in tokens)


def test_a_written_token_has_no_words_and_therefore_no_span() -> None:
    token = NormalizedToken(text="100", words=())

    assert token.span is None


def test_a_split_contraction_keeps_the_word_it_was_spoken_in() -> None:
    words = _spoken("we don't")

    tokens = normalize_words(words)

    assert _texts(tokens) == ("we", "do", "not")
    assert tokens[1].words == (words[1],)
    assert tokens[2].words == (words[1],)


def test_a_filler_leaves_no_token_behind() -> None:
    assert _texts(normalize_words(_spoken("um your pulse is 62"))) == (
        "your",
        "pulse",
        "is",
        "62",
    )


def test_normalization_is_deterministic() -> None:
    spoken = _spoken(
        "one hundred and fifty five over ninety eight millimeters of mercury"
    )

    assert normalize_words(spoken) == normalize_words(spoken)


def test_empty_input_normalizes_to_nothing() -> None:
    assert normalize_text("") == ()
    assert normalize_words(()) == ()
