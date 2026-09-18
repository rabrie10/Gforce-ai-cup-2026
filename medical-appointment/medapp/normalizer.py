"""One canonical written form for Questions and for transcribed speech.

A Question writes every clinical value in symbol form and the Conversation
speaks it in full: "41 mmol/mol" against *"41 millimoles per mole"*, "135/88"
against *"135 over 88"*, "100 mg" against *"100 milligrams"*. The error analysis
found 13 of 122 Positives unreachable for that reason alone, and 33 of 85 Hard
Negatives turning on the one number inside an otherwise identical sentence. Both
sides are therefore normalized by the same function, so that the same claim
spoken and written reduces to the same tokens.

Normalization is token-aligned rather than string-to-string: a transcript token
carries the Words it came from, including the several Words a merged token
covers, because an Evidence Span is returned as Word timings and a token that
has lost its Words cannot back one.

Whisper's English normalizer does the general work — casing, punctuation,
contractions, spelled-out numbers, fractions and decimals — and is vendored in
``medapp.vendor.whisper_english``. What this module adds is the clinical
vocabulary the error analysis listed, and the alignment the vendored normalizer
cannot provide because it works on whole strings.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from medapp.types import Word
from medapp.vendor.whisper_english import (
    EnglishTextNormalizer,
    remove_symbols_and_diacritics,
)
from utils import Span

_WHISPER = EnglishTextNormalizer()
_NUMBERS = _WHISPER.standardize_numbers
_SPELLINGS = _WHISPER.standardize_spellings

# "/" is kept where Whisper would drop it: it is the canonical joiner for a
# blood-pressure reading and for the compound units, so "mmol/mol" and "135/88"
# must survive as single tokens rather than becoming two.
_KEEP_SYMBOLS = ".%$¢€£/"

# Whisper's fillers, as whole tokens.
_FILLER = re.compile(_WHISPER.ignore_patterns)

_NUMERIC = re.compile(r"[-+$€£¢]?\d+(\.\d+)?%?")
_LEADING_POINT = re.compile(r"\.\d+")

# Words that belong to a number only in the company of one: Whisper reads "two
# and a half" as 2.5 and "point three" as .3, so a run has to be able to reach
# across them, but a stretch holding nothing else is not a number.
_BRIDGE_WORDS = frozenset(
    {"a", "and", "point", "per", "half", "halves", "quarter", "quarters"}
)

# The spoken and written forms of one term, mapped to the form both become.
# Applied before numbers are read, so that the digit inside "hba1c" is never
# offered to the number normalizer, which would spell it back out as "hba one c".
_TERMS: dict[tuple[str, ...], str] = {
    ("milligram",): "mg",
    ("milligrams",): "mg",
    ("mgs",): "mg",
    ("microgram",): "mcg",
    ("micrograms",): "mcg",
    ("mcgs",): "mcg",
    ("gram",): "g",
    ("grams",): "g",
    ("kilo",): "kg",
    ("kilos",): "kg",
    ("kilogram",): "kg",
    ("kilograms",): "kg",
    ("milliliter",): "ml",
    ("milliliters",): "ml",
    ("mls",): "ml",
    ("liter",): "l",
    ("liters",): "l",
    ("millimole",): "mmol",
    ("millimoles",): "mmol",
    ("mole",): "mol",
    ("moles",): "mol",
    ("minute",): "min",
    ("minutes",): "min",
    ("millimeters", "of", "mercury"): "mmhg",
    ("millimeter", "of", "mercury"): "mmhg",
    ("mm", "hg"): "mmhg",
    ("beats", "per", "min"): "bpm",
    ("hemoglobin", "a1c"): "hba1c",
    ("hb", "a1c"): "hba1c",
    ("a1c",): "hba1c",
    ("body", "mass", "index"): "bmi",
    ("electrocardiogram",): "ecg",
    ("ekg",): "ecg",
}

# The same, for forms that only exist once numbers have been read: "155 over 98"
# is a reading and "covid nineteen" reaches the digit through the number stage.
_READINGS: dict[tuple[str, ...], str] = {
    ("coronavirus",): "covid19",
    ("covid", "19"): "covid19",
}

# What a rate is measured in. "mg per day" written is "mg/day", and "mmol per
# mole" is "mmol/mol", so the two are joined by rule rather than one compound at
# a time — and a "/" between anything else is not a unit and is split back
# apart, because "and/or" is spoken as two words.
_MEASURES = frozenset(
    {"mg", "mcg", "g", "kg", "ml", "l", "mmol", "mol", "unit", "units"}
)
_PER_MEASURES = _MEASURES | {"day", "week", "min", "hour", "m2"}

_LONGEST_TERM = max(len(phrase) for phrase in (*_TERMS, *_READINGS))


@dataclass(frozen=True, slots=True)
class NormalizedToken:
    """One canonical token and the Words it was read from.

    Attributes:
        words: The Words the token covers, in time order — several where the
            token merges them, as "135 over 98" merges three into "135/98", and
            empty for a token normalized from written text, which has no
            timings.
    """

    text: str
    words: tuple[Word, ...]

    @property
    def span(self) -> Span | None:
        """The stretch of the Conversation the token was spoken in.

        Returns:
            The first Word's start and the last Word's end, or None for a token
            normalized from written text.
        """
        if not self.words:
            return None

        return (self.words[0].start, self.words[-1].end)


def normalize_text(text: str) -> tuple[str, ...]:
    """Normalize written text — a Question — into canonical tokens.

    Args:
        text: The text as written.

    Returns:
        The canonical tokens, which compare equal to those of a transcript span
        that says the same thing.
    """
    return tuple(token.text for token in _normalize(text.split(), words=None))


def normalize_words(words: Sequence[Word]) -> tuple[NormalizedToken, ...]:
    """Normalize transcribed Words into canonical tokens that keep their timings.

    Args:
        words: The Words in time order, as the Transcriber aligned them.

    Returns:
        The canonical tokens, each carrying the Words it was read from.
    """
    return _normalize([word.text for word in words], words=words)


def _normalize(
    texts: Sequence[str], words: Sequence[Word] | None
) -> tuple[NormalizedToken, ...]:
    """Run the pipeline over one token sequence, with or without Words."""
    tokens = _clean(texts, words)
    tokens = _rewrite(tokens, _TERMS)
    tokens = _drop_fillers(tokens)
    tokens = _join_measures(tokens)
    tokens = _read_numbers(tokens)
    tokens = _rewrite(tokens, _READINGS)
    tokens = _read_slashed(tokens)
    tokens = _split_compounds(tokens)

    return tuple(tokens)


def _clean(texts: Sequence[str], words: Sequence[Word] | None) -> list[NormalizedToken]:
    """Lower-case, expand contractions and strip punctuation, token by token.

    This is Whisper's text normalizer with its number and string-wide stages
    held back: everything here is a rewrite of one token into zero or more
    tokens, which is what lets every output token keep the Word it came from.
    A token may vanish — punctuation on its own — and a token may split, as
    "don't" splits into "do" and "not". Whisper's fillers are dropped a stage
    later: "mm" is one of them, and "mm Hg" is a unit.
    """
    cleaned: list[NormalizedToken] = []

    for index, text in enumerate(texts):
        source = () if words is None else (words[index],)

        piece = text.lower()
        piece = re.sub(r"[<\[][^>\]]*[>\]]", "", piece)
        piece = re.sub(r"\(([^)]+?)\)", "", piece)

        for pattern, replacement in _WHISPER.replacers.items():
            piece = re.sub(pattern, replacement, piece)

        piece = re.sub(r"(\d),(\d)", r"\1\2", piece)
        piece = re.sub(r"\.([^0-9]|$)", r" \1", piece)
        piece = remove_symbols_and_diacritics(piece, keep=_KEEP_SYMBOLS)
        piece = _SPELLINGS(piece)

        cleaned.extend(
            NormalizedToken(text=part, words=source) for part in piece.split()
        )

    return cleaned


def _drop_fillers(tokens: Sequence[NormalizedToken]) -> list[NormalizedToken]:
    """Drop Whisper's fillers, after the terms that are spelled like one.

    "mm" is both a filler and the first half of "mm Hg", so the fillers go once
    the units have been read rather than during cleaning.
    """
    return [token for token in tokens if not _FILLER.fullmatch(token.text)]


def _rewrite(
    tokens: Sequence[NormalizedToken], rules: dict[tuple[str, ...], str]
) -> list[NormalizedToken]:
    """Replace each phrase in ``rules`` with the one form it maps to.

    The longest phrase wins, so that "millimoles per mole" is read as a unit
    rather than as "millimoles" followed by anything. A merged token carries
    every Word of the phrase it replaced.

    Rewriting runs to a fixed point, because one rule's output is another's
    input: "mmol liter" only becomes "mmol/l" once "liter" has become "l".
    """
    rewritten = list(tokens)

    for _ in range(_LONGEST_TERM):
        rewritten, changed = _rewrite_once(rewritten, rules)

        if not changed:
            break

    return rewritten


def _rewrite_once(
    tokens: Sequence[NormalizedToken], rules: dict[tuple[str, ...], str]
) -> tuple[list[NormalizedToken], bool]:
    """One left-to-right pass of :func:`_rewrite`, and whether it changed anything."""
    rewritten: list[NormalizedToken] = []
    changed = False
    index = 0

    while index < len(tokens):
        for length in range(min(_LONGEST_TERM, len(tokens) - index), 0, -1):
            phrase = tuple(token.text for token in tokens[index : index + length])
            canonical = rules.get(phrase)

            if canonical is not None:
                rewritten.append(
                    NormalizedToken(
                        text=canonical,
                        words=_merged(tokens[index : index + length]),
                    )
                )
                changed = changed or phrase != (canonical,)
                index += length
                break
        else:
            rewritten.append(tokens[index])
            index += 1

    return rewritten, changed


def _join_measures(tokens: Sequence[NormalizedToken]) -> list[NormalizedToken]:
    """Join "mg per day", and the "mmol litre" a transcript elides, into "mg/day"."""
    joined: list[NormalizedToken] = []
    index = 0

    while index < len(tokens):
        rate = tokens[index : index + 3]

        pair = tokens[index : index + 2]

        if len(pair) == 2 and pair[0].text in _MEASURES and pair[1].text in _MEASURES:
            joined.append(
                NormalizedToken(
                    text=f"{pair[0].text}/{pair[1].text}", words=_merged(pair)
                )
            )
            index += 2
            continue

        if (
            len(rate) == 3
            and rate[1].text == "per"
            and rate[0].text in _MEASURES
            and rate[2].text in _PER_MEASURES
        ):
            joined.append(
                NormalizedToken(
                    text=f"{rate[0].text}/{rate[2].text}", words=_merged(rate)
                )
            )
            index += 3
            continue

        joined.append(tokens[index])
        index += 1

    return joined


def _split_compounds(tokens: Sequence[NormalizedToken]) -> list[NormalizedToken]:
    """Split a "/" that joins neither a reading nor a rate.

    "135/88" and "mmol/mol" are written forms of what is spoken as a whole;
    "and/or" is written for what is spoken as two words, and a Question that
    writes one has to reach a transcript that speaks the other.
    """
    split: list[NormalizedToken] = []

    for token in tokens:
        parts = token.text.split("/")

        if len(parts) == 2 and (
            all(part.isdigit() for part in parts)
            or (parts[0] in _MEASURES and parts[1] in _PER_MEASURES)
        ):
            split.append(token)
            continue

        split.extend(
            NormalizedToken(text=part, words=token.words) for part in parts if part
        )

    return split


def _read_numbers(tokens: Sequence[NormalizedToken]) -> list[NormalizedToken]:
    """Turn spoken numbers into digits, over each run of number words.

    The vendored normalizer reads a whole string, so it is given exactly the
    runs of tokens that can carry a number and nothing else. Anything outside a
    run is left alone — which also keeps terms such as "hba1c" away from a stage
    that would split the digit off.
    """
    read: list[NormalizedToken] = []
    index = 0

    while index < len(tokens):
        run = _run_length(tokens, index)

        if run == 0:
            read.append(tokens[index])
            index += 1
            continue

        read.extend(_without_leading_bridge(_read_run(tokens[index : index + run])))
        index += run

    return read


def _without_leading_bridge(
    run: Sequence[NormalizedToken],
) -> list[NormalizedToken]:
    """Drop bridge words the number stage read as themselves.

    "a hundred milligrams" is "100 mg": Whisper reads "a hundred" as 100 and
    leaves the "a" standing, and a leading bridge word carries nothing a
    Question would write.
    """
    start = 0

    while start < len(run) - 1 and run[start].text in _BRIDGE_WORDS:
        start += 1

    return list(run[start:])


def _run_length(tokens: Sequence[NormalizedToken], start: int) -> int:
    """Length of the run of number words at ``start``, or 0 if there is none.

    A run reaches across bridge words, because they are what "point three" and
    "two and a half" are held together by, but a stretch of bridge words with no
    number among them is not a run: a bare "and" is not a number.
    """
    end = start

    while end < len(tokens) and (
        _is_number_word(tokens[end].text) or tokens[end].text in _BRIDGE_WORDS
    ):
        end += 1

    run = tokens[start:end]

    if not any(_is_number_word(token.text) for token in run):
        return 0

    return len(run)


def _is_number_word(text: str) -> bool:
    """Whether a token can carry a number on its own."""
    if text in _BRIDGE_WORDS:
        return False

    return text in _NUMBERS.words or _NUMERIC.fullmatch(text) is not None


def _read_run(run: Sequence[NormalizedToken]) -> list[NormalizedToken]:
    """Normalize one run and give each token back the Words it was read from.

    The vendored normalizer returns a string, so the run is normalized again for
    each of its prefixes: a token is settled by the shortest prefix that already
    produces it, and the Words of that prefix are the Words it was read from.
    "one hundred" settles only at the second Word and so carries both. A run
    whose prefixes never settle a token — none has been seen, but the normalizer
    is free to revise what it has read — hands the rest of its Words to the
    token that closes it, which widens a span rather than losing one.
    """
    final = _NUMBERS(" ".join(token.text for token in run)).split()
    prefixes = [
        _NUMBERS(" ".join(token.text for token in run[: length + 1])).split()
        for length in range(len(run))
    ]

    read: list[NormalizedToken] = []
    consumed = 0

    for position, text in enumerate(final):
        if position == len(final) - 1:
            settled = len(run)
        else:
            settled = next(
                (
                    length + 1
                    for length, prefix in enumerate(prefixes)
                    if length + 1 > consumed
                    and prefix[: position + 1] == final[: position + 1]
                ),
                len(run),
            )

        read.append(NormalizedToken(text=text, words=_merged(run[consumed:settled])))
        consumed = settled

    return read


def _read_slashed(tokens: Sequence[NormalizedToken]) -> list[NormalizedToken]:
    """Join a spoken reading into its written form and restore leading zeroes.

    "155 over 98" is the spoken form of "155/98"; ".3", which is what Whisper
    reads "point three" as, is the written "0.3".
    """
    joined: list[NormalizedToken] = []
    index = 0

    while index < len(tokens):
        reading = tokens[index : index + 3]

        if (
            len(reading) == 3
            and reading[1].text == "over"
            and all(part.text.isdigit() for part in (reading[0], reading[2]))
        ):
            joined.append(
                NormalizedToken(
                    text=f"{reading[0].text}/{reading[2].text}",
                    words=_merged(reading),
                )
            )
            index += 3
            continue

        token = tokens[index]

        if _LEADING_POINT.fullmatch(token.text):
            token = NormalizedToken(text=f"0{token.text}", words=token.words)

        joined.append(token)
        index += 1

    return joined


def _merged(tokens: Iterable[NormalizedToken]) -> tuple[Word, ...]:
    """Every Word of the tokens a rewrite replaced, in order."""
    return tuple(word for token in tokens for word in token.words)
