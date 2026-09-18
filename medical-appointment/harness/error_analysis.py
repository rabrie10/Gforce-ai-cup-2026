"""Render every train and dev Question beside the transcript it is answered
from, so the failure taxonomy is written from the data rather than guessed.

Nothing here scores anything. It lays a Question next to the words the
annotation points at — and, for a Question with no annotation, next to the
passage of the Conversation it comes closest to — so that a person can read all
of them and record what the Chunker, the normalizer and the judges will have to
handle.

Read on train and dev only. ADR-0002 transcribes the test fold but does not
read it, and :func:`readable_fold` is where that is enforced for this module.
"""

import re
from dataclasses import dataclass
from functools import cache

from harness import transcript_cache
from harness.folds import FoldName, questions_in_fold
from medapp.types import Segment, Word
from utils import Span, gold_evidence

READABLE_FOLDS: tuple[FoldName, ...] = ("train", "dev")

# How many words either side of an annotated span are shown. Enough to see
# whether the evidence really stops where the annotation does.
NEIGHBOURING_WORDS = 12

# The window a Question with no annotation is matched against, in words. Close
# to the length of the longest annotated Evidence Span, so a Hard Negative's
# nearest passage is comparable to a Positive's.
NEGATIVE_WINDOW_WORDS = 40

# Words carried by every clinical Question, which say nothing about which
# passage it is about.
# fmt: off
_STOP_WORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "been", "did", "do", "does",
    "for", "from", "had", "has", "have", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "there", "they", "this", "to", "was", "were", "what",
    "when", "which", "who", "will", "with", "patient", "doctor",
])
# fmt: on


@dataclass(frozen=True, slots=True)
class TimedWords:
    """A Conversation's Words in order, with the Segment each one came from.

    Attributes:
        words: Every Word of the Conversation, in time order.
        segment_of: The index into the Conversation's Segments that each Word
            was transcribed in, parallel to ``words``.
    """

    words: tuple[Word, ...]
    segment_of: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.words)

    def text(self, start: int, end: int) -> str:
        """The words from ``start`` up to ``end``, as one string."""
        return "".join(word.text for word in self.words[start:end]).strip()


@dataclass(frozen=True, slots=True)
class PositiveRendering:
    """One annotated Evidence Span laid out against the transcript.

    Attributes:
        span: The annotated span, in seconds.
        span_text: The transcribed words the span covers.
        before: The words immediately preceding the span.
        after: The words immediately following it.
        segments_crossed: How many Transcriber Segments the span's words came
            from. More than one means the evidence crosses a Segment boundary,
            which is the case a Chunker built on Segments cannot represent.
        words_covered: How many Words the span covers. Zero means the
            annotation falls in a stretch the Transcriber produced no words
            for.
    """

    question_id: str
    transcript_id: str
    question: str
    span: Span
    span_text: str
    before: str
    after: str
    segments_crossed: int
    words_covered: int


@dataclass(frozen=True, slots=True)
class NegativeRendering:
    """One Question with no annotation, beside the passage it comes closest to.

    A Hard Negative is classified by reading the passage its wording points at:
    the Conversation either says something different there (a value swap),
    denies it, or never establishes it at all. The nearest passage is found by
    content-word overlap alone — deliberately crude, because a retriever does
    not exist yet and this is the material the retriever will be designed from.

    Attributes:
        overlap: How many of the Question's content words the passage contains.
        nearest_text: The transcribed passage with the highest overlap.
        nearest_span: Where that passage sits, in seconds.
    """

    question_id: str
    transcript_id: str
    question: str
    question_type: str
    overlap: int
    nearest_text: str
    nearest_span: Span


def readable_fold(fold: str) -> FoldName:
    """The fold name, refused if it is the held-out one.

    Raises:
        ValueError: If ``fold`` is the test fold, or names no fold at all. The
            taxonomy decides how the Chunker and the judges are built, so
            reading the test fold here would leak the held-out data into every
            design decision that follows.
    """
    if fold not in READABLE_FOLDS:
        raise ValueError(
            f"{fold!r} cannot be read: error analysis runs on "
            f"{' and '.join(READABLE_FOLDS)} only. The test fold is "
            "transcribed but not opened until tuning ends."
        )

    return fold  # type: ignore[return-value]


def timed_words(segments: tuple[Segment, ...]) -> TimedWords:
    """Flatten a Conversation's Segments into Words that remember their Segment."""
    words: list[Word] = []
    segment_of: list[int] = []

    for index, segment in enumerate(segments):
        for word in segment.words:
            words.append(word)
            segment_of.append(index)

    return TimedWords(words=tuple(words), segment_of=tuple(segment_of))


def words_in_span(transcript: TimedWords, span: Span) -> tuple[int, int]:
    """The half-open range of Word indices a span covers.

    A Word counts as covered when it overlaps the span at all, so a span whose
    boundaries fall mid-word still shows the words it cuts through rather than
    dropping them.

    A span over silence — the Transcriber drops stretches with no speech in
    them — covers no Words and returns an empty range at the position the span
    falls, so the words either side of it still bracket the right place.
    """
    start, end = span

    covered = [
        index
        for index, word in enumerate(transcript.words)
        if word.end > start and word.start < end
    ]

    if not covered:
        position = sum(1 for word in transcript.words if word.end <= start)
        return (position, position)

    return (covered[0], covered[-1] + 1)


def render_positive(
    row: dict[str, str],
    segments: tuple[Segment, ...],
    neighbouring_words: int = NEIGHBOURING_WORDS,
) -> PositiveRendering:
    """Lay one annotated Question out against its transcript.

    Raises:
        ValueError: If the row carries no annotated Evidence Span.
    """
    span = gold_evidence(row)
    if span is None:
        raise ValueError(
            f"{row['question_id']} carries no annotated Evidence Span, so "
            "there is nothing to render it against."
        )

    transcript = timed_words(segments)
    first, last = words_in_span(transcript, span)

    return PositiveRendering(
        question_id=row["question_id"],
        transcript_id=row["transcript_id"],
        question=row["question"],
        span=span,
        span_text=transcript.text(first, last),
        before=transcript.text(max(0, first - neighbouring_words), first),
        after=transcript.text(last, last + neighbouring_words),
        segments_crossed=len(set(transcript.segment_of[first:last])),
        words_covered=last - first,
    )


def content_words(text: str) -> frozenset[str]:
    """The words of a Question that say what it is about.

    Lower-cased, stripped of punctuation and of the words every Question
    carries. Single characters go too: splitting on punctuation leaves the
    clitic of ``patient's`` and ``don't`` behind, and an ``s`` shared between a
    Question and a passage says nothing about either. Crude on purpose: this is
    the material the normalizer and the retriever are designed from, not a
    component of either.
    """
    return frozenset(
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 1 and word not in _STOP_WORDS
    )


def nearest_passage(
    question: str,
    transcript: TimedWords,
    window_words: int = NEGATIVE_WINDOW_WORDS,
) -> tuple[int, int, int]:
    """The window of the Conversation sharing the most content words.

    Returns:
        The window's first and last-plus-one Word index, and the number of the
        Question's content words it contains.
    """
    wanted = content_words(question)

    if not wanted or not len(transcript):
        return (0, min(window_words, len(transcript)), 0)

    spoken = [content_words(word.text) for word in transcript.words]
    best = (0, min(window_words, len(transcript)), -1)

    for start in range(max(1, len(transcript) - window_words + 1)):
        end = min(start + window_words, len(transcript))
        found = len({word for window in spoken[start:end] for word in window} & wanted)

        if found > best[2]:
            best = (start, end, found)

    return best


def render_negative(
    row: dict[str, str],
    segments: tuple[Segment, ...],
    window_words: int = NEGATIVE_WINDOW_WORDS,
) -> NegativeRendering:
    """Lay one unannotated Question beside the passage it comes closest to."""
    transcript = timed_words(segments)
    first, last, overlap = nearest_passage(row["question"], transcript, window_words)

    covered = transcript.words[first:last]
    span: Span = (covered[0].start, covered[-1].end) if covered else (0.0, 0.0)

    return NegativeRendering(
        question_id=row["question_id"],
        transcript_id=row["transcript_id"],
        question=row["question"],
        question_type=row["question_type"],
        overlap=overlap,
        nearest_text=transcript.text(first, last),
        nearest_span=span,
    )


def positives(fold: FoldName) -> list[PositiveRendering]:
    """Every annotated Question of one readable fold, rendered.

    Raises:
        ValueError: If ``fold`` is the held-out test fold.
        FileNotFoundError: If a Conversation of the fold is not cached.
    """
    return [
        render_positive(row, _segments(row["transcript_id"]))
        for row in questions_in_fold(readable_fold(fold))
        if gold_evidence(row) is not None
    ]


def negatives(fold: FoldName) -> list[NegativeRendering]:
    """Every unannotated Question of one readable fold, rendered.

    Raises:
        ValueError: If ``fold`` is the held-out test fold.
        FileNotFoundError: If a Conversation of the fold is not cached.
    """
    return [
        render_negative(row, _segments(row["transcript_id"]))
        for row in questions_in_fold(readable_fold(fold))
        if gold_evidence(row) is None
    ]


@cache
def _segments(transcript_id: str) -> tuple[Segment, ...]:
    """One Conversation's cached Segments, read once per Conversation."""
    return transcript_cache.read(transcript_id)
