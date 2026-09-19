"""How faithfully the rewriter turns Questions into Claims, measured on a fold.

The Entailment judge reads the Claim and never the Question, so a rewrite that
drops a dose or leaves an interrogative behind is a wrong answer the judge
cannot see and the accuracy numbers cannot attribute. This measures the
rewriter on its own, before it is measured through a threshold.

Five properties, each of which the rewrite is supposed to guarantee and none of
which needs an annotation to check:

- **It asserts.** No question mark survives, and no auxiliary is still fronted.
  A hypothesis that is still a question is not one an NLI model can judge.
- **It keeps what the Question said.** Every word survives except the
  interrogative scaffolding the rules remove on purpose — the tag on a tag
  question, and the phrases that scope a Question over the recording rather
  than over the clinical facts. A Hard Negative turns on one number inside an
  otherwise identical sentence, so a rewrite that loses a word can lose the
  answer.
- **It is a clause.** Re-parsed, the Claim has a subject.
- **It did not cut a word in half.** The auxiliary is moved between two words
  of the Question, never into the middle of one: "mmol/mol" is three tokens
  with no space between them, and an auxiliary landing inside it makes a unit
  into two.
- **It did not strand a coordinator.** No predicate begins with "and" or "or",
  which is what a subject stopping short of its own second conjunct leaves
  behind.

**What this does not check is whether the split is the right one.** Every
property above holds of "the patient complaining is of chest symptoms", which
is a mis-split the parser caused and nothing here can see: word order is
preserved by construction, so re-parsing the Claim reports whatever split it
was handed as the subject. Deciding the boundary is correct needs the parse
this measurement is trying to hold to account, and the residual is recorded in
ADR-0002 by reading the rewrites rather than by asserting a number here.

Shapes are counted beside them, so a fold holding a shape no rule covers shows
up as a count rather than as a silent pass-through.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from harness.folds import FoldName, questions_in_fold
from medapp.claims import (
    AUXILIARIES,
    SCOPE_PREFIX,
    TAG_QUESTION,
    ClaimShape,
    Parser,
    Rewriter,
)

_WORDS = re.compile(r"[a-z0-9]+")
_COORDINATORS = frozenset({"and", "or", "nor", "but"})


@dataclass(frozen=True, slots=True)
class ClaimCheck:
    """One Question, its Claim, and what the rewrite failed to guarantee.

    Attributes:
        faults: The properties that do not hold, empty when the rewrite is
            faithful. Named rather than counted so the printed report says what
            went wrong and not only how often.
    """

    question_id: str
    question: str
    claim: str
    shape: ClaimShape
    faults: tuple[str, ...]

    @property
    def faithful(self) -> bool:
        """Whether every property the rewrite guarantees holds."""
        return not self.faults


@dataclass(frozen=True, slots=True)
class FidelityReport:
    """The rewriter on one fold.

    Attributes:
        shapes: How many Questions were recognized as each shape.
        unfaithful: Every Question whose Claim failed a property, in fold
            order.
    """

    fold: FoldName
    questions: int
    shapes: dict[ClaimShape, int]
    unfaithful: tuple[ClaimCheck, ...]

    @property
    def fidelity(self) -> float:
        """The fraction of Questions whose Claim holds every property."""
        if not self.questions:
            return 0.0

        return (self.questions - len(self.unfaithful)) / self.questions


def check(
    question_id: str, question: str, rewriter: Rewriter, parser: Parser
) -> ClaimCheck:
    """Rewrite one Question and check what the rewrite guarantees.

    Args:
        question_id: The row's id, so a fault can be looked up in the CSV.
        question: The Question as it was asked.
        rewriter: The rewriter under measurement.
        parser: A parser to re-read the Claim with. The rewriter's own is fine:
            what is under test is the Claim, not the parse it came from.
    """
    claim = rewriter.rewrite(question)
    words = claim.text.split()
    faults: list[str] = []

    if "?" in claim.text:
        faults.append("still a question")

    if words and words[0].lower() in AUXILIARIES:
        faults.append("auxiliary still fronted")

    dropped = _words(question) - _words(claim.text) - _scaffolding(question)
    if dropped:
        faults.append(f"lost {' '.join(sorted(dropped))}")

    if not _has_subject(parser, claim.text):
        faults.append("no subject")

    if _splits_a_word(question, claim.text):
        faults.append("auxiliary moved inside a word")

    if _strands_a_coordinator(words, claim.shape):
        faults.append("predicate begins with a coordinator")

    return ClaimCheck(
        question_id=question_id,
        question=question,
        claim=claim.text,
        shape=claim.shape,
        faults=tuple(faults),
    )


def measure_fold(fold: FoldName, rewriter: Rewriter, parser: Parser) -> FidelityReport:
    """Check every Question of one fold.

    Raises:
        KeyError: If ``fold`` is not one of train, dev or test.
    """
    checks = [
        check(row["question_id"], row["question"], rewriter, parser)
        for row in questions_in_fold(fold)
    ]

    return FidelityReport(
        fold=fold,
        questions=len(checks),
        shapes=_shapes(checks),
        unfaithful=tuple(one for one in checks if not one.faithful),
    )


def _shapes(checks: Sequence[ClaimCheck]) -> dict[ClaimShape, int]:
    """How many Questions were recognized as each shape."""
    counted: dict[ClaimShape, int] = {}

    for one in checks:
        counted[one.shape] = counted.get(one.shape, 0) + 1

    return counted


def _words(text: str) -> set[str]:
    """The word-like tokens of a string, for comparing what survived."""
    return set(_WORDS.findall(text.lower()))


def _scaffolding(question: str) -> set[str]:
    """The words the rules remove on purpose, for this Question.

    The tag of a tag question and the phrase that scopes it over the recording.
    Read with the rewriter's own patterns rather than with a second list, so
    that a rule the rewriter changes cannot go unchecked here.
    """
    tag = TAG_QUESTION.search(question.strip().rstrip("?").strip())
    scope = SCOPE_PREFIX.match(question.strip())

    return _words(tag.group(0) if tag else "") | _words(scope.group(0) if scope else "")


def _splits_a_word(question: str, claim: str) -> bool:
    """Whether the rewrite moved the auxiliary into the middle of a word.

    The Claim is the Question's words reordered, so every whitespace-separated
    word of the Claim but the auxiliary itself must be a word of the Question.
    The comma before a tag and the question mark at the end are the Question's
    own punctuation and are stripped from both sides first.
    """
    return bool(_spelled(claim) - _spelled(question) - AUXILIARIES)


def _spelled(text: str) -> set[str]:
    """The whitespace-separated words of a string, without their punctuation."""
    return {
        stripped
        for word in text.lower().split()
        if (stripped := word.strip(".,;:!?\"'()"))
    }


def _strands_a_coordinator(words: Sequence[str], shape: ClaimShape) -> bool:
    """Whether an auxiliary was placed in front of "and" or "or".

    A subject that stopped one conjunct short leaves the coordinator at the
    head of the predicate, where no predicate begins.
    """
    if shape != "auxiliary_fronted":
        return False

    return any(
        following.lower() in _COORDINATORS
        for word, following in zip(words, words[1:], strict=False)
        if word.lower() in AUXILIARIES
    )


def _has_subject(parser: Parser, claim: str) -> bool:
    """Whether the Claim re-parses as a clause with a subject."""
    parsed = parser(claim)
    root = next((token for token in parsed if token.dep_ == "ROOT"), None)

    return root is not None and any(
        child.dep_ in {"nsubj", "nsubjpass", "expl"} for child in root.children
    )
