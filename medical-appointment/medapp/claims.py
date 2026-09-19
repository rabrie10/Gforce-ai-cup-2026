"""The Question rewritten as the Claim it asserts.

An NLI model judges a *premise* against a *hypothesis*, and both are
declarative. A Question is not: "Did the doctor prescribe erythromycin?"
asserts nothing, and handing it to the Entailment judge as written asks the
model to decide whether a passage entails a question, which is not a judgement
it was trained to make. So the Question is first rewritten into the Claim a yes
would be agreeing with — "the doctor did prescribe erythromycin" — and that is
what the judge reads.

The rewrite is rule-based over the shapes the supplied Questions come in, which
are three:

- **Tag questions** — "The HbA1c came out at 47 mmol/mol, right?" The
  declarative is already there; the tag comes off.
- **Auxiliary-fronted questions** — "Is the patient treated for diabetes?",
  "Does the patient own a cat?", "Is there any mention of a concert?" The
  fronted auxiliary moves back behind the subject.
- **Declaratives** — anything already in the asserting form, which is what the
  first two reduce to.

Auxiliary-fronted questions are un-fronted rather than re-conjugated: "Did the
patient receive a tetanus vaccination?" becomes "the patient did receive a
tetanus vaccination" and not "the patient received a tetanus vaccination".
Emphatic *do* is the same assertion in English, and it is reached by moving one
token rather than by inflecting a main verb, which would need an irregular-verb
lexicon to get "come" to "came" and would be wrong wherever the lexicon was
thin. The Claim's job is to be a faithful declarative, not an elegant one.

Finding where the subject ends is the one part a pattern cannot do: "Did the
doctor plan continued follow-up?" and "Does the plan include a referral?" put
the same word on either side of the boundary. So the Question is parsed, and
the subject is read off the parse — the auxiliary's own attachment first, which
is what the parser is most reliable about on an interrogative, and the subject
of the fronted copula after it.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:  # pragma: no cover - imported for typing only.
    from spacy.tokens import Doc, Span, Token

# The Question's spelling, not the Conversation's: these are read before the
# normalizer, because the comma before a tag and the question mark at the end
# are what the shapes are recognized by and the normalizer strips both.
AUXILIARIES = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "am",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
    }
)

# A tag appended to a declarative: ", right?", ", correct?", ", wasn't it?".
# The declarative in front of it is the Claim, whole.
TAG_QUESTION = re.compile(
    r",\s*(?:right|correct|true|"
    r"(?:is|was|were|are|did|does|do|has|have|had|will|would|should|could|ca)"
    r"\s?n[o']?t\s+(?:it|they|he|she|there))\s*$",
    re.IGNORECASE,
)

# Phrases that scope the Question over the recording rather than over the
# clinical facts, and that a declarative Claim has no room for: "At any point
# does the patient..." asserts "the patient does...". "At any point" is also a
# negative-polarity phrase, so carrying it into an affirmative Claim would
# leave the Claim ungrammatical as well as redundant.
SCOPE_PREFIX = re.compile(
    r"^(?:at any point|according to the conversation|in the conversation|"
    r"during the conversation|based on the conversation)\s*,?\s+",
    re.IGNORECASE,
)

# What sits between the subject and the main verb and belongs with the
# predicate rather than the subject: auxiliaries stacked behind the fronted one
# — "Is the check-up *being* done?" — the negation, and the adverb of "Is the
# patient *still* being treated?", which the fronted auxiliary moves in front
# of.
_STACKED_AUXILIARY_DEPS = frozenset({"aux", "auxpass", "neg", "advmod"})

# Where a fronted copula hangs its subject, in the order they are looked for. A
# well-parsed copular question puts it on ``nsubj``; ``attr`` is where the
# parser puts it when it reads the predicate as the complement instead, and
# ``expl`` is the "there" of an existential.
_SUBJECT_DEPS = ("nsubj", "nsubjpass", "expl", "attr")

# What a subject noun phrase carries on to its right: a prepositional phrase
# ("four daily doses *of one million IU of penicillin*") and a coordinate
# ("redness *and swelling*"). A participle or an adjective hanging off the same
# noun is the predicate the auxiliary introduces rather than part of the
# subject, so the part of speech is checked as well as the dependency — the
# parser labels "the patient's tiredness persistent" as a coordination too.
_NOMINAL_POS = frozenset({"NOUN", "PROPN", "PRON", "NUM"})

# A clause hanging off the subject noun is the predicate, never part of the
# subject.
_PREDICATE_DEPS = frozenset({"acl", "relcl", "advcl"})

# The small pipeline. The large one was measured against it over all 387
# supplied Questions and is a wash — it reads six subject boundaries correctly
# that this one misses and five that this one gets right, which is not 425 MB
# of image for a rewrite that is already right 97% of the time.
SPACY_MODEL = "en_core_web_sm"

ClaimShape = Literal["tag_question", "auxiliary_fronted", "declarative"]


@dataclass(frozen=True, slots=True)
class Claim:
    """What a yes to one Question would be agreeing with.

    Attributes:
        text: The declarative, as written — the Entailment judge normalizes it
            into the Conversation's written form, exactly as the reranker
            normalizes a Question.
        shape: Which of the three shapes the Question was recognized as, which
            is what the fidelity measurement counts.
    """

    text: str
    shape: ClaimShape


class Parser(Protocol):
    """The part of a spaCy pipeline this module uses."""

    def __call__(self, text: str) -> "Doc": ...


class Rewriter(Protocol):
    """Turns a Question into the Claim a yes would be agreeing with."""

    def rewrite(self, question: str) -> Claim:
        """Rewrite one Question as the Claim it asserts."""
        ...


class ClaimRewriter:
    """Rewrites Questions as the Claims they assert."""

    def __init__(self, parser: Parser | None = None) -> None:
        """Load the parser, unless one is supplied.

        Args:
            parser: An already-loaded spaCy pipeline. The argument exists so
                tests and the dev-loop scripts can hold one across many
                Questions rather than reloading it per call.
        """
        self._parser = parser if parser is not None else load_parser()

    def rewrite(self, question: str) -> Claim:
        """Rewrite one Question as the Claim a yes would agree with.

        Args:
            question: The Question as it was asked, punctuation included.

        Returns:
            The Claim, with the shape the Question was recognized as. A
            Question that is already declarative is returned as one rather than
            being forced through a rule that does not fit it.
        """
        text = question.strip().rstrip("?").strip()

        tag = TAG_QUESTION.search(text)
        if tag:
            return Claim(text=text[: tag.start()].strip(), shape="tag_question")

        text = SCOPE_PREFIX.sub("", text)
        prefix, clause = _split_fronted_adverbial(text)

        if not clause or clause.split()[0].lower() not in AUXILIARIES:
            return Claim(text=f"{prefix}{clause}".strip(), shape="declarative")

        parsed = self._parser(clause)

        if _is_misparsed_do_support(parsed):
            # Fronted "do" is an auxiliary in every English question, so a
            # parse that made it the clause's own verb has misread the
            # sentence, and the subject boundary it reports is not one to move
            # a token across. Do-support carries no meaning of its own, so it
            # is dropped rather than put somewhere the parse cannot justify:
            # "the patient complain of dryness" reads as the assertion it is,
            # where "the patient complain of dryness does" does not.
            return Claim(
                text=f"{prefix}{parsed[1:].text}".strip(), shape="auxiliary_fronted"
            )

        end = _subject_end(parsed)

        if end is None:
            return Claim(text=f"{prefix}{clause}".strip(), shape="declarative")

        return Claim(
            text=" ".join(
                part
                for part in (
                    f"{prefix}{parsed[1:end].text}",
                    parsed[0].text.lower(),
                    parsed[end:].text,
                )
                if part
            ).strip(),
            shape="auxiliary_fronted",
        )


def _split_fronted_adverbial(text: str) -> tuple[str, str]:
    """Separate an adverbial fronted before the auxiliary from the clause.

    "Alongside asthma, is the patient treated for diabetes?" fronts an
    adverbial the Claim keeps — it is part of what the Question asserts, unlike
    the scoping phrases above — and the auxiliary to move sits behind it.

    Returns:
        The adverbial with its comma and a trailing space, empty when there is
        none, and the clause the auxiliary fronts.
    """
    head, separator, tail = text.partition(",")

    if separator and tail.strip() and tail.split()[0].lower() in AUXILIARIES:
        return f"{head.strip()}, ", tail.strip()

    return "", text


def _is_misparsed_do_support(parsed: "Doc") -> bool:
    """Whether the parser read fronted do-support as the clause's own verb.

    It is never that: "Does the patient complain?" is do-support, and a parse
    that makes "Does" the root with the rest of the clause hanging off it as a
    noun phrase has lost the main verb inside that phrase, which is exactly the
    boundary the rewrite needs.
    """
    auxiliary = parsed[0]

    return auxiliary.lower_ in {"do", "does", "did"} and auxiliary.dep_ == "ROOT"


def _subject_end(parsed: "Doc") -> int | None:
    """Where the subject of the fronted auxiliary ends, as a token index.

    Two readings, tried in order. A fronted auxiliary the parser attached to a
    verb further along — "Does the patient *complain*", "Will the patient be
    *referred*" — bounds the subject between the two, which is the reading the
    parser is most dependable about on an interrogative. A fronted copula has
    no such verb, so its subject is read off its own dependents instead.

    Returns:
        The index one past the last token of the subject, or None if the
        Question is not the auxiliary-fronted shape after all.
    """
    auxiliary = parsed[0]
    head = auxiliary.head

    if auxiliary.dep_ in {"aux", "auxpass"} and head.i > auxiliary.i + 1:
        end: int = head.i
        while end > 1 and parsed[end - 1].dep_ in _STACKED_AUXILIARY_DEPS:
            end -= 1

        # A clause hanging off the subject — "the vaccination *given as part
        # of diabetes care*" — is the predicate the auxiliary introduces, and
        # a parse that put the clause's root past it has put the boundary in
        # the wrong place.
        end = min(
            (token.i for token in parsed[1:end] if token.dep_ in _PREDICATE_DEPS),
            default=end,
        )

        if end > 1:
            return end

    subject = _copula_subject(parsed)

    if subject is None:
        return None

    # The same rule the extension follows: a subject that ran to the end of the
    # clause has left the fronted copula nothing to predicate, so part of it is
    # the predicate however the parse labelled it.
    if subject.end >= len(parsed) and subject.end > 2:
        subject = parsed[subject.start : _predicate_starts_at(parsed, subject)]

    return _extend_over_modifiers(parsed, subject)


def _predicate_starts_at(parsed: "Doc", subject: "Span") -> int:
    """Where to cut a subject that swallowed the whole clause.

    A measurement is the predicate of the copula that introduces it — "Was the
    HbA1c 42 mmol/mol?" says the HbA1c *was* that — so the cut goes in front of
    the number where the span holds one it did not start with. Failing that the
    last word is the predicate, and it is the last *word*: "mmol/mol" is three
    tokens with no space between them, and cutting inside it would put the
    auxiliary in the middle of a unit.
    """
    number: int | None = next(
        (
            token.i
            for token in parsed[subject.start + 1 : subject.end]
            if token.pos_ == "NUM"
        ),
        None,
    )

    if number is not None and number > subject.start:
        return number

    last_word: int = next(
        (
            index
            for index in range(subject.end - 1, subject.start + 1, -1)
            if parsed[index - 1].whitespace_
        ),
        subject.end - 1,
    )

    return last_word


def _copula_subject(parsed: "Doc") -> "Span | None":
    """The noun phrase a fronted copula predicates over.

    The dependent is one token; the phrase around it is what moves, so the
    parser's own noun-phrase chunking supplies the boundaries where it covers
    the dependent and starts at the front of the clause.
    """
    for dependency in _SUBJECT_DEPS:
        for child in parsed[0].children:
            if child.dep_ != dependency or child.i < 1:
                continue

            for chunk in parsed.noun_chunks:
                if chunk.start <= child.i < chunk.end and chunk.start <= 1:
                    return parsed[1 : chunk.end]

            return parsed[1 : child.i + 1]

    # Neither reading applied, which on these Questions means the parse went
    # wrong rather than that the shape is different — the auxiliary is fronted
    # and something follows it. The noun phrase at the front of the clause is
    # the subject; failing even that, its first token is.
    for chunk in parsed.noun_chunks:
        if chunk.start <= 1 < chunk.end:
            return parsed[1 : chunk.end]

    return parsed[1:2] if len(parsed) > 2 else None


def _extend_over_modifiers(parsed: "Doc", subject: "Span") -> int:
    """Carry the subject over the modifiers that hang to its right.

    A prepositional phrase and a coordinated noun belong to the subject and sit
    after its head; a participle or an adjective hanging off the same head is
    the predicate the auxiliary introduces. So the span grows only over the
    former, and only as far as the preposition's own object — "an appointment
    *for the removal* going to be arranged" hangs the predicate off the
    preposition, and taking the whole subtree would take that too. A
    conjunction extends it only when a noun actually follows, or "redness and
    swelling" would take the "and" and leave the noun behind the auxiliary.

    The exception is an adjective behind an indefinite pronoun — "anything
    abnormal" — where English puts the modifier after its head and the subject
    really does run past it.

    No extension may reach the end of the clause. "Is the heart examination
    without abnormal findings?" hangs its predicate off the subject noun
    exactly as "Are the elements on the abdomen about 2 cm across?" hangs a
    modifier there, and nothing in the parse tells them apart — but a subject
    that has swallowed the whole clause has left the auxiliary nothing to
    introduce, and that is the one reading that cannot be right.
    """
    end: int = subject.end
    inside = {token.i for token in subject}

    if (
        subject.root.pos_ == "PRON"
        and end < len(parsed)
        and parsed[end].dep_ == "amod"
        and parsed[end].head.i == subject.root.i
    ):
        inside.add(end)
        end += 1

    while end < len(parsed):
        token: Token = parsed[end]

        if token.dep_ == "prep" and token.pos_ == "ADP" and token.head.i in inside:
            taken = next(
                (child for child in token.children if child.dep_ == "pobj"), None
            )
        elif token.dep_ == "cc":
            # A coordinator cannot begin a predicate, so the subject carries
            # over it either way: onto the noun the parser coordinated, or —
            # where the parser read that noun as something else — onto
            # whatever token follows. "Are redness and swelling absent?" is
            # the second case, and stopping at the "and" would put the
            # auxiliary in front of it.
            taken = _coordinated_noun(parsed, token, inside) or _after(parsed, token)
        else:
            taken = None

        if taken is None:
            break

        reached = max(descendant.i for descendant in taken.subtree) + 1
        if reached >= len(parsed):
            break

        inside |= {descendant.i for descendant in taken.subtree}
        end = reached

    return end


def _after(parsed: "Doc", token: "Token") -> "Token | None":
    """The token following this one, if the clause has one."""
    return parsed[token.i + 1] if token.i + 1 < len(parsed) else None


def _coordinated_noun(
    parsed: "Doc", conjunction: "Token", inside: set[int]
) -> "Token | None":
    """The noun a conjunction coordinates onto the subject, if there is one."""
    following = parsed[conjunction.i + 1 : conjunction.i + 3]

    return next(
        (
            token
            for token in following
            if token.dep_ == "conj"
            and token.pos_ in _NOMINAL_POS
            and token.head.i in inside
        ),
        None,
    )


def load_parser(model: str = SPACY_MODEL) -> Parser:
    """Load the parser, with the pipeline trimmed to what the rewrite reads.

    Tagging, parsing and noun-phrase chunking are what locate the subject.
    Named entities and word vectors are not read here and cost time per
    Question, so those components are left out.

    Raises:
        OSError: If the model is not installed. It is a dependency rather than
            a download, so this is a broken environment and not something to
            work around at runtime.
    """
    # Imported here rather than at module scope: loading spaCy costs a second,
    # and the harness modules that only need the Chunker should not pay for it.
    import spacy

    loaded: Parser = spacy.load(model, exclude=["ner", "lemmatizer"])

    return loaded
