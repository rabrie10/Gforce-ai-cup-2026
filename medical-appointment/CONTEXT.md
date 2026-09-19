# Context

The domain language of the medical-appointment case. A glossary, not a spec.

## Conversation

One recorded consultation between a doctor and a patient, arriving as a single
MP3. A Conversation is the unit of a request: it arrives complete, with every
Question asked about it, and is never sent again. Nothing is shared between
Conversations.

## Question

A yes/no question asked about exactly one Conversation. Ten arrive per
Conversation. A Question is one of three kinds:

- **Positive** — something the Conversation establishes. True answer: yes.
- **Hard Negative** — a near-miss on something the Conversation establishes:
  the right drug at the wrong dose, the right course at the wrong length. True
  answer: no. Lexically almost identical to the true statement.
- **Off-Topic** — a subject the Conversation never raises. True answer: no.

"Hard Negative" is not a negative *answer*; it names how the Question was
constructed. Both Hard Negative and Off-Topic Questions are answered no.

## Evidence Span

The stretch of audio a yes is read from, as a start and end second from the
beginning of the Conversation. An Evidence Span exists only for a Question
whose true answer is yes; a no has nothing to point at, by definition.

An Evidence Span is a **passage, not a region**. In the supplied annotations it
runs a median of 2.88 seconds. A span covering the whole Conversation is not a
hedge — it scores near nothing.

## Segment

A contiguous stretch of transcribed speech with its own start and end time, as
produced by the Transcriber. A Segment is a unit of *transcription*; an
Evidence Span is a unit of *evidence*. They are not the same thing and one
Evidence Span does not necessarily correspond to one Segment.

## Transcriber

The component that turns a Conversation's audio into timed Segments. A single
implementation configured by model, device and compute type — not a set of
interchangeable backends.

## Answerer

The component that decides, for one Question against one Conversation's
Segments, whether the answer is yes and which Evidence Span makes it so. Those
two outputs are one decision, not two: the passage that justifies a yes is
exactly the Evidence Span to return.

Selected at startup by configuration so that candidate Answerers can be
measured against each other. There is no runtime switch between them and no
degraded Answerer to fall back to.

## Normalizer

The one written form a Question and the speech that answers it are both reduced
to, so that "41 mmol/mol" and *"41 millimoles per mole"* are the same tokens.
Casing, punctuation, spelled-out numbers, units and the clinical abbreviations
the error analysis listed are all its job.

It normalizes *tokens*, not strings: a token carries the Words it was read from,
including the several Words a merged token covers, because an Evidence Span is
returned as Word timings and a token that has lost its Words cannot back one.

## Chunk

A candidate Evidence Span: **one spoken sentence**, built from timed words
rather than from Segment boundaries, and carrying the word timings that let it
be returned as an Evidence Span unchanged.

Chunks do not overlap and come in one kind. A scheme of overlapping windows at
a ladder of lengths represents more annotated Evidence Spans — its oracle is
higher — and cites worse ones, because a dozen views of the same stretch differ
only in their boundaries and nothing downstream ranks boundaries. ADR-0004
records the measurement. Coverage the Answerer cannot select within is not
coverage.

Chunk is not a synonym for Segment. A Segment is what the Transcriber produced
and runs several sentences long; a Chunk is what the Answerer considers.

## Score

What the attempt is graded on: `0.4 x Accuracy + 0.6 x mean tIoU`. Not a
component metric and not an average of them. The tIoU half is averaged over
every annotated Evidence Span whether the Question was answered yes or not, so
a Positive answered no scores zero twice — once on each half — which makes
rejecting a Positive about twice as expensive as accepting a Hard Negative.
Every threshold is chosen against this, under the slice floor ADR-0003 records;
the component measurements say which judgement moved, not whether the move was
worth making.


## Relevance

Whether a Chunk is *about* what a Question asks about. Low Relevance across
every Chunk in a Conversation is what an Off-Topic Question looks like: nothing
in the Conversation comes close, so the answer is no.

Relevance alone cannot answer a Hard Negative. A Hard Negative's best Chunk
scores very high on Relevance precisely because it is lexically near-identical
to the truth.

## Entailment

Whether a Chunk actually *establishes* what a Question asserts, as opposed to
merely discussing the same subject. Entailment is what separates a Positive
Question from a Hard Negative: both find a highly Relevant Chunk, but only the
Positive's Chunk entails the claim. The Hard Negative's Chunk either contradicts
it or is silent on it — a plausible detail that was never agreed is neutral,
not contradicted, and is still a no. Yes requires entailment; nothing less.

Relevance and Entailment are distinct judgements answering distinct failures. A
single score cannot do both: the threshold that rejects an Off-Topic Question
would have to reject the very Chunks that make Positive Questions true.

The Chunk that entails is the Chunk cited. That is the same statement as the
Answerer's: the answer and the Evidence Span are one decision, and a Verdict
carries the Chunk it read the answer from rather than a span beside a ranking,
so the two cannot drift apart.

How many of the top Relevant Chunks are judged is a measured quantity rather
than a principle. With overlapping Chunks it had to be more than one, because
Relevance ranks by what a passage is *about* and among a dozen views of one
stretch it has no reason to prefer the one that states the Claim. With one
Chunk per sentence the top Relevant Chunk is already that one wherever any of
them is, and judging deeper costs more tIoU than the Positives it rescues are
worth.
