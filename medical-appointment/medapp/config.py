import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# CTranslate2's device set, which the Transcriber runs on. Metal is not in it:
# the shipped ctranslate2 wheels build a CPU and a CUDA backend and nothing
# else, so on Apple Silicon the Transcriber is a CPU program however many GPU
# cores the machine has. ``cpu_threads`` is what tunes it there, not a device.
Device = Literal["cpu", "cuda"]
# PyTorch's device set, which the cross-encoders run on. Wider than
# CTranslate2's by exactly one entry, which is why it cannot be the same
# setting: "mps" is valid for the reranker and the NLI judge and invalid for
# the Transcriber, and one Literal covering both would either forbid it
# everywhere or admit it where it raises.
TorchDevice = Literal["auto", "cpu", "cuda", "mps"]
ComputeType = Literal["int8", "int8_float16", "float16", "float32"]
AnswerStrategy = Literal[
    "cite_first_segment",
    "retrieve_rerank",
    "retrieve_rerank_entail",
    "single_llm",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Every value the system resolves from its environment.

    Attributes:
        device: Where the Transcriber decodes — CTranslate2's device set, which
            has no Metal entry. See :data:`Device`.
        torch_device: Where the cross-encoders run. ``auto`` resolves to the
            best available at startup, which is Metal on Apple Silicon. Measured
            rather than assumed: on the dev machine it is worth about 2% over
            CPU, because ten Question-Chunk pairs of under a hundred subwords do
            not fill a GPU and the kernel launches cost what the arithmetic
            saves. It is wired correctly anyway, because a machine whose
            batches are larger would see the win and because "cannot be
            selected" is a worse answer than "measured and did not help".
        cpu_threads: How many threads CTranslate2 decodes with. Zero resolves
            to one fewer than the machine has, leaving a core for the event
            loop; CTranslate2's own default is a fixed 4, which leaves most of
            a many-core machine idle. On the 11-core dev machine this took the
            worst-case decode from 49.0 s to 37.8 s — the cheapest latency win
            available, and one that needs no change of model. Resolved rather
            than set to a literal so the number is right on whatever machine
            serves, instead of right here and wrong elsewhere.
        answer_strategy: Which Answerer is constructed at startup. Candidates
            are compared by running the system twice, not by a runtime switch.
            A strategy no Answerer implements yet fails at startup rather than
            being quietly substituted.
        route_suffix: Appended to the ``/predict`` path so the deployed
            endpoint sits at an unguessable route. Empty in development.
        deadline_seconds: Checked between stages and between Questions, and
            strictly under the service's 60-second request timeout.
        answering_reserve_seconds: How much of the deadline is kept back from
            transcription so the ten Questions can still be answered against
            whatever was decoded. Decoding stops at the rest of the budget and
            the Conversation is answered truncated, which scores something;
            decoding until the request times out scores nothing on all ten.
        model_cache_dir: Where model weights live, pre-fetched into the image
            at build time rather than downloaded on first request.
        transcript_cache_dir: The development transcript cache. Written and
            read by ``scripts/`` only; nothing under ``medapp`` touches it.
        whisper_language: Fixed rather than detected per Conversation, so a
            short or noisy opening cannot send one transcript through a
            different language's decoder.
        vad_filter: Skips silence before decoding, and is the standard guard
            against the hallucination loops that multiply decode time on a
            quiet stretch.
        condition_on_previous_text: Off for the same reason; a repetition that
            enters the prompt otherwise sustains itself.
        beam_size: Worst-case latency scales with it, so it is tuned on the dev
            fold against timing error rather than taken on faith.
        chunk_max_words: Where a run of Words the Transcriber punctuated
            nowhere is cut into a Chunk anyway. A guard, not a granularity: a
            Chunk is one sentence.
        retrieval_candidates: How many ranked Chunks a Verdict carries. Every
            sentence of the Conversation is scored; this is how many of them
            the Verdict exposes, which the judges downstream read and the
            component metrics are computed from.
        rerank_batch_size: How many Question-Chunk pairs go through the
            cross-encoder at once. A Conversation's sentences are scored in one
            call, in batches of this size.
        rerank_max_tokens: Where a Question-Chunk pair is truncated, counted in
            the model's subwords rather than in the words the Chunk ladder is
            cut at. Slack rather than a limit, and well under the model's 8192,
            which would cost padding for nothing.
        relevance_threshold: The Relevance a Conversation's best Chunk must
            reach for the answer to be yes. Below it nothing in the
            Conversation is about what the Question asks about, which is what
            an Off-Topic Question looks like. Chosen on the dev fold for
            stability under bootstrap resampling by
            ``python -m scripts.relevance_threshold``, never by argmax.
        nli_batch_size: How many Chunk-Claim pairs go through the NLI model at
            once.
        nli_max_tokens: Where a Chunk-Claim pair is truncated, counted in the
            model's subwords.
        entailment_threshold: The entailment probability the best Relevant
            Chunk must reach for the answer to be yes. Below it the Chunk
            either contradicts the Claim or is silent on it, and CONTEXT.md
            answers both no. Chosen on the dev fold for stability under
            bootstrap resampling by ``python -m scripts.entailment_threshold``,
            on Hard-Negative accuracy with Positive accuracy held at its
            Relevance-only value.
        entail_depth: How many of the top Relevant Chunks the Entailment judge
            reads. The Verdict cites the best-entailing of them, which is what
            makes the answer and the Evidence Span one decision. Swept against
            the competition score by ``python -m scripts.entail_depth``.
        relevance_gate: Whether the Relevance threshold still rejects a
            Question before Entailment is judged. Kept only because the
            ablation the same script runs measured Off-Topic accuracy dropping
            without it.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEDAPP_",
        # ``model_`` is a pydantic-protected prefix and several fields are
        # model names.
        protected_namespaces=(),
        frozen=True,
    )

    device: Device = "cpu"
    torch_device: TorchDevice = "auto"
    compute_type: ComputeType = "int8"
    cpu_threads: int = Field(default=0, ge=0)

    # Measured on the dev machine against 'large-v3' on the same 39
    # Conversations. Worst-case decode 140.2 s to 37.8 s — 'large-v3' cannot
    # serve a 60-second budget on a CPU at all — and the score does not pay for
    # the speed: 0.642 against 0.626 on dev, with the annotated-boundary
    # distance a word edge can reach *halved* at the median, 0.120 s to
    # 0.060 s, which is the floor under every tIoU this system can score.
    whisper_model: str = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    whisper_language: str = "en"
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    # One, not five. On 'large-v3-turbo' the wider beam bought nothing
    # measurable and cost 11 s of the worst-case decode (49.0 s against
    # 37.8 s at ten threads), which is latency margin this budget has better
    # uses for.
    beam_size: int = Field(default=1, ge=1, le=10)

    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # The supplied Conversations run a median of five Words to the sentence
    # and a longest of fifty-nine, so this fires in none of them. It is the
    # guard against a Conversation the Transcriber punctuates sparsely, where
    # one Chunk would otherwise cover the whole audio.
    chunk_max_words: int = Field(default=60, ge=1)
    retrieval_candidates: int = Field(default=10, ge=1)

    # Judging one Question — ranking plus the cross-encoder over 10 candidates
    # — measured on dev at 180 ms on average and 308 ms worst on the Mac's CPU,
    # so a Conversation's ten Questions cost 3.1 s of the 15 s the latency
    # budget gives the answering half.
    rerank_batch_size: int = Field(default=16, ge=1)
    # The longest Question-Chunk pair on dev is 92 subwords: 69 for the top
    # rung of the Chunk ladder and 23 for the Question. Nothing is truncated.
    rerank_max_tokens: int = Field(default=128, ge=1)
    # Chosen jointly with the Entailment threshold and depth by
    # ``python -m scripts.tune``, over train and dev together — 24
    # Conversations and 240 Questions, not the 16 the earlier value was read
    # off. It is far lower than the 0.4 the word-ladder Chunker wanted, and for
    # a reason rather than by drift: a Chunk cut at one to four tokens scores
    # high on a cross-encoder for a Question it shares those tokens with, so
    # the ladder needed a high bar to reject anything. A sentence is a whole
    # utterance and scores lower on the same match, and holding the ladder's
    # bar against it rejected 33 of 122 Positives.
    relevance_threshold: float = 0.1

    nli_batch_size: int = Field(default=16, ge=1)
    # The Claim is a rewrite of the Question and no longer, and the premise is
    # one Chunk, so the pair fits well inside the reranker's own limit.
    nli_max_tokens: int = Field(default=128, ge=1)
    # Chosen by the same joint sweep. The value is small because the NLI
    # model's softmax saturates: most pairs score within 1e-3 of zero on
    # entailment, and what separates a Positive from a Hard Negative sits in
    # that tail rather than near 0.5. It is a real boundary in that tail and
    # not a formality — over train and dev it is worth 0.906 Hard-Negative
    # accuracy against 0.765 at a threshold of zero, at no cost in Positives.
    entailment_threshold: float = 0.0002
    # One, chosen by the same sweep. Judging the second-most Relevant sentence
    # as well raises Positive accuracy — 0.861 against 0.844 — and loses more
    # than it gains on the tIoU half, 0.430 against 0.459, because the Chunk it
    # cites when it wins is the one Relevance ranked second. With a Chunk per
    # sentence rather than a dozen overlapping windows per stretch, the top
    # Relevant Chunk is already the one that states the Claim where any of them
    # does.
    entail_depth: int = Field(default=1, ge=1)
    # Ablated on train and dev: Off-Topic accuracy 1.000 with the gate and
    # 0.000 without it. Entailment does not reject an Off-Topic Question on
    # its own — the best Chunk of an unrelated Conversation still entails a
    # loosely-worded Claim often enough — so the gate stays, at 0.093 of
    # Positive accuracy.
    relevance_gate: bool = True

    answer_strategy: AnswerStrategy = "retrieve_rerank_entail"
    route_suffix: str = ""

    deadline_seconds: float = Field(default=50.0, gt=0, lt=60)
    # Judging one Question is 180 ms on average and 308 ms worst on dev, so ten
    # cost about 3.1 s. The reserve is that, doubled for the Chunker and the
    # per-request index built once ahead of them, and doubled again as margin
    # against a Conversation longer than any supplied one.
    answering_reserve_seconds: float = Field(default=12.0, ge=0)

    model_cache_dir: Path = PROJECT_ROOT / "models"
    transcript_cache_dir: Path = PROJECT_ROOT / "transcripts"


def resolved_cpu_threads(cpu_threads: int) -> int:
    """How many threads to decode with on this machine.

    Args:
        cpu_threads: The configured count. Zero means resolve it here.

    Returns:
        The configured count, or one fewer than this machine's cores when it
        was zero. At least one, so a single-core machine still decodes.
    """
    if cpu_threads > 0:
        return cpu_threads

    return max(1, (os.cpu_count() or 2) - 1)


def resolved_torch_device(device: TorchDevice) -> str:
    """The PyTorch device to load a cross-encoder on.

    ``auto`` picks the best backend the machine actually has, checked at
    startup rather than inferred from the platform name: a Mac without a
    working Metal build and a Linux box without a visible GPU both have to fall
    back to CPU, and asking torch is the only way to know.

    Args:
        device: The configured device, possibly ``auto``.

    Returns:
        A device string torch accepts.
    """
    if device != "auto":
        return device

    # Imported here rather than at module scope: importing torch costs seconds
    # and every module reads Settings, including the ones that never load a
    # model.
    import torch

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


settings = Settings()
