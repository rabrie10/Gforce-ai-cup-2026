"""The one module that reads the environment.

Device, compute type, model names, the per-request deadline, the Answer
Strategy and the cache directories are resolved here and nowhere else, so no
other module branches on platform. Defaults are the development-machine values;
the deployment VM overrides them:

    MEDAPP_DEVICE=cuda MEDAPP_COMPUTE_TYPE=float16 python api.py
"""

from pathlib import Path
from typing import Literal, cast, get_args

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
RetrievalMode = Literal["bm25", "dense", "hybrid"]
RETRIEVAL_MODES: tuple[RetrievalMode, ...] = get_args(RetrievalMode)
AnswerStrategy = Literal[
    "cite_first_segment",
    "retrieve_bm25",
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
        cpu_threads: How many threads CTranslate2 decodes with. Zero leaves its
            own default, which is conservative on a many-core machine: on the
            11-core dev machine setting it to 10 took the worst-case decode from
            49.0 s to 37.8 s, which is the single cheapest latency win
            available and needs no change of model.
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
        chunk_word_lengths: The ladder of Chunk lengths, in normalized tokens.
            Tuned on the dev fold against oracle-selected tIoU.
        chunk_stride_fraction: How far apart the Chunks of one length start, as
            a fraction of that length. Below 1 they overlap, which annotated
            Evidence Spans require.
        retrieval_candidates: How many ranked Chunks a Verdict carries. The
            judges downstream read them and the component metrics are computed
            from them, so it is deeper than the one Chunk cited. It is also the
            k the reranker is handed: every candidate the retriever ranks is
            rescored.
        retrieval_mode: Which retriever ranks the Chunks a Question is answered
            from — BM25 alone, the dense embedder alone, or the two fused.
            ADR-0001 builds BM25 first and adopts fusion only where it wins on
            Hard-Negative accuracy as well as recall; the embedder is
            constructed only when this names it, so a mode that lost costs the
            request path nothing.
        dense_batch_size: How many Chunks are embedded at once. The whole
            Conversation is embedded once per request, not once per Question.
        fusion_depth: How deep each retriever's ranking is read before the two
            are fused. Deeper than the candidates carried, because a Chunk one
            retriever ranks shallowly is exactly what the other is there to
            rescue.
        fusion_rank_constant: Reciprocal Rank Fusion's constant, the ``k`` in
            ``1 / (k + rank)``. It sets how much a top rank outweighs a deep
            one; 60 is the value the method was published with.
        rerank_batch_size: How many Question-Chunk pairs go through the
            cross-encoder at once. One Question's whole candidate list fits in
            one batch at the default k.
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
        span_pad_start_seconds: Seconds the returned Evidence Span's start is
            extended earlier by, before it is snapped back to a Word edge.
            Chosen on the dev fold for mean tIoU over annotated Positives by
            ``python -m scripts.span_padding``.
        span_pad_end_seconds: The same, for the end. Chosen alongside
            ``span_pad_start_seconds`` by the same sweep.
        span_pad_start_fraction: A further extension of the start, as a
            fraction of the cited Chunk's own duration. A fixed offset cannot
            suit every Chunk — annotated Evidence Spans run from 0.16 s to
            14.2 s and the Chunk ladder from one token to forty-eight — so the
            padding is fixed plus proportional, and both parts are swept
            against the competition score by ``python -m scripts.tune``.
        span_pad_end_fraction: The same, for the end.
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

    whisper_model: str = "large-v3"
    whisper_language: str = "en"
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    beam_size: int = Field(default=5, ge=1, le=10)

    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    dense_model: str = "BAAI/bge-small-en-v1.5"

    chunk_word_lengths: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 48)
    chunk_stride_fraction: float = Field(default=0.2, gt=0, le=1)
    retrieval_candidates: int = Field(default=10, ge=1)

    # Measured on dev by ``python -m scripts.retrieval_modes``, 16
    # Conversations and 160 Questions, every mode answered end to end through
    # the shipped Answerer. BM25 0.855 Hard-Negative accuracy at recall@5
    # 0.667; dense 0.790 at 0.733; fused 0.823 at 0.707. Both candidates raise
    # recall and lower the gate, which is the trade ADR-0001 named in advance,
    # so the baseline stands and the embedder does not load here.
    retrieval_mode: RetrievalMode = "bm25"
    dense_batch_size: int = Field(default=64, ge=1)
    fusion_depth: int = Field(default=50, ge=1)
    fusion_rank_constant: float = Field(default=60.0, gt=0)

    # Judging one Question — ranking plus the cross-encoder over 10 candidates
    # — measured on dev at 180 ms on average and 308 ms worst on the Mac's CPU,
    # so a Conversation's ten Questions cost 3.1 s of the 15 s the latency
    # budget gives the answering half.
    rerank_batch_size: int = Field(default=16, ge=1)
    # The longest Question-Chunk pair on dev is 92 subwords: 69 for the top
    # rung of the Chunk ladder and 23 for the Question. Nothing is truncated.
    rerank_max_tokens: int = Field(default=128, ge=1)
    # Measured on dev, 16 Conversations and 160 Questions: Off-Topic accuracy
    # 1.000 (90% interval [1.000, 1.000]), Positive TPR 0.907, TNR 0.776,
    # overall accuracy 0.838 [0.787, 0.887]. Higher candidates score better
    # overall — 0.863 at 0.79 — entirely by rejecting Hard Negatives, at 0.12
    # of Positive TPR.
    # Held at the value the Relevance sweep chose. The joint sweep in
    # ``python -m scripts.tune`` can score higher by raising it, but only by
    # trading Hard-Negative accuracy for mean tIoU, which the slice floor
    # refuses: Off-Topic accuracy is 1.000 either way and there is nothing here
    # to buy.
    relevance_threshold: float = 0.3

    nli_batch_size: int = Field(default=16, ge=1)
    # The Claim is a rewrite of the Question and no longer, and the premise is
    # one Chunk, so the pair fits well inside the reranker's own limit.
    nli_max_tokens: int = Field(default=128, ge=1)
    # Measured on dev, 16 Conversations and 160 Questions: Hard-Negative
    # accuracy 0.855 (90% interval [0.787, 0.918]) against 0.694 with
    # Relevance alone, Positive accuracy 0.867 against 0.907 — the floor the
    # sweep holds is 0.851, the bottom of ticket 08's interval — Off-Topic
    # 1.000, TNR 0.894, overall accuracy 0.881 [0.831, 0.925] against 0.838.
    # The value is small because the NLI model's softmax saturates: most pairs
    # score within 1e-3 of zero on entailment, and what separates a Positive
    # from a Hard Negative sits in that tail rather than near 0.5.
    # Held at the value the Hard-Negative sweep chose, for the same reason the
    # Relevance threshold is: lowering it scores 0.628 against 0.626 by letting
    # Hard-Negative accuracy fall to 0.790, and buying two thousandths of the
    # score with six hundredths of the judgement the case is about is not a
    # trade worth making on a fold of 16 Conversations.
    entailment_threshold: float = 0.000394
    # Chosen at 2 by ``python -m scripts.tune``, jointly with the thresholds
    # and the pads, on dev: score 0.626 (90% interval [0.578, 0.674]) against
    # 0.618 at depth 1, with Positive accuracy 0.867 to 0.893, missed Positives
    # 10 to 8 and mean tIoU 0.443 to 0.451. Both halves move, which is the
    # point: judging two Chunks and citing the one that entailed makes the
    # answer and the Evidence Span one decision, and the Chunk Relevance ranks
    # first is often not the one that states the Claim.
    entail_depth: int = Field(default=2, ge=1)
    # Ablated at that threshold on dev: Off-Topic accuracy 1.000 with the gate
    # and 0.000 without it. Entailment does not reject an Off-Topic Question on
    # its own — the best Chunk of an unrelated Conversation still entails a
    # loosely-worded Claim often enough — so the gate stays, at 0.093 of
    # Positive accuracy.
    relevance_gate: bool = True

    # Chosen by the same joint sweep, and no longer fixed offsets alone. A
    # fixed pad has to suit both a Chunk cut at one token and one cut at
    # forty-eight, and it cannot: the fractions come out negative against
    # positive seconds, which is the sweep saying "extend every Chunk by about
    # a word, and trim the long ones back in proportion". Mean tIoU 0.468
    # against 0.443 for the best purely fixed padding, on the same recording.
    span_pad_start_seconds: float = 0.75
    span_pad_end_seconds: float = 0.30
    span_pad_start_fraction: float = -0.20
    span_pad_end_fraction: float = -0.10

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


def checked_retrieval_mode(mode: str) -> RetrievalMode:
    """The mode name, checked where it arrives as free text.

    Raises:
        ValueError: If it names no retrieval mode. A typo that reached
            ``build_index_factory`` would be measured as whichever mode the
            fall-through built and reported under the name that was typed.
    """
    if mode not in RETRIEVAL_MODES:
        raise ValueError(
            f"{mode!r} is not a retrieval mode: the modes are "
            f"{', '.join(RETRIEVAL_MODES)}."
        )

    return cast(RetrievalMode, mode)


settings = Settings()
