"""The one module that reads the environment.

Device, compute type, model names, the per-request deadline, the Answer
Strategy and the cache directories are resolved here and nowhere else, so no
other module branches on platform. Defaults are the development-machine values;
the deployment VM overrides them:

    MEDAPP_DEVICE=cuda MEDAPP_COMPUTE_TYPE=float16 python api.py
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Device = Literal["cpu", "cuda"]
ComputeType = Literal["int8", "int8_float16", "float16", "float32"]
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
        answer_strategy: Which Answerer is constructed at startup. Candidates
            are compared by running the system twice, not by a runtime switch.
            A strategy no Answerer implements yet fails at startup rather than
            being quietly substituted.
        route_suffix: Appended to the ``/predict`` path so the deployed
            endpoint sits at an unguessable route. Empty in development.
        deadline_seconds: Checked between stages and between Questions, and
            strictly under the service's 60-second request timeout.
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
            k the reranker is handed: every candidate BM25 ranks is rescored.
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
    compute_type: ComputeType = "int8"

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
    entailment_threshold: float = 0.000394
    # Ablated at that threshold on dev: Off-Topic accuracy 1.000 with the gate
    # and 0.000 without it. Entailment does not reject an Off-Topic Question on
    # its own — the best Chunk of an unrelated Conversation still entails a
    # loosely-worded Claim often enough — so the gate stays, at 0.093 of
    # Positive accuracy.
    relevance_gate: bool = True

    answer_strategy: AnswerStrategy = "retrieve_rerank_entail"
    route_suffix: str = ""

    deadline_seconds: float = Field(default=50.0, gt=0, lt=60)

    model_cache_dir: Path = PROJECT_ROOT / "models"
    transcript_cache_dir: Path = PROJECT_ROOT / "transcripts"


settings = Settings()
