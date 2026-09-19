"""Reciprocal Rank Fusion over two rankings, and the factory Settings resolve.

The retrievers are stubbed, because what fusion is is arithmetic over two
orderings: a Chunk both retrievers rank near the top beats one that either
ranks first alone, and a Chunk only one of them found is still reachable.
Whether fusing helps on this corpus is measured on the folds by
``python -m scripts.retrieval_modes``, not asserted here.
"""

from collections.abc import Sequence

import numpy as np
import pytest

from medapp.config import Settings
from medapp.dense import DenseIndex
from medapp.retrieval import Bm25Index, FusedIndex, build_index_factory
from medapp.types import Chunk


class _Ranks:
    """A retriever that returns a fixed ranking, and records the depth asked."""

    def __init__(self, ranked: Sequence[Chunk]) -> None:
        self.ranked = tuple(ranked)
        self.depths: list[int] = []

    def rank(self, question: str, limit: int) -> tuple[Chunk, ...]:
        self.depths.append(limit)

        return self.ranked[:limit]


def chunk(name: str) -> Chunk:
    return Chunk(
        text=name, start=float(ord(name[0])), end=float(ord(name[0])) + 1.0, words=()
    )


A, B, C, D = (chunk(name) for name in "ABCD")


def fused(
    first: Sequence[Chunk], second: Sequence[Chunk], **overrides: object
) -> tuple[FusedIndex, _Ranks, _Ranks]:
    one, two = _Ranks(first), _Ranks(second)
    settings = Settings(**overrides)  # type: ignore[arg-type]

    return FusedIndex((one, two), settings), one, two


def test_a_chunk_both_retrievers_rank_well_beats_one_either_ranks_first():
    """The whole point of fusing: agreement outweighs a single first place."""
    index, _, _ = fused([A, B], [C, B])

    assert index.rank("Was 100 mg prescribed?", 3)[0] is B


def test_a_chunk_only_one_retriever_found_is_still_reachable():
    """The 18 Positives BM25 cannot reach at any k are exactly this case."""
    index, _, _ = fused([A], [D])

    assert set(index.rank("Was 100 mg prescribed?", 2)) == {A, D}


def test_the_ranking_stops_at_the_limit_it_was_asked_for():
    index, _, _ = fused([A, B, C], [C, B, A])

    assert len(index.rank("Was 100 mg prescribed?", 2)) == 2


def test_chunks_that_fuse_alike_keep_the_order_the_first_retriever_ranked_them():
    index, _, _ = fused([A, B], [A, B])

    assert list(index.rank("Was 100 mg prescribed?", 2)) == [A, B]


def test_both_retrievers_are_read_to_the_fusion_depth_not_the_limit():
    """A Chunk one retriever ranks deeply is what the other is there to
    rescue, so the depth read is not the depth returned."""
    index, one, two = fused([A, B], [C, D], fusion_depth=50)

    index.rank("Was 100 mg prescribed?", 3)

    assert one.depths == [50]
    assert two.depths == [50]


def test_a_question_neither_retriever_ranks_anything_for_fuses_to_nothing():
    index, _, _ = fused([], [])

    assert index.rank("And it is, or it is not?", 3) == ()


def test_a_question_only_one_retriever_ranks_anything_for_still_fuses():
    index, _, _ = fused([], [C, D])

    assert list(index.rank("Was 100 mg prescribed?", 2)) == [C, D]


def test_a_ranking_must_hold_at_least_one_chunk():
    index, _, _ = fused([A], [A])

    with pytest.raises(ValueError):
        index.rank("Was 100 mg prescribed?", 0)


def test_fusion_needs_at_least_two_rankings_to_fuse():
    with pytest.raises(ValueError):
        FusedIndex((_Ranks([A]),), Settings())


def test_the_rank_constant_settings_resolve_decides_how_far_a_top_rank_carries():
    """At a small constant the first rank dominates; at a large one agreement
    deeper down catches up."""
    steep, _, _ = fused([A, B, B, B], [B, A, A, A], fusion_rank_constant=0.5)

    assert steep.rank("Was 100 mg prescribed?", 1)[0] is A


def test_bm25_is_the_mode_that_loads_no_embedder():
    factory = build_index_factory(Settings(retrieval_mode="bm25"))

    assert isinstance(factory([A, B]), Bm25Index)


def test_the_dense_mode_ranks_with_the_embedder_alone():
    factory = build_index_factory(Settings(retrieval_mode="dense"), embedder=_Embeds())

    assert isinstance(factory([A, B]), DenseIndex)


def test_the_hybrid_mode_fuses_both():
    factory = build_index_factory(Settings(retrieval_mode="hybrid"), embedder=_Embeds())

    assert isinstance(factory([A, B]), FusedIndex)


def test_a_mode_that_is_not_a_mode_fails_rather_than_falling_through():
    """``Settings.model_copy(update=...)`` does not validate, so a sweep that
    builds one Settings per mode can reach here with anything. A typo that
    built *some* retriever would be measured and recorded under the name that
    was typed."""
    with pytest.raises(ValueError, match="not a retrieval mode"):
        build_index_factory(
            Settings().model_copy(update={"retrieval_mode": "hybird"}), _Embeds()
        )


def test_a_ranking_deeper_than_the_fusion_depth_is_still_that_deep():
    """Fusing two top-50s cannot return 100 Chunks, so the depth read has to
    follow the limit asked for."""
    index, one, two = fused([A, B], [C, D], fusion_depth=2)

    index.rank("Was 100 mg prescribed?", 10)

    assert one.depths == [10]
    assert two.depths == [10]


def test_a_mode_that_needs_an_embedder_and_was_given_none_fails_at_startup():
    """Not at the first request, and not by quietly falling back to BM25."""
    for mode in ("dense", "hybrid"):
        with pytest.raises(ValueError, match="embedder"):
            build_index_factory(Settings(retrieval_mode=mode))  # type: ignore[arg-type]


class _Embeds:
    """An embedder that places every text on the same axis."""

    def encode(self, sentences, **options):
        return np.ones((len(list(sentences)), 2), dtype=np.float32)
