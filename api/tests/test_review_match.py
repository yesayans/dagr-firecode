"""Review-level aggregation: agreement beats a single loud cosine."""

from __future__ import annotations

import numpy as np

from src.review_match import aggregate_cluster_match, min_agreement


def test_min_agreement_floor():
    assert min_agreement(1) == 1
    assert min_agreement(3) == 2
    assert min_agreement(10) == 3  # 30% of 10


def test_aggregate_requires_agreement():
    # 5 members; 3 vote item 0, 2 vote item 1 — agreement clears floor
    road = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ]
    )
    # normalize last
    road[2] = road[2] / np.linalg.norm(road[2])
    members = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    agg = aggregate_cluster_match(members, road, [0, 1, 2], min_count=2, min_rate=0.3)
    assert agg is not None
    assert agg.item_index == 0
    assert agg.n_agree == 3
    assert agg.score > 0.9


def test_aggregate_rejects_split_vote():
    road = np.array([[1.0, 0.0], [0.0, 1.0]])
    # 2 vs 2 — need max(2, ceil(0.3*4))=2, so 2 agrees... actually both have 2.
    # most_common picks first with count 2. Still passes min_count=2.
    # Use 10 members with 2 agreeing on winner max — 2 < ceil(0.3*10)=3
    members = np.vstack(
        [
            np.tile([1.0, 0.0], (2, 1)),
            np.tile([0.0, 1.0], (8, 1)),
        ]
    )
    # Winner is item 1 with 8 votes — passes
    agg = aggregate_cluster_match(members, road, [0, 1], min_count=2, min_rate=0.3)
    assert agg is not None
    assert agg.item_index == 1

    # Force failure: 10 members, 2 and 2 and rest orthogonal to both? 
    # Simpler: only 2 members disagreeing
    members = np.array([[1.0, 0.0], [0.0, 1.0]])
    agg = aggregate_cluster_match(members, road, [0, 1], min_count=2, min_rate=0.3)
    # each gets 1 vote; need max(2, ceil(0.6))=2 → fail
    assert agg is None
