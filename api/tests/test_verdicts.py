"""Verdict rules including MISUNDERSTOOD."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.gap_analyzer import GapMatrix


def _cluster(review_ids, cohesion=0.8, mean_rating=1.5, size=None):
    size = size or len(review_ids)
    return {
        "cluster_id": 0,
        "size": size,
        "centroid": np.array([1.0, 0.0]),
        "cohesion": cohesion,
        "keywords": ["sleep", "timer"],
        "review_ids": review_ids,
        "mean_rating": mean_rating,
        "rating_spread": 0.4,
        "representative_text": "sleep timer broken",
    }


def test_ignored_below_threshold():
    gm = GapMatrix(match_threshold=0.45)
    reviews = pd.DataFrame(
        {
            "review_id": ["a1", "a2", "a3", "a4", "a5"],
            "review_text": ["x"] * 5,
            "rating": [1] * 5,
            "created_at": ["2024-06-01"] * 5,
        }
    )
    items = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "text": "unrelated roadmap item about dark mode theme",
                "state": "open",
                "milestone_title": "v3",
                "age_days": 10,
                "closed_at": None,
                "url": "http://x",
                "updated_at": "2025-01-01T00:00:00Z",
                "labels": "",
                "kind": "issue",
            }
        ]
    )
    # Centroid [1,0], roadmap emb nearly orthogonal → low sim
    road_emb = np.array([[0.0, 1.0]])
    out = gm.analyze(
        clusters=[_cluster(["a1", "a2", "a3", "a4", "a5"])],
        review_embeddings=np.zeros((5, 2)),
        reviews_df=reviews,
        roadmap_items=items,
        roadmap_embeddings=road_emb,
        roadmap_source="github",
        total_reviews=5,
    )
    assert len(out) == 1
    assert out[0].verdict == "IGNORED"


def test_misunderstood_closed_item_recent_reviews():
    gm = GapMatrix(match_threshold=0.45)
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(5)],
            "review_text": ["sleep timer still broken"] * 5,
            "rating": [1] * 5,
            "created_at": ["2024-08-01"] * 5,
        }
    )
    items = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "text": "Fix sleep timer stopping playback",
                "state": "closed",
                "milestone_title": "1.8",
                "age_days": 10,
                "closed_at": "2023-01-01T00:00:00Z",
                "url": "http://gh/1",
                "updated_at": "2023-01-01T00:00:00Z",
                "labels": "",
                "kind": "issue",
            }
        ]
    )
    road_emb = np.array([[1.0, 0.0]])  # identical to centroid
    out = gm.analyze(
        clusters=[_cluster([f"r{i}" for i in range(5)])],
        review_embeddings=np.zeros((5, 2)),
        reviews_df=reviews,
        roadmap_items=items,
        roadmap_embeddings=road_emb,
        roadmap_source="github",
        total_reviews=5,
    )
    assert len(out) == 1
    assert out[0].verdict == "MISUNDERSTOOD"


def test_under_prioritized_stale_or_no_milestone():
    gm = GapMatrix(match_threshold=0.45)
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(5)],
            "review_text": ["download queue stuck"] * 5,
            "rating": [2] * 5,
            "created_at": ["2024-08-01"] * 5,
        }
    )
    items = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "text": "Improve download queue reliability",
                "state": "open",
                "milestone_title": None,
                "age_days": 20,
                "closed_at": None,
                "url": "http://gh/2",
                "updated_at": "2025-01-01T00:00:00Z",
                "labels": "",
                "kind": "issue",
            }
        ]
    )
    road_emb = np.array([[1.0, 0.0]])
    out = gm.analyze(
        clusters=[_cluster([f"r{i}" for i in range(5)])],
        review_embeddings=np.zeros((5, 2)),
        reviews_df=reviews,
        roadmap_items=items,
        roadmap_embeddings=road_emb,
        roadmap_source="github",
        total_reviews=5,
    )
    assert out[0].verdict == "UNDER-PRIORITIZED"


def test_well_covered_dropped():
    gm = GapMatrix(match_threshold=0.45)
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(5)],
            "review_text": ["want chapters"] * 5,
            "rating": [3] * 5,
            "created_at": ["2024-08-01"] * 5,
        }
    )
    items = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "text": "Chapter support improvements",
                "state": "open",
                "milestone_title": "v3.2",
                "age_days": 30,
                "closed_at": None,
                "url": "http://gh/3",
                "updated_at": "2025-06-01T00:00:00Z",
                "labels": "",
                "kind": "issue",
            }
        ]
    )
    road_emb = np.array([[1.0, 0.0]])
    out = gm.analyze(
        clusters=[_cluster([f"r{i}" for i in range(5)])],
        review_embeddings=np.zeros((5, 2)),
        reviews_df=reviews,
        roadmap_items=items,
        roadmap_embeddings=road_emb,
        roadmap_source="github",
        total_reviews=5,
    )
    assert out == []


def test_none_mode_unverified():
    gm = GapMatrix(match_threshold=0.45)
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(5)],
            "review_text": ["anything"] * 5,
            "rating": [2] * 5,
            "created_at": ["2024-08-01"] * 5,
        }
    )
    out = gm.analyze(
        clusters=[_cluster([f"r{i}" for i in range(5)])],
        review_embeddings=np.zeros((5, 2)),
        reviews_df=reviews,
        roadmap_items=pd.DataFrame(),
        roadmap_embeddings=np.zeros((0, 2)),
        roadmap_source="none",
        total_reviews=5,
    )
    assert len(out) == 1
    assert out[0].verdict == "UNVERIFIED"
