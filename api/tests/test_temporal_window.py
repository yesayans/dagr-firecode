"""Corpus-anchored temporal rules: MISUNDERSTOOD vs historical/contemporary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.gap_analyzer import GapMatrix, ReviewWindow, compute_review_window, parse_dt


def _cluster(review_ids):
    return {
        "cluster_id": 0,
        "size": len(review_ids),
        "centroid": np.array([1.0, 0.0]),
        "cohesion": 0.85,
        "keywords": ["sleep", "timer"],
        "review_ids": review_ids,
        "mean_rating": 1.5,
        "rating_spread": 0.4,
        "representative_text": "sleep timer still broken",
    }


def _run(reviews, items, window=None):
    gm = GapMatrix(match_threshold=0.45, roadmap_matching_enabled=True)
    n = len(reviews)
    road_emb = np.array([[1.0, 0.0]] * max(len(items), 1))
    review_emb = np.tile(np.array([1.0, 0.0]), (n, 1))
    return gm.analyze(
        clusters=[_cluster(reviews["review_id"].tolist())],
        review_embeddings=review_emb,
        reviews_df=reviews,
        roadmap_items=pd.DataFrame(items),
        roadmap_embeddings=road_emb,
        roadmap_source="github",
        total_reviews=n,
        review_window=window,
    )


def test_parse_sealuzh_date_format():
    dt = parse_dt("April 03 2016")
    assert dt is not None
    assert dt.year == 2016 and dt.month == 4 and dt.day == 3


def test_review_window_from_corpus():
    reviews = pd.DataFrame(
        {
            "review_id": ["a", "b"],
            "review_text": ["x", "y"],
            "rating": [1, 2],
            "created_at": ["April 03 2016", "September 27 2016"],
        }
    )
    w = compute_review_window(reviews)
    assert w.start.date().isoformat() == "2016-04-03"
    assert w.end.date().isoformat() == "2016-09-27"


def test_misunderstood_fires_on_historical_corpus():
    """Item closed before 2016 window end; 2016 reviews still complain → MISUNDERSTOOD."""
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(5)],
            "review_text": ["sleep timer still broken"] * 5,
            "rating": [1] * 5,
            "created_at": ["2016-06-15"] * 5,
        }
    )
    items = [
        {
            "item_id": "i1",
            "text": "Fix sleep timer stopping playback",
            "state": "closed",
            "milestone_title": "1.8",
            "closed_at": "2015-06-01T00:00:00Z",
            "updated_at": "2015-06-01T00:00:00Z",
            "created_at": "2015-01-01T00:00:00Z",
            "url": "http://gh/1",
            "labels": "",
            "kind": "issue",
        }
    ]
    window = ReviewWindow(
        start=parse_dt("2016-04-01"),
        end=parse_dt("2016-09-30"),
    )
    out = _run(reviews, items, window)
    assert len(out) == 1
    assert out[0].verdict == "MISUNDERSTOOD"
    assert out[0].metrics["review_window_end"].startswith("2016-09-30")
    assert out[0].metrics["reference_date"] == out[0].metrics["review_window_end"]


def test_same_cluster_not_misunderstood_when_close_after_window():
    """
    Closed in 2019 while corpus is 2016 → as-of window the item was still open
    (or not yet closed). With milestone + fresh touch relative to window it may
    drop as covered, or UNDER-PRIORITIZED — never MISUNDERSTOOD via post-window close.
    """
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(5)],
            "review_text": ["sleep timer still broken"] * 5,
            "rating": [1] * 5,
            "created_at": ["2016-06-15"] * 5,
        }
    )
    items = [
        {
            "item_id": "i1",
            "text": "Fix sleep timer stopping playback",
            "state": "closed",
            "milestone_title": "2.0",
            "closed_at": "2019-03-01T00:00:00Z",
            "updated_at": "2019-03-01T00:00:00Z",
            "created_at": "2016-01-10T00:00:00Z",
            "url": "http://gh/1",
            "labels": "",
            "kind": "issue",
        }
    ]
    window = ReviewWindow(
        start=parse_dt("2016-04-01"),
        end=parse_dt("2016-09-30"),
    )
    out = _run(reviews, items, window)
    # As of 2016-09-30: open, milestoned, updated before window → age from created
    # created 2016-01-10 → age ~263 days < 365 → well covered → dropped
    assert all(c.verdict != "MISUNDERSTOOD" for c in out)


def test_contemporary_corpus_still_gets_misunderstood():
    """Control: modern reviews + earlier close still MISUNDERSTOOD (not only historical)."""
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(5)],
            "review_text": ["sleep timer still broken"] * 5,
            "rating": [1] * 5,
            "created_at": ["2024-08-01"] * 5,
        }
    )
    items = [
        {
            "item_id": "i1",
            "text": "Fix sleep timer stopping playback",
            "state": "closed",
            "milestone_title": "1.8",
            "closed_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "created_at": "2022-01-01T00:00:00Z",
            "url": "http://gh/1",
            "labels": "",
            "kind": "issue",
        }
    ]
    out = _run(reviews, items)
    assert len(out) == 1
    assert out[0].verdict == "MISUNDERSTOOD"


def test_future_item_does_not_cover_but_validates_later():
    """Item created after window cannot cover; surfaces as later_addressed_by."""
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(5)],
            "review_text": ["sleep timer still broken"] * 5,
            "rating": [1] * 5,
            "created_at": ["2016-06-15"] * 5,
        }
    )
    items = [
        {
            "item_id": "future",
            "text": "Rewrite sleep timer to reliably stop playback",
            "state": "closed",
            "milestone_title": "2.5",
            "closed_at": "2019-05-01T00:00:00Z",
            "updated_at": "2019-05-01T00:00:00Z",
            "created_at": "2018-11-01T00:00:00Z",
            "url": "http://gh/future",
            "labels": "",
            "kind": "issue",
        }
    ]
    window = ReviewWindow(
        start=parse_dt("2016-04-01"),
        end=parse_dt("2016-09-30"),
    )
    out = _run(reviews, items, window)
    assert len(out) == 1
    assert out[0].verdict == "IGNORED"  # no contemporaneous match
    assert out[0].metrics["validated_by_later_roadmap"] is True
    assert out[0].metrics["later_addressed_by"]["url"] == "http://gh/future"


def test_later_addressed_degrades_cleanly_when_absent():
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(5)],
            "review_text": ["unrelated theme about widgets"] * 5,
            "rating": [1] * 5,
            "created_at": ["2016-06-15"] * 5,
        }
    )
    items = [
        {
            "item_id": "x",
            "text": "totally different roadmap about dark mode",
            "state": "open",
            "milestone_title": None,
            "closed_at": None,
            "updated_at": "2016-01-01T00:00:00Z",
            "created_at": "2015-01-01T00:00:00Z",
            "url": "http://gh/x",
            "labels": "",
            "kind": "issue",
        }
    ]
    # Orthogonal embeddings → IGNORED, no later match
    gm = GapMatrix(match_threshold=0.45, roadmap_matching_enabled=True)
    out = gm.analyze(
        clusters=[_cluster(reviews["review_id"].tolist())],
        review_embeddings=np.tile(np.array([1.0, 0.0]), (5, 1)),
        reviews_df=reviews,
        roadmap_items=pd.DataFrame(items),
        roadmap_embeddings=np.array([[0.0, 1.0]]),
        roadmap_source="github",
        total_reviews=5,
        review_window=ReviewWindow(
            start=parse_dt("2016-04-01"), end=parse_dt("2016-09-30")
        ),
    )
    assert out[0].verdict == "IGNORED"
    assert out[0].metrics["later_addressed_by"] is None
    assert out[0].metrics["validated_by_later_roadmap"] is False
