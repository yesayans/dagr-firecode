"""Need-bearing selection keeps polite 4★ wants; drops empty praise."""

from __future__ import annotations

import pandas as pd

from src.embedding_engine import EmbeddingEngine
from src.gap_analyzer import GapMatrix
from src.need_filter import is_need_bearing, select_need_bearing


STREAM_CACHE = (
    "Lack of a persistent stream cache leads to a potentially long pause "
    "when resuming playback on slow networks."
)
PLAYBACK_SPEED = (
    "Faster playback would be better. It'll playback at x2 the speed which is great "
    "but I would love it to go to at least 2.5 or maybe 3?"
)
SEARCH_WANT = (
    "I love the search function and how easy it is to add new podcasts, download "
    "episodes, and play. It would be nice to search episode titles across all feeds."
)
PRAISE = "Great app, love it, thanks! Best podcast player ever."


def test_classifier_polite_wants_vs_empty_praise():
    assert is_need_bearing(STREAM_CACHE, 4.0) is True
    assert is_need_bearing(PLAYBACK_SPEED, 4.0) is True
    assert is_need_bearing(SEARCH_WANT, 4.0) is True
    assert is_need_bearing(PRAISE, 4.0) is False
    assert is_need_bearing("App crashes on open", 2.0) is True
    # "need is" must not trip want-language (was a false positive)
    assert is_need_bearing(
        "Everything you need is here in this great app. Reliable and straightforward.",
        5.0,
    ) is False
    # "without crashing" is praise, not a defect
    assert is_need_bearing(
        "Easy and reliable. Excellent app. Downloads and plays without crashing.",
        5.0,
    ) is False


def test_polite_four_star_wants_survive_and_empty_praise_does_not():
    """
    A corpus of polite 4★ want-language reviews must produce gap clusters;
    contentless praise must not survive selection to form a gap theme.
    """
    rows = []
    # Theme A: stream cache (need-bearing 4★)
    for i in range(6):
        rows.append(
            {
                "review_id": f"cache{i}",
                "review_text": STREAM_CACHE if i % 2 == 0 else (
                    "Missing a stream cache means long pauses on slow networks when I resume."
                ),
                "rating": 4.0,
                "created_at": "2016-06-01",
            }
        )
    # Theme B: playback speed (need-bearing 4★)
    for i in range(6):
        rows.append(
            {
                "review_id": f"speed{i}",
                "review_text": PLAYBACK_SPEED if i % 2 == 0 else (
                    "I wish playback speed could go to 2.5x or 3x, 2x is not enough."
                ),
                "rating": 4.0,
                "created_at": "2016-06-02",
            }
        )
    # Contentless praise — must be filtered out before clustering
    for i in range(20):
        rows.append(
            {
                "review_id": f"praise{i}",
                "review_text": PRAISE,
                "rating": 4.0,
                "created_at": "2016-06-03",
            }
        )

    full = pd.DataFrame(rows)
    need_df, stats = select_need_bearing(full)
    assert stats["reviews_total"] == 32
    assert stats["reviews_need_bearing"] == 12
    assert set(need_df["review_id"]) == {f"cache{i}" for i in range(6)} | {
        f"speed{i}" for i in range(6)
    }
    assert not any(need_df["review_id"].str.startswith("praise"))

    clustered = EmbeddingEngine().embed_and_cluster(need_df)
    assert clustered["clusters"], "expected clusters from want-language reviews"

    blobs = " ".join(
        " ".join(c.get("keywords") or [])
        + " "
        + (c.get("representative_text") or "")
        for c in clustered["clusters"]
    ).lower()
    assert any(tok in blobs for tok in ("cache", "stream", "network", "pause", "slow"))
    assert any(tok in blobs for tok in ("playback", "speed", "faster", "love"))

    gaps = GapMatrix(match_threshold=0.22).analyze(
        clusters=clustered["clusters"],
        review_embeddings=clustered["embeddings"],
        reviews_df=clustered["reviews_df"],
        roadmap_items=pd.DataFrame(),
        roadmap_embeddings=__import__("numpy").zeros((0, 1)),
        roadmap_source="none",
        total_reviews=stats["reviews_total"],
    )
    assert gaps, "polite 4-star want clusters must produce gaps"
    for g in gaps:
        assert g.metrics["need_bearing_share"] == 1.0
        assert g.verdict == "UNVERIFIED"

    # Praise-only corpus → nothing to cluster into gaps
    praise_only = full[full["review_id"].str.startswith("praise")]
    need_praise, st2 = select_need_bearing(praise_only)
    assert st2["reviews_need_bearing"] == 0
    assert need_praise.empty
