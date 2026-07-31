"""No-LLM path quotes a real review; never fabricates a template need."""

from __future__ import annotations

from src.gap_analyzer import CandidateGap
from src.llm_extractor import LatentNeedExtractor


QUOTE = (
    "Lack of a persistent stream cache leads to a potentially long pause "
    "when resuming playback on slow networks."
)


def test_representative_extract_quotes_verbatim():
    ext = LatentNeedExtractor()
    cand = CandidateGap(
        cluster_id=0,
        review_ids=["a", "b", "c", "d", "e"],
        verdict="IGNORED",
        best_similarity=0.1,
        matched_item=None,
        metrics={
            "deterministic_confidence": 70.0,
            "components": {},
            "weights": {},
            "cluster_size": 5,
            "total_reviews": 100,
            "cluster_share": 0.05,
            "best_similarity": 0.1,
            "matched_item_title": None,
            "matched_item_url": None,
            "matched_item_state": None,
            "matched_item_age_days": None,
            "mean_rating": 3.8,
            "rating_spread": 0.4,
            "cohesion": 0.7,
            "llm_confidence": None,
            "keywords": ["stream", "cache", "network"],
        },
        keywords=["stream", "cache", "network"],
        representative_text=QUOTE,
        cohesion=0.7,
        mean_rating=3.8,
        cluster_size=5,
    )
    reviews = {
        "a": {"review_id": "a", "review_text": QUOTE, "rating": 4},
        "b": {"review_id": "b", "review_text": "wish for better buffering", "rating": 4},
        "c": {"review_id": "c", "review_text": "slow networks pause playback", "rating": 3},
        "d": {"review_id": "d", "review_text": "need persistent cache please", "rating": 4},
        "e": {"review_id": "e", "review_text": "resume is laggy offline", "rating": 3},
    }
    out = ext._representative_extract(cand, reviews, "github")
    assert out.llm_used is False
    assert out.need_source == "representative_review"
    assert out.latent_need == QUOTE
    assert out.representative_review_id == "a"
    assert "Reliable control" not in out.latent_need
    assert "stream" in out.one_sentence_summary.lower() or "cache" in out.one_sentence_summary.lower()
