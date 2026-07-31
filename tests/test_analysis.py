"""Analysis: statistics, trends, and the confidence model."""

from __future__ import annotations

from datetime import date

import pytest

from aipm.analysis.confidence import (
    ConfidenceInputs,
    compute_confidence,
    count_explicit_requests,
    diversity_score,
    grounding_score,
    hiddenness,
    support_score,
)
from aipm.analysis.stats import (
    compute_overview_stats,
    helpful_vote_stats,
    looks_quota_sampled,
    score_distribution,
    trend_delta,
)
from aipm.analysis.trends import compute_trends, months_covered, temporal_spread
from aipm.config import Settings
from aipm.schemas import App, Review

WEIGHTS = Settings().confidence_weights()


def review(score=3, day=1, month=1, helpful=0, text="the app crashes on login") -> Review:
    return Review(
        review_id=f"r{month}_{day}_{score}_{helpful}",
        app_id="a1",
        text=text,
        score=score,
        review_date=date(2024, month, day),
        helpful_count=helpful,
    )


class TestScoreStats:
    def test_distribution_covers_all_star_levels(self):
        assert set(score_distribution([review(score=5)]).keys()) == {1, 2, 3, 4, 5}

    def test_avg_and_negative_share(self):
        stats = compute_overview_stats([review(score=1), review(score=1), review(score=5)])
        assert stats.avg_score == pytest.approx(2.333, abs=1e-3)
        assert stats.pct_negative == pytest.approx(2 / 3, abs=1e-3)

    def test_store_score_carried_separately_from_sample_mean(self):
        """The scraped sample mean is an artefact; the store rating is the truth."""
        app = App(app_id="a1", name="A", score=4.7)
        stats = compute_overview_stats([review(score=1)] * 3, app=app)
        assert stats.store_score == 4.7
        assert stats.avg_score == 1.0

    def test_empty_input(self):
        assert compute_overview_stats([]).n_reviews == 0


class TestQuotaDetection:
    def test_flat_distribution_flagged(self):
        assert looks_quota_sampled({1: 3000, 2: 3000, 3: 3000, 4: 3000, 5: 3000})

    def test_single_star_level_flagged(self):
        assert looks_quota_sampled({1: 1620, 2: 0, 3: 0, 4: 0, 5: 0})

    def test_natural_distribution_not_flagged(self):
        assert not looks_quota_sampled({1: 800, 2: 200, 3: 150, 4: 400, 5: 1200})

    def test_small_samples_not_flagged(self):
        assert not looks_quota_sampled({1: 2, 2: 2, 3: 2, 4: 2, 5: 2})


class TestHelpfulVotes:
    def test_percentiles_reported(self):
        reviews = [review(helpful=h) for h in [0, 0, 0, 1, 2, 5, 100]]
        stats = helpful_vote_stats(reviews)
        assert stats.total == 108
        assert stats.max == 100
        assert stats.median == 1
        assert stats.share_with_votes == pytest.approx(4 / 7, abs=1e-3)

    def test_empty(self):
        assert helpful_vote_stats([]).total == 0


class TestTrends:
    def test_one_point_per_month(self):
        reviews = [review(month=m) for m in range(1, 7)]
        assert len(compute_trends(reviews)) == 6

    def test_gap_months_emitted_as_zero(self):
        """A gap must render as a gap, not interpolate across missing data."""
        points = compute_trends([review(month=1), review(month=4)])
        assert len(points) == 4
        assert points[1].n_reviews == 0

    def test_rolling_average_present(self):
        points = compute_trends([review(month=m) for m in range(1, 7)], rolling_window=3)
        assert points[-1].rolling_avg is not None

    def test_max_months_trims_to_recent(self):
        reviews = [review(month=m) for m in range(1, 13)]
        assert len(compute_trends(reviews, max_months=4)) == 4

    def test_undated_reviews_ignored(self):
        r = review()
        r.review_date = None
        assert compute_trends([r]) == []

    def test_months_covered(self):
        assert months_covered([review(month=1), review(month=1, day=2), review(month=3)]) == 2

    def test_temporal_spread_bounds(self):
        corpus = [review(month=m) for m in range(1, 5)]
        assert temporal_spread(corpus, corpus) == 1.0
        assert temporal_spread([corpus[0]], corpus) == pytest.approx(0.25)

    def test_trend_delta_sign(self):
        improving = compute_trends(
            [review(score=1, month=m) for m in range(1, 4)]
            + [review(score=5, month=m) for m in range(4, 7)]
        )
        assert trend_delta(improving) > 0


class TestConfidenceComponents:
    def test_support_is_log_scaled(self):
        """Doubling from 20 to 40 must matter more than 2000 to 2020."""
        low = support_score(40, 5000) - support_score(20, 5000)
        high = support_score(2020, 5000) - support_score(2000, 5000)
        assert low > high

    def test_support_bounds(self):
        assert support_score(0, 100) == 0.0
        assert support_score(100, 100) == pytest.approx(1.0)

    def test_grounding_zero_when_nothing_cited(self):
        """An unevidenced need is exactly what this component exists to catch."""
        assert grounding_score(0, 0) == 0.0

    def test_grounding_is_survival_rate(self):
        assert grounding_score(4, 3) == 0.75

    def test_diversity_penalises_duplicates(self):
        assert diversity_score(0.8) == pytest.approx(0.2)


class TestCompositeConfidence:
    def _inputs(self, **overrides) -> ConfidenceInputs:
        base = dict(
            n_units=100, n_units_max=200, cohesion=0.7, separation=0.6, temporal=0.8,
            duplicate_share=0.05, n_citations_offered=4, n_citations_validated=4,
            n_months_present=9,
        )
        base.update(overrides)
        return ConfidenceInputs(**base)

    def test_strong_evidence_scores_high(self):
        assert compute_confidence(self._inputs(), WEIGHTS).total >= 0.7

    def test_weak_evidence_scores_low(self):
        weak = self._inputs(
            n_units=3, cohesion=0.1, separation=0.1, temporal=0.1,
            duplicate_share=0.9, n_citations_offered=4, n_citations_validated=0,
        )
        assert compute_confidence(weak, WEIGHTS).total < 0.45

    def test_total_is_bounded(self):
        assert 0.0 <= compute_confidence(self._inputs(), WEIGHTS).total <= 1.0

    def test_band_matches_total(self):
        assert compute_confidence(self._inputs(), WEIGHTS).band == "high"

    def test_explanation_mentions_volume_and_citations(self):
        explanation = compute_confidence(self._inputs(), WEIGHTS).explanation
        assert "100 supporting review segments" in explanation
        assert "4 of 4 citations verified" in explanation

    def test_explanation_names_the_weakest_component(self):
        explanation = compute_confidence(self._inputs(cohesion=0.05), WEIGHTS).explanation
        assert "Weakest signal: cohesion" in explanation

    def test_explanation_discloses_duplicate_evidence(self):
        explanation = compute_confidence(self._inputs(duplicate_share=0.5), WEIGHTS).explanation
        assert "near-duplicates" in explanation

    def test_llm_cannot_influence_the_number(self):
        """Confidence depends only on measured inputs - there is no model term."""
        assert compute_confidence(self._inputs(), WEIGHTS).total == (
            compute_confidence(self._inputs(), WEIGHTS).total
        )


class TestHiddenness:
    def test_explicit_requests_reduce_hiddenness(self):
        assert hiddenness(8, 10) < hiddenness(1, 10)

    def test_cross_cluster_boost(self):
        assert hiddenness(5, 10, cross_cluster=True) > hiddenness(5, 10)

    def test_bounds(self):
        assert hiddenness(0, 0) == 0.0
        assert 0.0 <= hiddenness(10, 10) <= 1.0

    def test_counts_request_phrases(self):
        texts = ["please add dark mode", "it crashes", "I wish there was a way to export"]
        assert count_explicit_requests(texts) == 2
