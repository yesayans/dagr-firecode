"""Descriptive statistics. No LLM anywhere in this module.

The dashboard has to render even when the model endpoint is down, so everything
here is pure Python over the review list.

One caveat drives several design choices: this corpus is a **quota-capped
scrape**. Some apps hold exactly N reviews per star level, others hold only
1-star reviews. The mean of that sample is an artefact of the scraper, not user
sentiment, so `store_score` is carried separately and the artefact is flagged.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from aipm.schemas import App, HelpfulVoteStats, OverviewStats, Review, TrendPoint

#: Below this coefficient of variation, per-star counts are suspiciously even.
_QUOTA_CV_THRESHOLD = 0.02
_MIN_REVIEWS_FOR_QUOTA_CHECK = 50


def score_distribution(reviews: Sequence[Review]) -> dict[int, int]:
    """Counts per star level, always covering 1..5 so charts have no gaps."""
    distribution = {star: 0 for star in range(1, 6)}
    for review in reviews:
        if review.score is not None and 1 <= review.score <= 5:
            distribution[int(review.score)] += 1
    return distribution


def helpful_vote_stats(reviews: Sequence[Review]) -> HelpfulVoteStats:
    """Helpful-vote distribution. Percentiles, because the mean is badly skewed."""
    votes = [max(0, r.helpful_count) for r in reviews]
    if not votes:
        return HelpfulVoteStats()
    ordered = sorted(votes)
    p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    n_with_votes = sum(1 for v in votes if v > 0)
    return HelpfulVoteStats(
        total=int(sum(votes)),
        mean=round(statistics.fmean(votes), 3),
        median=float(statistics.median(ordered)),
        p90=float(ordered[p90_index]),
        max=int(ordered[-1]),
        share_with_votes=round(n_with_votes / len(votes), 4),
    )


def looks_quota_sampled(distribution: dict[int, int]) -> bool:
    """True when the star distribution looks manufactured rather than observed.

    Two failure modes in this corpus:
    * every star level holds an identical count (a per-star quota), and
    * only one or two star levels are present at all (a filtered scrape).
    """
    counts = [c for c in distribution.values() if c > 0]
    total = sum(counts)
    if total < _MIN_REVIEWS_FOR_QUOTA_CHECK:
        return False
    if len(counts) <= 2:
        return True
    if len(counts) < 5:
        return False
    mean = statistics.fmean(counts)
    if mean <= 0:
        return False
    return (statistics.pstdev(counts) / mean) < _QUOTA_CV_THRESHOLD


def compute_overview_stats(
    reviews: Sequence[Review],
    *,
    app: App | None = None,
    n_clusters: int = 0,
    noise_ratio: float = 0.0,
    trends: Sequence[TrendPoint] | None = None,
) -> OverviewStats:
    """Assemble everything the dashboard KPI row needs."""
    if not reviews:
        return OverviewStats(store_score=app.score if app else None)

    scored = [r.score for r in reviews if r.score is not None]
    distribution = score_distribution(reviews)
    dates = sorted(r.review_date for r in reviews if r.review_date is not None)
    n_negative = sum(1 for s in scored if s <= 2)

    return OverviewStats(
        n_reviews=len(reviews),
        avg_score=round(statistics.fmean(scored), 3) if scored else 0.0,
        score_distribution=distribution,
        pct_negative=round(n_negative / len(scored), 4) if scored else 0.0,
        trend_delta_90d=trend_delta(trends or []),
        n_clusters=n_clusters,
        noise_ratio=round(noise_ratio, 4),
        date_range=(dates[0], dates[-1]) if dates else None,
        helpful_votes=helpful_vote_stats(reviews),
        store_score=app.score if app else None,
        sample_is_quota_capped=looks_quota_sampled(distribution),
        n_star_levels=sum(1 for c in distribution.values() if c > 0),
    )


def trend_delta(trends: Sequence[TrendPoint], *, months: int = 3) -> float:
    """Change in average score over the last `months` periods vs the preceding ones.

    Reported in stars. Positive means sentiment improved.
    """
    if len(trends) < months * 2:
        return 0.0
    recent = [t.avg_score for t in trends[-months:] if t.n_reviews > 0]
    prior = [t.avg_score for t in trends[-months * 2 : -months] if t.n_reviews > 0]
    if not recent or not prior:
        return 0.0
    return round(statistics.fmean(recent) - statistics.fmean(prior), 3)


def affected_rating_gap(
    reviews_in_need: Sequence[Review], *, baseline: float
) -> float:
    """`baseline - mean(score of affected reviews)`, the impact term.

    `baseline` should be the app's **store** rating, not the sampled mean: with a
    quota-capped corpus the sampled mean is fixed near 3.0 and would make impact
    meaningless.
    """
    scored = [r.score for r in reviews_in_need if r.score is not None]
    if not scored or baseline is None:
        return 0.0
    return round(max(0.0, baseline - statistics.fmean(scored)), 3)
