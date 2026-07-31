"""Review volume and rating over time.

Months, not days: daily review counts for a single app are too sparse to read.
Gap months are emitted explicitly with `n_reviews=0` so the chart shows a real
gap rather than interpolating across a period with no data.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from datetime import date

from aipm.schemas import Review, TrendPoint

DEFAULT_ROLLING_WINDOW = 3


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _next_month(value: date) -> date:
    return date(value.year + 1, 1, 1) if value.month == 12 else date(value.year, value.month + 1, 1)


def month_range(start: date, end: date) -> list[date]:
    months: list[date] = []
    cursor = _month_start(start)
    last = _month_start(end)
    while cursor <= last:
        months.append(cursor)
        cursor = _next_month(cursor)
    return months


def compute_trends(
    reviews: Sequence[Review],
    *,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    max_months: int | None = None,
) -> list[TrendPoint]:
    """Monthly volume and average score, with a trailing rolling average.

    `max_months` trims to the most recent N months; a 15-year tail of three
    reviews a month is noise on a dashboard.
    """
    dated = [(r.review_date, r.score) for r in reviews if r.review_date is not None]
    if not dated:
        return []

    buckets: dict[date, list[int]] = {}
    for review_date, score in dated:
        buckets.setdefault(_month_start(review_date), []).append(
            score if score is not None else 0
        )

    months = month_range(min(buckets), max(buckets))
    if max_months is not None and len(months) > max_months:
        months = months[-max_months:]

    points: list[TrendPoint] = []
    for month in months:
        scores = [s for s in buckets.get(month, []) if s > 0]
        points.append(
            TrendPoint(
                period=month,
                n_reviews=len(buckets.get(month, [])),
                avg_score=round(statistics.fmean(scores), 3) if scores else 0.0,
            )
        )
    return _apply_rolling_average(points, rolling_window)


def _apply_rolling_average(points: list[TrendPoint], window: int) -> list[TrendPoint]:
    """Trailing mean over months that actually had reviews."""
    if window <= 1:
        return points
    out: list[TrendPoint] = []
    for i, point in enumerate(points):
        window_slice = points[max(0, i - window + 1) : i + 1]
        scores = [p.avg_score for p in window_slice if p.n_reviews > 0]
        out.append(
            point.model_copy(
                update={"rolling_avg": round(statistics.fmean(scores), 3) if scores else None}
            )
        )
    return out


def months_covered(reviews: Sequence[Review]) -> int:
    """Distinct months containing at least one review. Feeds the temporal score."""
    return len({_month_start(r.review_date) for r in reviews if r.review_date is not None})


def temporal_spread(
    reviews_in_cluster: Sequence[Review], all_reviews: Sequence[Review]
) -> float:
    """Fraction of the corpus's active months in which this theme appears.

    A theme spread across most months is a standing problem; one confined to a
    single month is probably a bad release or a review-bombing incident.
    """
    corpus_months = months_covered(all_reviews)
    if corpus_months == 0:
        return 0.0
    return min(1.0, months_covered(reviews_in_cluster) / corpus_months)
