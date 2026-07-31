"""Aggregate review time-series and rating charts for job stats."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from src.gap_analyzer import parse_dt


def build_review_charts(
    reviews_df: pd.DataFrame | None,
    *,
    reviews_need_bearing: int | None = None,
) -> dict[str, Any]:
    """Return compact chart payloads safe to embed in Job.stats.

    Reviews over time are always year buckets (with zero-filled gaps).
    """
    empty: dict[str, Any] = {
        "reviews_by_period": [],
        "period": "year",
        "rating_histogram": [],
        "need_bearing": {"need_bearing": 0, "other": 0},
    }
    if reviews_df is None or reviews_df.empty:
        return empty

    n = len(reviews_df)
    need_n = (
        int(reviews_need_bearing)
        if reviews_need_bearing is not None
        else 0
    )
    need_n = max(0, min(need_n, n))

    rating_counts: Counter[int] = Counter()
    if "rating" in reviews_df.columns:
        for v in reviews_df["rating"].tolist():
            try:
                r = int(round(float(v)))
            except (TypeError, ValueError):
                continue
            if 1 <= r <= 5:
                rating_counts[r] += 1

    rating_histogram = [
        {"stars": s, "count": int(rating_counts.get(s, 0))} for s in range(1, 6)
    ]

    year_counts: Counter[int] = Counter()
    if "created_at" in reviews_df.columns:
        for v in reviews_df["created_at"].tolist():
            dt = parse_dt(v)
            if dt is None:
                continue
            year_counts[dt.year] += 1

    reviews_by_period: list[dict[str, Any]] = []
    if year_counts:
        y0, y1 = min(year_counts), max(year_counts)
        reviews_by_period = [
            {"period": f"{y:04d}", "count": int(year_counts.get(y, 0))}
            for y in range(y0, y1 + 1)
        ]

    return {
        "reviews_by_period": reviews_by_period,
        "period": "year",
        "rating_histogram": rating_histogram,
        "need_bearing": {
            "need_bearing": need_n,
            "other": max(0, n - need_n),
        },
    }
