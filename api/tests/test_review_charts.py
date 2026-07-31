"""Review chart aggregation."""

from __future__ import annotations

import pandas as pd

from src.review_charts import build_review_charts


def test_build_review_charts_counts_years_and_ratings():
    df = pd.DataFrame(
        [
            {"review_id": "a", "rating": 1, "created_at": "2014-03-01"},
            {"review_id": "b", "rating": 5, "created_at": "2015-06-15"},
            {"review_id": "c", "rating": 5, "created_at": "2015-11-01"},
            {"review_id": "d", "rating": 3, "created_at": "2016-01-20"},
        ]
    )
    charts = build_review_charts(df, reviews_need_bearing=2)
    assert charts["period"] == "year"
    assert [p["period"] for p in charts["reviews_by_period"]] == [
        "2014",
        "2015",
        "2016",
    ]
    assert [p["count"] for p in charts["reviews_by_period"]] == [1, 2, 1]
    stars = {r["stars"]: r["count"] for r in charts["rating_histogram"]}
    assert stars[5] == 2
    assert stars[1] == 1
    assert charts["need_bearing"] == {"need_bearing": 2, "other": 2}


def test_build_review_charts_fills_missing_years():
    df = pd.DataFrame(
        [
            {"review_id": "a", "rating": 2, "created_at": "2013-01-01"},
            {"review_id": "b", "rating": 4, "created_at": "2016-06-01"},
        ]
    )
    charts = build_review_charts(df)
    assert charts["period"] == "year"
    assert charts["reviews_by_period"] == [
        {"period": "2013", "count": 1},
        {"period": "2014", "count": 0},
        {"period": "2015", "count": 0},
        {"period": "2016", "count": 1},
    ]


def test_build_review_charts_empty():
    charts = build_review_charts(None)
    assert charts["reviews_by_period"] == []
    assert charts["period"] == "year"
    assert charts["rating_histogram"] == []
