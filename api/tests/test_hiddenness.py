"""Hiddenness / insight ranking (ported from alternative aipm)."""

from src.hiddenness import (
    annotate_metrics_hiddenness,
    count_explicit_requests,
    hiddenness,
    insight_score,
    refresh_insight_score,
)


def test_count_explicit_requests_markers():
    texts = [
        "Please add dark mode",
        "App crashes on sync",
        "Would be nice if offline worked",
    ]
    assert count_explicit_requests(texts) == 2


def test_hiddenness_all_symptoms():
    assert hiddenness(0, 10) == 1.0


def test_hiddenness_all_explicit():
    assert hiddenness(5, 5) == 0.0


def test_hiddenness_mixed_and_cross_cluster_boost():
    base = hiddenness(2, 10)
    boosted = hiddenness(2, 10, cross_cluster=True)
    assert base == 0.8
    assert boosted > base
    assert boosted <= 1.0


def test_insight_score_product():
    assert insight_score(0.5, 80.0) == 0.4


def test_annotate_and_refresh_metrics():
    metrics = {"deterministic_confidence": 80.0}
    annotate_metrics_hiddenness(
        metrics,
        [
            "Please add folders",
            "Sync fails every night",
            "I use the website instead",
        ],
    )
    assert metrics["explicit_request_count"] == 1
    assert metrics["mention_count"] == 3
    assert metrics["hiddenness"] == round(1 - 1 / 3, 4)
    assert metrics["insight_score"] == insight_score(metrics["hiddenness"], 80.0)

    refresh_insight_score(metrics, 90.0)
    assert metrics["insight_score"] == insight_score(metrics["hiddenness"], 90.0)


def test_empty_texts_zero_hiddenness():
    metrics = {"deterministic_confidence": 50.0}
    annotate_metrics_hiddenness(metrics, [])
    assert metrics["hiddenness"] == 0.0
    assert metrics["insight_score"] == 0.0
