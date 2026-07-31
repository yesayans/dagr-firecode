"""Confidence formula must reconstruct exactly from stored metrics."""

from __future__ import annotations

from src.gap_analyzer import compute_confidence, reconstruct_confidence


def test_confidence_github_weights_reconstruct():
    conf, components, weights = compute_confidence(
        cluster_size=12,
        max_cluster_size=30,
        best_similarity=0.12,
        cohesion=0.81,
        mean_rating=1.75,
        rating_spread=0.6,
        mode="github",
    )
    assert weights == {
        "volume": 0.30,
        "novelty": 0.25,
        "consistency": 0.20,
        "severity": 0.15,
        "spread": 0.10,
    }
    metrics = {"components": components, "weights": weights}
    assert reconstruct_confidence(metrics) == conf


def test_confidence_none_weights_and_novelty():
    conf, components, weights = compute_confidence(
        cluster_size=8,
        max_cluster_size=20,
        best_similarity=None,
        cohesion=0.7,
        mean_rating=2.0,
        rating_spread=0.4,
        mode="none",
    )
    assert weights["novelty"] == 0.0
    assert components["novelty"] == 1.0
    assert reconstruct_confidence({"components": components, "weights": weights}) == conf


def test_confidence_manual_sum():
    conf, components, weights = compute_confidence(
        cluster_size=9,
        max_cluster_size=9,
        best_similarity=0.0,
        cohesion=1.0,
        mean_rating=1.0,
        rating_spread=1.0,
        mode="web",
    )
    manual = round(
        100
        * (
            0.30 * 1.0
            + 0.25 * 1.0
            + 0.20 * 1.0
            + 0.15 * 1.0
            + 0.10 * 1.0
        ),
        2,
    )
    assert conf == manual
    assert conf == 100.0
