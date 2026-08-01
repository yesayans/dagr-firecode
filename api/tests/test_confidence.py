"""Confidence formula must reconstruct exactly from stored metrics."""

from __future__ import annotations

import pytest

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


def test_missing_ratings_drop_severity_and_spread_rather_than_inventing_them():
    """A source with no ratings must not get a fabricated neutral 3.0.

    Defaulting unknown ratings to 3.0 pinned severity at 0.50 and spread at
    0.00 for every cluster. That is 35% of the weight frozen into a constant,
    which left confidence a pure function of cluster size — the biggest cluster
    always won, regardless of how angry its reviews were.
    """
    conf, components, weights = compute_confidence(
        cluster_size=20,
        max_cluster_size=60,
        best_similarity=None,
        cohesion=0.7,
        mean_rating=None,
        rating_spread=None,
        mode="none",
    )

    assert "severity" not in components
    assert "spread" not in components
    assert set(weights) == set(components)
    # Renormalised, so the score still spans the full 0-100 range.
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= conf <= 100.0


def test_weights_always_match_components_so_reconstruction_stays_exact():
    for mean_rating, spread in ((3.4, 0.6), (None, None)):
        conf, components, weights = compute_confidence(
            cluster_size=12,
            max_cluster_size=40,
            best_similarity=None,
            cohesion=0.5,
            mean_rating=mean_rating,
            rating_spread=spread,
            mode="none",
        )
        assert set(weights) == set(components)
        assert reconstruct_confidence(
            {"components": components, "weights": weights}
        ) == pytest.approx(conf)


def test_ratings_still_discriminate_when_present():
    """With real ratings, a small angry cluster outranks a large mild one."""
    angry, _, _ = compute_confidence(
        cluster_size=5, max_cluster_size=60, best_similarity=None,
        cohesion=0.7, mean_rating=1.4, rating_spread=0.8, mode="none",
    )
    mild, _, _ = compute_confidence(
        cluster_size=60, max_cluster_size=60, best_similarity=None,
        cohesion=0.7, mean_rating=4.2, rating_spread=0.3, mode="none",
    )
    assert angry > mild
