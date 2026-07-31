"""Clustering: reduction, density clustering with fallback, keywords, metrics."""

from __future__ import annotations

import numpy as np
import pytest

from aipm.clustering.cluster import (
    dominant_share,
    NOISE_LABEL,
    ClusteringConfig,
    cluster_embeddings,
)
from aipm.clustering.keywords import extract_cluster_keywords
from aipm.clustering.metrics import centroid, cohesion, medoid_index, separation
from aipm.clustering.reduce import ReductionConfig, reduce_embeddings
from aipm.clustering.representatives import select_representatives


def blobs(n_per: int = 40, dim: int = 16, seed: int = 0) -> np.ndarray:
    """Three well-separated, L2-normalised clusters."""
    rng = np.random.default_rng(seed)
    centres = np.eye(3, dim) * 5.0
    points = np.vstack([c + rng.standard_normal((n_per, dim)) * 0.25 for c in centres])
    return (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)


class TestReduce:
    def test_output_shapes(self):
        result = reduce_embeddings(blobs(), ReductionConfig(n_components=5))
        assert result.embedding.shape[0] == 120
        assert result.projection_2d.shape == (120, 2)

    def test_n_components_capped_below_sample_count(self):
        result = reduce_embeddings(blobs(n_per=2), ReductionConfig(n_components=32))
        assert result.embedding.shape[1] <= 6

    def test_deterministic_for_a_fixed_seed(self):
        cfg = ReductionConfig(n_components=4, random_state=7)
        data = blobs()
        np.testing.assert_allclose(
            reduce_embeddings(data, cfg).embedding,
            reduce_embeddings(data, cfg).embedding,
            rtol=1e-4, atol=1e-4,
        )

    def test_empty_input(self):
        assert reduce_embeddings(np.zeros((0, 8), dtype=np.float32)).embedding.shape[0] == 0


class TestCluster:
    def test_recovers_separated_groups(self):
        result = cluster_embeddings(blobs(), ClusteringConfig(min_cluster_size=10))
        assert result.n_clusters >= 3

    def test_falls_back_when_hdbscan_collapses(self):
        """Uniform noise has no density structure; KMeans must take over."""
        rng = np.random.default_rng(3)
        data = rng.standard_normal((80, 6)).astype(np.float32)
        result = cluster_embeddings(
            data, ClusteringConfig(min_cluster_size=40, min_clusters_before_fallback=3)
        )
        assert result.fallback and result.method == "kmeans"
        assert result.n_clusters >= 3

    def test_tiny_input_does_not_crash(self):
        result = cluster_embeddings(np.eye(3, 4, dtype=np.float32))
        assert result.n_clusters >= 1

    def test_empty_input(self):
        assert cluster_embeddings(np.zeros((0, 4), dtype=np.float32)).n_clusters == 0

    def test_noise_excluded_from_cluster_indices(self):
        result = cluster_embeddings(blobs(), ClusteringConfig(min_cluster_size=10))
        assert NOISE_LABEL not in result.cluster_indices()

    def test_probabilities_align_with_labels(self):
        result = cluster_embeddings(blobs(), ClusteringConfig(min_cluster_size=10))
        assert len(result.probabilities) == len(result.labels)

    @pytest.mark.parametrize("n_units,expected_floor", [(100, 15), (10000, 100)])
    def test_min_cluster_size_scales_with_corpus(self, n_units, expected_floor):
        cfg = ClusteringConfig(min_cluster_size_floor=15, min_cluster_size_ratio=0.01)
        assert cfg.resolve_min_cluster_size(n_units) == expected_floor


class TestMetrics:
    def test_cohesion_higher_for_tight_group(self):
        rng = np.random.default_rng(0)
        tight = np.tile(np.eye(1, 8), (20, 1)) + rng.standard_normal((20, 8)) * 0.01
        tight /= np.linalg.norm(tight, axis=1, keepdims=True)
        loose = rng.standard_normal((20, 8))
        loose /= np.linalg.norm(loose, axis=1, keepdims=True)
        assert cohesion(tight.astype(np.float32)) > cohesion(loose.astype(np.float32))

    def test_cohesion_of_singleton_is_zero(self):
        assert cohesion(np.eye(1, 4, dtype=np.float32)) == 0.0

    def test_separation_bounds(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert separation(a, [b]) == pytest.approx(1.0)
        assert separation(a, [a]) == pytest.approx(0.0)

    def test_lone_cluster_is_maximally_separated(self):
        assert separation(np.array([1.0, 0.0], dtype=np.float32), []) == 1.0

    def test_medoid_is_a_member_index(self):
        data = blobs(n_per=10)[:10]
        assert 0 <= medoid_index(data) < 10

    def test_centroid_is_unit_length(self):
        assert np.linalg.norm(centroid(blobs()[:20])) == pytest.approx(1.0, rel=1e-5)


class TestKeywords:
    def test_extracts_distinguishing_terms(self):
        texts = (
            ["the app crashes on login every time"] * 5
            + ["delivery driver could not find my address"] * 5
        )
        result = extract_cluster_keywords(texts, {0: list(range(5)), 1: list(range(5, 10))})
        assert any("crash" in k for k in result[0])
        assert any("driver" in k or "address" in k for k in result[1])

    def test_handles_stopword_only_text(self):
        result = extract_cluster_keywords(["the and of"] * 3, {0: [0, 1, 2]})
        assert result[0] == []

    def test_empty_input(self):
        assert extract_cluster_keywords([], {}) == {}


class TestRepresentatives:
    def test_returns_all_when_cluster_is_small(self):
        assert select_representatives(blobs(n_per=1)[:3], n=10) == [0, 1, 2]

    def test_respects_requested_count(self):
        assert len(select_representatives(blobs(), n=8)) == 8

    def test_first_pick_is_the_medoid(self):
        data = blobs(n_per=20)[:20]
        assert select_representatives(data, n=5)[0] == medoid_index(data)

    def test_selection_is_diverse_not_just_nearest(self):
        """Pure relevance ranking would return 5 near-identical points."""
        data = blobs()
        picks = select_representatives(data, n=6, diversity=0.6)
        assert len({p // 40 for p in picks}) >= 2

    def test_weights_length_validated(self):
        with pytest.raises(ValueError):
            select_representatives(blobs(), n=5, weights=[1.0, 2.0])

    def test_helpful_votes_do_not_break_selection(self):
        data = blobs()
        weights = [10_000.0] + [0.0] * (len(data) - 1)
        assert len(select_representatives(data, n=5, weights=weights)) == 5

    def test_empty_input(self):
        assert select_representatives(np.zeros((0, 4), dtype=np.float32)) == []


class TestDominantClusterHandling:
    def test_dominant_share_computed_over_clustered_units_only(self):
        labels = np.array([0, 0, 0, 0, 1, NOISE_LABEL, NOISE_LABEL])
        assert dominant_share(labels) == pytest.approx(0.8)

    def test_no_clustered_units(self):
        assert dominant_share(np.array([NOISE_LABEL, NOISE_LABEL])) == 0.0

    @staticmethod
    def _nested_density() -> np.ndarray:
        """Five sub-themes packed inside one density region, plus two distant ones.

        Excess-of-mass merges the five into a single cluster holding ~82% of the
        points - the exact failure seen on real review corpora. Leaf selection
        recovers them.
        """
        rng = np.random.default_rng(5)
        near = np.vstack([
            np.array([i * 1.5, 0, 0, 0, 0, 0]) + rng.standard_normal((150, 6)) * 0.25
            for i in range(5)
        ])
        far = np.vstack([
            np.array([40.0, i * 12, 0, 0, 0, 0]) + rng.standard_normal((80, 6)) * 0.3
            for i in range(2)
        ])
        return np.vstack([near, far]).astype(np.float32)

    def test_mega_cluster_is_broken_up(self):
        data = self._nested_density()
        permissive = cluster_embeddings(
            data, ClusteringConfig(min_cluster_size=25, max_dominant_cluster_share=1.0)
        )
        guarded = cluster_embeddings(
            data, ClusteringConfig(min_cluster_size=25, max_dominant_cluster_share=0.5)
        )
        assert dominant_share(permissive.labels) > 0.5
        assert dominant_share(guarded.labels) <= 0.5
        assert guarded.n_clusters > permissive.n_clusters

    def test_leaf_selection_is_recorded_in_the_method(self):
        """The UI needs to know which selection actually produced the clusters."""
        result = cluster_embeddings(
            self._nested_density(),
            ClusteringConfig(min_cluster_size=25, max_dominant_cluster_share=0.5),
        )
        assert result.method == "hdbscan-leaf"

    def test_healthy_clustering_is_left_alone(self):
        result = cluster_embeddings(
            blobs(), ClusteringConfig(min_cluster_size=10, max_dominant_cluster_share=0.5)
        )
        assert result.method == "hdbscan-eom"
