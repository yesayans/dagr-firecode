"""Density clustering with an automatic KMeans fallback.

HDBSCAN is right for this data because the number of themes is unknown and a lot
of review text is genuinely noise. But it can collapse - returning one cluster or
labelling everything -1 - and a demo that shows nothing is worse than a demo that
shows approximate themes. So collapse is detected and KMeans takes over, with the
substitution recorded so the UI can say so out loud.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aipm.utils.logging import get_logger

log = get_logger(__name__)

NOISE_LABEL = -1


@dataclass(frozen=True)
class ClusteringConfig:
    min_cluster_size: int | None = None  # None -> derived from corpus size
    min_samples: int | None = None
    min_cluster_size_floor: int = 15
    min_cluster_size_ratio: float = 0.01
    min_clusters_before_fallback: int = 3
    max_noise_ratio: float = 0.90
    #: If one cluster holds more than this share of the clustered units it is not
    #: a theme, it is the corpus. Excess-of-mass selection does this regularly on
    #: review text, and the resulting "cluster" has near-zero cohesion and yields
    #: a need like "users need the app to work". Triggers leaf re-selection.
    max_dominant_cluster_share: float = 0.50
    random_state: int = 42

    def resolve_min_cluster_size(self, n_units: int) -> int:
        if self.min_cluster_size is not None:
            return max(2, self.min_cluster_size)
        derived = int(self.min_cluster_size_ratio * n_units)
        return max(2, min(max(self.min_cluster_size_floor, derived), max(2, n_units // 3)))


@dataclass
class ClusteringResult:
    labels: np.ndarray  # (n,) int, -1 = noise
    probabilities: np.ndarray  # (n,) float membership strength
    method: str
    fallback: bool
    n_clusters: int
    noise_ratio: float

    def cluster_indices(self) -> dict[int, np.ndarray]:
        """Member row indices per cluster, noise excluded."""
        return {
            int(label): np.flatnonzero(self.labels == label)
            for label in sorted(set(self.labels.tolist()))
            if label != NOISE_LABEL
        }


def cluster_embeddings(
    embedding: np.ndarray, config: ClusteringConfig | None = None
) -> ClusteringResult:
    cfg = config or ClusteringConfig()
    n_units = len(embedding)
    if n_units == 0:
        return ClusteringResult(
            np.zeros(0, dtype=int), np.zeros(0, dtype=float), "none", False, 0, 0.0
        )
    if n_units < max(4, cfg.min_clusters_before_fallback):
        # Too little data to cluster meaningfully; one bucket is the honest answer.
        return ClusteringResult(
            np.zeros(n_units, dtype=int), np.ones(n_units), "single", True, 1, 0.0
        )

    result = _hdbscan(embedding, cfg)

    if _has_collapsed(result, cfg):
        log.warning(
            "HDBSCAN produced %d cluster(s) at %.0f%% noise; falling back to KMeans",
            result.n_clusters, result.noise_ratio * 100,
        )
        return _kmeans(embedding, cfg)

    # Excess-of-mass often returns one cluster containing most of the corpus.
    # Leaf selection splits the same condensed tree at its leaves instead,
    # producing finer and far more coherent themes.
    if _is_dominated(result, cfg):
        share = dominant_share(result.labels)
        log.warning(
            "HDBSCAN: largest cluster holds %.0f%% of clustered units; "
            "retrying with leaf selection", share * 100,
        )
        leaf = _hdbscan(embedding, cfg, selection_method="leaf")
        if not _has_collapsed(leaf, cfg) and not _is_dominated(leaf, cfg):
            return leaf
        log.warning(
            "leaf selection still degenerate (%d clusters, %.0f%% dominant); "
            "falling back to KMeans",
            leaf.n_clusters, dominant_share(leaf.labels) * 100,
        )
        return _kmeans(embedding, cfg)

    return result


def _has_collapsed(result: ClusteringResult, cfg: ClusteringConfig) -> bool:
    return (
        result.n_clusters < cfg.min_clusters_before_fallback
        or result.noise_ratio > cfg.max_noise_ratio
    )


def dominant_share(labels: np.ndarray) -> float:
    """Share of *clustered* (non-noise) units held by the largest cluster."""
    clustered = labels[labels != NOISE_LABEL]
    if len(clustered) == 0:
        return 0.0
    _, counts = np.unique(clustered, return_counts=True)
    return float(counts.max() / len(clustered))


def _is_dominated(result: ClusteringResult, cfg: ClusteringConfig) -> bool:
    return dominant_share(result.labels) > cfg.max_dominant_cluster_share


def _summarise(labels: np.ndarray) -> tuple[int, float]:
    n_clusters = len({int(x) for x in labels} - {NOISE_LABEL})
    noise_ratio = float((labels == NOISE_LABEL).sum() / len(labels)) if len(labels) else 0.0
    return n_clusters, noise_ratio


def _hdbscan(
    embedding: np.ndarray, cfg: ClusteringConfig, *, selection_method: str = "eom"
) -> ClusteringResult:
    from sklearn.cluster import HDBSCAN

    min_cluster_size = cfg.resolve_min_cluster_size(len(embedding))
    log.info(
        "HDBSCAN: %d units, min_cluster_size=%d, selection=%s",
        len(embedding), min_cluster_size, selection_method,
    )
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=cfg.min_samples,
        cluster_selection_method=selection_method,
        metric="euclidean",
        # Explicit: sklearn 1.10 flips this default, and we must not let the
        # clusterer mutate the caller's reduced embedding in place.
        copy=True,
    )
    labels = model.fit_predict(embedding)
    probabilities = np.asarray(
        getattr(model, "probabilities_", np.ones(len(embedding))), dtype=float
    )
    n_clusters, noise_ratio = _summarise(labels)
    log.info(
        "HDBSCAN: %d clusters, %.1f%% noise, largest holds %.0f%%",
        n_clusters, noise_ratio * 100, dominant_share(labels) * 100,
    )
    return ClusteringResult(
        labels, probabilities, f"hdbscan-{selection_method}", False, n_clusters, noise_ratio
    )


def _kmeans(embedding: np.ndarray, cfg: ClusteringConfig) -> ClusteringResult:
    from sklearn.cluster import KMeans

    k = _choose_k(len(embedding), cfg)
    log.info("KMeans fallback: %d units -> k=%d", len(embedding), k)
    model = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=10)
    labels = model.fit_predict(embedding)

    # KMeans has no membership probability. Use distance to the assigned centroid,
    # rescaled to 0..1, so downstream code has a comparable confidence signal.
    distances = np.linalg.norm(embedding - model.cluster_centers_[labels], axis=1)
    span = distances.max() - distances.min()
    probabilities = (
        np.ones(len(embedding)) if span <= 0 else 1.0 - (distances - distances.min()) / span
    )
    n_clusters, noise_ratio = _summarise(labels)
    return ClusteringResult(labels, probabilities, "kmeans", True, n_clusters, noise_ratio)


def _choose_k(n_units: int, cfg: ClusteringConfig) -> int:
    """A defensible k without a silhouette sweep: roughly sqrt(n/2), bounded."""
    target = int(np.sqrt(max(n_units, 1) / 2))
    return max(cfg.min_clusters_before_fallback, min(target, 12, max(2, n_units // 2)))
