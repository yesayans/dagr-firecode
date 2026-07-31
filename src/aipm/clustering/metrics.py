"""Deterministic cluster quality metrics.

These feed the confidence model, so they are computed from vectors only - the
LLM never sees them before they are fixed, and never produces them.
"""

from __future__ import annotations

import numpy as np

#: Sampling cap for the O(n^2) cohesion computation. A 5k-member cluster would
#: otherwise build a 25M-entry similarity matrix for a number that is stable at
#: a fraction of that.
_COHESION_SAMPLE_CAP = 512


def centroid(vectors: np.ndarray) -> np.ndarray:
    """Mean vector, renormalised so it stays comparable under cosine."""
    if len(vectors) == 0:
        return np.zeros(vectors.shape[1] if vectors.ndim == 2 else 0, dtype=np.float32)
    mean = vectors.mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm if norm > 1e-12 else mean).astype(np.float32)


def cohesion(vectors: np.ndarray, *, random_state: int = 42) -> float:
    """Mean pairwise cosine similarity within a cluster, in 0..1.

    Inputs are L2-normalised, so the dot product is the cosine. Negative
    similarities are clipped away: for a *tightness* score they read the same as
    "not similar".
    """
    n = len(vectors)
    if n < 2:
        return 0.0
    if n > _COHESION_SAMPLE_CAP:
        rng = np.random.default_rng(random_state)
        vectors = vectors[rng.choice(n, _COHESION_SAMPLE_CAP, replace=False)]
        n = _COHESION_SAMPLE_CAP

    similarity = np.asarray(vectors, dtype=np.float32) @ np.asarray(vectors, dtype=np.float32).T
    # Exclude the diagonal (self-similarity is always 1 and would inflate this).
    total = float(similarity.sum() - np.trace(similarity))
    return float(np.clip(total / (n * (n - 1)), 0.0, 1.0))


def separation(this_centroid: np.ndarray, other_centroids: list[np.ndarray]) -> float:
    """Distance to the nearest other cluster centroid, normalised to 0..1.

    1.0 means the nearest neighbouring theme is orthogonal; 0.0 means it sits on
    top of this one. A lone cluster scores 1.0 - nothing competes with it.
    """
    if not other_centroids:
        return 1.0
    others = np.vstack(other_centroids).astype(np.float32)
    similarities = others @ np.asarray(this_centroid, dtype=np.float32)
    nearest = float(np.max(similarities))
    # Cosine similarity of the nearest neighbour -> distance, clipped to 0..1.
    return float(np.clip(1.0 - nearest, 0.0, 1.0))


def medoid_index(vectors: np.ndarray) -> int:
    """Index of the member closest to the centroid - the most typical example."""
    if len(vectors) == 0:
        return -1
    similarities = np.asarray(vectors, dtype=np.float32) @ centroid(vectors)
    return int(np.argmax(similarities))
