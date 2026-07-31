"""Choosing which cluster members the LLM actually reads.

Never send a whole cluster to the LLM: it is expensive and the model will latch
onto whatever appeared first. Send 8-12 examples that are (a) typical, (b) mutually
diverse, and (c) weighted toward reviews other users found helpful.

Selection is Maximal Marginal Relevance seeded with the medoid, so the first pick
is the most typical member and each later pick trades off typicality against
redundancy with what is already chosen.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from aipm.clustering.metrics import centroid, medoid_index


def select_representatives(
    vectors: np.ndarray,
    *,
    n: int = 10,
    weights: Sequence[float] | None = None,
    diversity: float = 0.35,
) -> list[int]:
    """Return row indices of the chosen representatives, most typical first.

    `weights` is an optional per-member prior (helpful votes, quality weight).
    `diversity` in 0..1 sets how strongly redundancy is penalised.
    """
    n_members = len(vectors)
    if n_members == 0:
        return []
    if n_members <= n:
        return list(range(n_members))

    vectors = np.asarray(vectors, dtype=np.float32)
    relevance = vectors @ centroid(vectors)  # typicality

    if weights is not None:
        prior = np.asarray(weights, dtype=np.float32)
        if len(prior) != n_members:
            raise ValueError(f"weights length {len(prior)} != {n_members} members")
        # Log-compress: one review with 4000 helpful votes must not dominate.
        prior = np.log1p(np.maximum(prior, 0.0))
        span = prior.max() - prior.min()
        prior = np.zeros_like(prior) if span <= 0 else (prior - prior.min()) / span
        relevance = 0.75 * relevance + 0.25 * prior

    selected = [medoid_index(vectors)]
    # Running max similarity to the selected set; updated incrementally so the
    # loop stays O(n * k) rather than recomputing the full matrix each round.
    redundancy = vectors @ vectors[selected[0]]

    while len(selected) < n:
        score = (1.0 - diversity) * relevance - diversity * redundancy
        score[selected] = -np.inf
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            break
        selected.append(best)
        redundancy = np.maximum(redundancy, vectors @ vectors[best])
    return selected
