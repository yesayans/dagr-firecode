"""Review-level roadmap matching with cluster aggregation.

Short review text is where char_wb retrieval works. Cluster centroids / concatenated
blobs average away distinctive morphology and leave generic app-domain n-grams —
so we match each member review, then require agreement.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AggregatedMatch:
    item_index: int
    score: float
    n_agree: int
    n_members: int
    agreement_rate: float
    member_top1_mean: float
    runner_up_item_index: int | None = None
    runner_up_votes: int = 0

    @property
    def margin_votes(self) -> int:
        return self.n_agree - self.runner_up_votes


def min_agreement(n_members: int, *, min_count: int = 2, min_rate: float = 0.3) -> int:
    if n_members <= 1:
        return 1
    return max(min_count, int(np.ceil(min_rate * n_members)))


def per_review_best(
    review_emb: np.ndarray,
    roadmap_emb: np.ndarray,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each review row, return (best_global_idx, best_sim) among candidate indices.
    """
    if review_emb is None or len(review_emb) == 0 or not indices:
        empty_i = np.zeros(0, dtype=int)
        empty_s = np.zeros(0, dtype=float)
        return empty_i, empty_s
    cand = np.asarray(roadmap_emb)[indices]
    sims = np.asarray(review_emb) @ cand.T
    local = np.argmax(sims, axis=1)
    best_sims = sims[np.arange(len(sims)), local]
    best_idx = np.asarray(indices, dtype=int)[local]
    return best_idx, best_sims.astype(float)


def aggregate_cluster_match(
    member_emb: np.ndarray,
    roadmap_emb: np.ndarray,
    indices: list[int],
    *,
    min_count: int = 2,
    min_rate: float = 0.3,
    require_agreement: bool = True,
) -> AggregatedMatch | None:
    """
    Vote on each member's top-1 roadmap item; score = mean top-1 sim among agreers.

    When require_agreement is True (live matching), returns None below the agreement
    floor. When False (null-distribution construction), always returns the plurality
    winner so the score distribution is well-defined.
    """
    if member_emb is None or len(member_emb) == 0 or not indices:
        return None
    best_idx, best_sims = per_review_best(member_emb, roadmap_emb, indices)
    if len(best_idx) == 0:
        return None

    votes = Counter(int(i) for i in best_idx.tolist())
    ranked = votes.most_common()
    winner, n_agree = ranked[0]
    runner_idx = ranked[1][0] if len(ranked) > 1 else None
    runner_votes = ranked[1][1] if len(ranked) > 1 else 0

    n_members = len(best_idx)
    need = min_agreement(n_members, min_count=min_count, min_rate=min_rate)
    if require_agreement and n_agree < need:
        return None

    agree_sims = best_sims[best_idx == winner]
    score = float(np.mean(agree_sims)) if len(agree_sims) else 0.0
    return AggregatedMatch(
        item_index=int(winner),
        score=score,
        n_agree=int(n_agree),
        n_members=n_members,
        agreement_rate=float(n_agree / max(n_members, 1)),
        member_top1_mean=float(np.mean(best_sims)),
        runner_up_item_index=runner_idx,
        runner_up_votes=int(runner_votes),
    )


def member_embeddings_for_cluster(
    cluster: dict[str, Any],
    reviews_df: Any,
    review_embeddings: np.ndarray,
) -> np.ndarray:
    """Resolve cluster members to embedding rows."""
    emb = np.asarray(review_embeddings)
    indices = cluster.get("member_indices")
    if indices is not None and len(indices) > 0:
        return emb[np.asarray(indices, dtype=int)]

    ids = list(cluster.get("review_ids") or [])
    if not ids or reviews_df is None or getattr(reviews_df, "empty", True):
        return np.zeros((0, emb.shape[1] if emb.ndim == 2 else 1))
    id_to_pos = {
        str(rid): i for i, rid in enumerate(reviews_df["review_id"].astype(str).tolist())
    }
    rows = [id_to_pos[str(r)] for r in ids if str(r) in id_to_pos]
    if not rows:
        return np.zeros((0, emb.shape[1] if emb.ndim == 2 else 1))
    return emb[np.asarray(rows, dtype=int)]
