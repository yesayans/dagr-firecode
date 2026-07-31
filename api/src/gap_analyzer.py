"""GapMatrix: similarity matching, verdict rules, deterministic confidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.config import Settings, get_settings


WEIGHTS_ROADMAP = {
    "volume": 0.30,
    "novelty": 0.25,
    "consistency": 0.20,
    "severity": 0.15,
    "spread": 0.10,
}
WEIGHTS_NONE = {
    "volume": 0.35,
    "novelty": 0.00,
    "consistency": 0.30,
    "severity": 0.20,
    "spread": 0.15,
}


@dataclass
class CandidateGap:
    cluster_id: int
    review_ids: list[str]
    verdict: str
    best_similarity: float | None
    matched_item: dict[str, Any] | None
    metrics: dict[str, Any]
    keywords: list[str] = field(default_factory=list)
    representative_text: str = ""
    cohesion: float = 0.0
    mean_rating: float = 3.0
    cluster_size: int = 0


def compute_confidence(
    *,
    cluster_size: int,
    max_cluster_size: int,
    best_similarity: float | None,
    cohesion: float,
    mean_rating: float,
    rating_spread: float,
    mode: str,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Return (deterministic_confidence, components, weights)."""
    weights = dict(WEIGHTS_NONE if mode == "none" else WEIGHTS_ROADMAP)
    volume = float(np.log1p(cluster_size) / np.log1p(max(max_cluster_size, 1)))
    if mode == "none":
        novelty = 1.0
    else:
        sim = 0.0 if best_similarity is None else float(best_similarity)
        novelty = 1.0 - sim
    consistency = float(np.clip(cohesion, 0.0, 1.0))
    severity = float(np.clip((5.0 - mean_rating) / 4.0, 0.0, 1.0))
    spread = float(np.clip(rating_spread, 0.0, 1.0))
    components = {
        "volume": volume,
        "novelty": novelty,
        "consistency": consistency,
        "severity": severity,
        "spread": spread,
    }
    score = 100.0 * sum(weights[k] * components[k] for k in weights)
    return round(score, 2), components, weights


def reconstruct_confidence(metrics: dict[str, Any]) -> float:
    """Recompute confidence from persisted metrics components × weights."""
    components = metrics["components"]
    weights = metrics["weights"]
    score = 100.0 * sum(float(weights[k]) * float(components[k]) for k in weights)
    return round(score, 2)


class GapMatrix:
    """Implement CONTRACT.md sections 3 and 4 exactly."""

    def __init__(self, settings: Settings | None = None, match_threshold: float | None = None) -> None:
        self.settings = settings or get_settings()
        self.match_threshold = (
            match_threshold
            if match_threshold is not None
            else self.settings.active_match_threshold()
        )

    def analyze(
        self,
        *,
        clusters: list[dict[str, Any]],
        review_embeddings: np.ndarray,
        reviews_df: pd.DataFrame,
        roadmap_items: pd.DataFrame,
        roadmap_embeddings: np.ndarray,
        roadmap_source: str,
        total_reviews: int,
    ) -> list[CandidateGap]:
        mode = roadmap_source if roadmap_source in ("github", "web", "hybrid", "none") else "none"
        if mode == "none" or roadmap_items is None or roadmap_items.empty:
            return self._analyze_none(clusters, reviews_df, total_reviews)

        max_size = max((c["size"] for c in clusters), default=1)
        candidates: list[CandidateGap] = []

        item_states = []
        for _, row in roadmap_items.iterrows():
            item_states.append(row.to_dict())

        for cluster in clusters:
            centroid = cluster["centroid"]
            if roadmap_embeddings is None or len(roadmap_embeddings) == 0:
                best_sim = None
                best_idx = None
            else:
                sims = roadmap_embeddings @ centroid
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])

            verdict, matched = self._verdict(
                best_sim, best_idx, item_states, cluster, reviews_df
            )
            if verdict is None:
                # well covered — drop
                continue

            conf, components, weights = compute_confidence(
                cluster_size=cluster["size"],
                max_cluster_size=max_size,
                best_similarity=best_sim,
                cohesion=cluster["cohesion"],
                mean_rating=cluster["mean_rating"],
                rating_spread=cluster["rating_spread"],
                mode=mode,
            )
            matched_title = matched.get("text", "")[:120] if matched else None
            metrics = {
                "cluster_size": cluster["size"],
                "total_reviews": total_reviews,
                "cluster_share": cluster["size"] / max(total_reviews, 1),
                "best_similarity": best_sim,
                "matched_item_title": matched_title,
                "matched_item_url": (matched or {}).get("url"),
                "matched_item_state": (matched or {}).get("state"),
                "matched_item_age_days": (matched or {}).get("age_days"),
                "mean_rating": cluster["mean_rating"],
                "rating_spread": cluster["rating_spread"],
                "cohesion": cluster["cohesion"],
                "components": components,
                "weights": weights,
                "deterministic_confidence": conf,
                "llm_confidence": None,
                "keywords": cluster.get("keywords") or [],
            }
            candidates.append(
                CandidateGap(
                    cluster_id=cluster["cluster_id"],
                    review_ids=list(cluster["review_ids"]),
                    verdict=verdict,
                    best_similarity=best_sim,
                    matched_item=matched,
                    metrics=metrics,
                    keywords=cluster.get("keywords") or [],
                    representative_text=cluster.get("representative_text") or "",
                    cohesion=cluster["cohesion"],
                    mean_rating=cluster["mean_rating"],
                    cluster_size=cluster["size"],
                )
            )

        candidates.sort(
            key=lambda c: c.metrics["deterministic_confidence"], reverse=True
        )
        return candidates

    def _analyze_none(
        self,
        clusters: list[dict[str, Any]],
        reviews_df: pd.DataFrame,
        total_reviews: int,
    ) -> list[CandidateGap]:
        max_size = max((c["size"] for c in clusters), default=1)
        out: list[CandidateGap] = []
        for cluster in clusters:
            conf, components, weights = compute_confidence(
                cluster_size=cluster["size"],
                max_cluster_size=max_size,
                best_similarity=None,
                cohesion=cluster["cohesion"],
                mean_rating=cluster["mean_rating"],
                rating_spread=cluster["rating_spread"],
                mode="none",
            )
            metrics = {
                "cluster_size": cluster["size"],
                "total_reviews": total_reviews,
                "cluster_share": cluster["size"] / max(total_reviews, 1),
                "best_similarity": None,
                "matched_item_title": None,
                "matched_item_url": None,
                "matched_item_state": None,
                "matched_item_age_days": None,
                "mean_rating": cluster["mean_rating"],
                "rating_spread": cluster["rating_spread"],
                "cohesion": cluster["cohesion"],
                "components": components,
                "weights": weights,
                "deterministic_confidence": conf,
                "llm_confidence": None,
                "keywords": cluster.get("keywords") or [],
            }
            out.append(
                CandidateGap(
                    cluster_id=cluster["cluster_id"],
                    review_ids=list(cluster["review_ids"]),
                    verdict="UNVERIFIED",
                    best_similarity=None,
                    matched_item=None,
                    metrics=metrics,
                    keywords=cluster.get("keywords") or [],
                    representative_text=cluster.get("representative_text") or "",
                    cohesion=cluster["cohesion"],
                    mean_rating=cluster["mean_rating"],
                    cluster_size=cluster["size"],
                )
            )
        out.sort(key=lambda c: c.metrics["deterministic_confidence"], reverse=True)
        return out

    def _verdict(
        self,
        best_sim: float | None,
        best_idx: int | None,
        items: list[dict[str, Any]],
        cluster: dict[str, Any],
        reviews_df: pd.DataFrame,
    ) -> tuple[str | None, dict[str, Any] | None]:
        threshold = self.match_threshold
        if best_sim is None or best_sim < threshold:
            return "IGNORED", None

        matched = items[best_idx] if best_idx is not None and items else None
        if matched is None:
            return "IGNORED", None

        state = str(matched.get("state") or "open").lower()
        closed_like = state in ("closed", "shipped", "released", "done", "completed")
        # Labels may mark shipped current features
        labels = str(matched.get("labels") or "").lower()
        kind = str(matched.get("kind") or "").lower()
        if kind in ("current_feature",) or "shipped" in labels:
            closed_like = True

        if closed_like:
            if self._still_complaining_after_close(cluster, reviews_df, matched):
                return "MISUNDERSTOOD", matched
            # Closed and users not complaining more recently → treat as covered
            return None, matched

        # Open item
        age = matched.get("age_days")
        if age is None and matched.get("updated_at"):
            age = _age_days(matched.get("updated_at"))
        stale = age is not None and float(age) > 365
        has_milestone = bool(matched.get("milestone_title"))
        if stale or not has_milestone:
            return "UNDER-PRIORITIZED", matched
        # open, fresh, milestoned → well covered
        return None, matched

    def _still_complaining_after_close(
        self,
        cluster: dict[str, Any],
        reviews_df: pd.DataFrame,
        matched: dict[str, Any],
    ) -> bool:
        closed_at = matched.get("closed_at")
        if not closed_at:
            # Shipped feature without close date — if reviews exist (filtered ≤4★), treat as misunderstood
            return True
        try:
            close_dt = datetime.fromisoformat(str(closed_at).replace("Z", "+00:00"))
        except Exception:
            return True

        ids = set(cluster["review_ids"])
        sub = reviews_df[reviews_df["review_id"].isin(ids)]
        recent = False
        for _, row in sub.iterrows():
            created = row.get("created_at")
            if not created or (isinstance(created, float) and np.isnan(created)):
                # Unknown date but low rating → count as still complaining
                recent = True
                continue
            try:
                dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt > close_dt:
                    recent = True
                    break
            except Exception:
                recent = True
        return recent


def _age_days(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return max(0.0, (now - dt).total_seconds() / 86400.0)
    except Exception:
        return None


def main() -> None:
    # Quick confidence reconstruct check
    conf, comps, weights = compute_confidence(
        cluster_size=10,
        max_cluster_size=20,
        best_similarity=0.1,
        cohesion=0.8,
        mean_rating=2.0,
        rating_spread=0.6,
        mode="github",
    )
    metrics = {"components": comps, "weights": weights}
    assert reconstruct_confidence(metrics) == conf
    print({"confidence": conf, "ok": True})


if __name__ == "__main__":
    main()
