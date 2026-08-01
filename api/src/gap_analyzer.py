"""GapMatrix: review-level matching, verdict rules, deterministic confidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.hiddenness import annotate_metrics_hiddenness
from src.matching_space import MatchingSpace
from src.review_match import (
    AggregatedMatch,
    aggregate_cluster_match,
    member_embeddings_for_cluster,
)

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

_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%B %d %Y",
    "%b %d %Y",
    "%B %d, %Y",
    "%b %d, %Y",
)


@dataclass
class ReviewWindow:
    start: datetime
    end: datetime

    def to_iso(self) -> dict[str, str]:
        return {
            "review_window_start": self.start.isoformat(),
            "review_window_end": self.end.isoformat(),
            "reference_date": self.end.isoformat(),
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


def parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, float) and np.isnan(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "nan", "nat"}:
        return None
    if s.endswith("Z"):
        s_z = s[:-1] + "+0000"
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
            try:
                return datetime.strptime(s_z, fmt).astimezone(timezone.utc)
            except ValueError:
                pass
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def compute_review_window(reviews_df: pd.DataFrame) -> ReviewWindow:
    """Anchor temporal rules to the corpus, not wall-clock now."""
    dates: list[datetime] = []
    if reviews_df is not None and not reviews_df.empty and "created_at" in reviews_df.columns:
        for v in reviews_df["created_at"].tolist():
            dt = parse_dt(v)
            if dt is not None:
                dates.append(dt)
    if not dates:
        now = datetime.now(timezone.utc)
        return ReviewWindow(start=now, end=now)
    return ReviewWindow(start=min(dates), end=max(dates))


def _item_created(item: dict[str, Any]) -> datetime | None:
    return parse_dt(item.get("created_at"))


def _item_closed(item: dict[str, Any]) -> datetime | None:
    return parse_dt(item.get("closed_at"))


def _item_updated(item: dict[str, Any]) -> datetime | None:
    return parse_dt(item.get("updated_at"))


def _is_closed_like(item: dict[str, Any]) -> bool:
    state = str(item.get("state") or "").lower()
    return state in {"closed", "shipped", "released", "done", "resolved"}


def classify_item_vs_window(item: dict[str, Any], window: ReviewWindow) -> str:
    """
    Return 'future' | 'closed' | 'open' relative to review_window_end.
    Future = created after the corpus window (retrospective-only).
    """
    created = _item_created(item)
    if created is not None and created > window.end:
        return "future"
    closed = _item_closed(item)
    if closed is not None and closed <= window.end:
        return "closed"
    if _is_closed_like(item) and (closed is None or closed <= window.end):
        # Closed without usable date, or closed in-window
        if closed is not None and closed > window.end:
            return "open"
        return "closed"
    return "open"


def age_days_as_of(item: dict[str, Any] | None, window: ReviewWindow) -> float | None:
    if not item:
        return None
    touch = _item_updated(item) or _item_created(item)
    if touch is None:
        return None
    return max(0.0, (window.end - touch).total_seconds() / 86400.0)


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
    """Implement CONTRACT.md sections 3 and 4 (corpus-anchored temporal rules)."""

    def __init__(
        self,
        settings: Settings | None = None,
        match_threshold: float | None = None,
        match_margin: float | None = None,
        matching_space: MatchingSpace | None = None,
        roadmap_matching_enabled: bool | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.matching_space = matching_space
        self.roadmap_matching_enabled = (
            bool(roadmap_matching_enabled)
            if roadmap_matching_enabled is not None
            else bool(self.settings.roadmap_matching_enabled)
        )
        self.match_threshold = (
            match_threshold
            if match_threshold is not None
            else (
                matching_space.threshold
                if matching_space is not None
                else self.settings.active_match_threshold()
            )
        )
        self.match_margin = (
            match_margin
            if match_margin is not None
            else self.settings.active_match_margin()
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
        review_window: ReviewWindow | None = None,
    ) -> list[CandidateGap]:
        mode = (
            roadmap_source
            if roadmap_source in ("github", "web", "hybrid", "none")
            else "none"
        )
        window = review_window or compute_review_window(reviews_df)
        window_meta = window.to_iso()

        if (
            not self.roadmap_matching_enabled
            or mode == "none"
            or roadmap_items is None
            or roadmap_items.empty
        ):
            return self._analyze_none(clusters, reviews_df, total_reviews, window_meta)

        items = [row.to_dict() for _, row in roadmap_items.iterrows()]
        n_items = len(items)
        if roadmap_embeddings is None or len(roadmap_embeddings) == 0:
            emb = np.zeros((n_items, 1))
        else:
            emb = np.asarray(roadmap_embeddings)

        contemp_idx = [
            i
            for i, it in enumerate(items)
            if classify_item_vs_window(it, window) != "future"
        ]
        future_idx = [
            i
            for i, it in enumerate(items)
            if classify_item_vs_window(it, window) == "future"
        ]

        max_size = max((c["size"] for c in clusters), default=1)
        candidates: list[CandidateGap] = []

        for cluster in clusters:
            agg = self._cluster_match(
                cluster, reviews_df, review_embeddings, emb, contemp_idx
            )
            best_sim = agg.score if agg is not None else None
            best_idx = agg.item_index if agg is not None else None
            best_margin = None
            if agg is not None:
                best_margin = float(agg.agreement_rate)

            verdict, matched = self._verdict(
                best_sim,
                best_idx,
                agg,
                items,
                cluster,
                reviews_df,
                window,
            )
            if verdict is None:
                continue

            later = self._later_addressed(
                cluster, reviews_df, review_embeddings, emb, items, future_idx
            )
            later_extra = self._later_addressed_post_close(
                cluster, reviews_df, review_embeddings, emb, items, window
            )
            later = later or later_extra

            conf, components, weights = compute_confidence(
                cluster_size=cluster["size"],
                max_cluster_size=max_size,
                best_similarity=best_sim,
                cohesion=cluster["cohesion"],
                mean_rating=cluster["mean_rating"],
                rating_spread=cluster["rating_spread"],
                mode=mode,
            )
            matched_title = None
            if matched:
                matched_title = (matched.get("title") or matched.get("text") or "")[
                    :120
                ]
            age = age_days_as_of(matched, window) if matched else None
            metrics = {
                "cluster_size": cluster["size"],
                "total_reviews": total_reviews,
                "cluster_share": cluster["size"] / max(total_reviews, 1),
                "best_similarity": best_sim,
                "best_similarity_margin": best_margin,
                "match_agreement": agg.n_agree if agg else 0,
                "match_agreement_rate": agg.agreement_rate if agg else 0.0,
                "match_threshold": self.match_threshold,
                "match_threshold_source": (
                    "null_model" if self.matching_space is not None else "absolute"
                ),
                "matched_item_title": matched_title,
                "matched_item_url": (matched or {}).get("url"),
                "matched_item_state": (matched or {}).get("state"),
                "matched_item_age_days": age,
                "mean_rating": cluster["mean_rating"],
                "rating_spread": cluster["rating_spread"],
                "cohesion": cluster["cohesion"],
                "components": components,
                "weights": weights,
                "deterministic_confidence": conf,
                "llm_confidence": None,
                "keywords": cluster.get("keywords") or [],
                "need_bearing_share": float(cluster.get("need_bearing_share") or 1.0),
                **window_meta,
                "later_addressed_by": later,
                "validated_by_later_roadmap": later is not None,
            }
            annotate_metrics_hiddenness(
                metrics, self._member_texts(cluster, reviews_df)
            )
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
            key=lambda c: (
                float(c.metrics.get("insight_score") or 0.0),
                float(c.metrics.get("deterministic_confidence") or 0.0),
            ),
            reverse=True,
        )
        return candidates

    def _member_texts(
        self, cluster: dict[str, Any], reviews_df: pd.DataFrame
    ) -> list[str]:
        ids = set(str(x) for x in (cluster.get("review_ids") or []))
        if reviews_df is None or reviews_df.empty or not ids:
            return []
        sub = reviews_df[reviews_df["review_id"].astype(str).isin(ids)]
        return [str(t) for t in sub["review_text"].tolist()]

    def _cluster_match(
        self,
        cluster: dict[str, Any],
        reviews_df: pd.DataFrame,
        review_embeddings: np.ndarray,
        roadmap_embeddings: np.ndarray,
        indices: list[int],
    ) -> AggregatedMatch | None:
        if self.matching_space is not None:
            return self.matching_space.match_cluster(
                self._member_texts(cluster, reviews_df), indices
            )
        member_emb = member_embeddings_for_cluster(
            cluster, reviews_df, review_embeddings
        )
        return aggregate_cluster_match(member_emb, roadmap_embeddings, indices)

    def _analyze_none(
        self,
        clusters: list[dict[str, Any]],
        reviews_df: pd.DataFrame,
        total_reviews: int,
        window_meta: dict[str, str],
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
                "need_bearing_share": float(cluster.get("need_bearing_share") or 1.0),
                **window_meta,
                "later_addressed_by": None,
                "validated_by_later_roadmap": False,
            }
            annotate_metrics_hiddenness(
                metrics, self._member_texts(cluster, reviews_df)
            )
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
        out.sort(
            key=lambda c: (
                float(c.metrics.get("insight_score") or 0.0),
                float(c.metrics.get("deterministic_confidence") or 0.0),
            ),
            reverse=True,
        )
        return out

    def _accepts_match(self, agg: AggregatedMatch | None) -> bool:
        if agg is None:
            return False
        if self.matching_space is not None:
            return self.matching_space.accepts(agg)
        return agg.score >= self.match_threshold

    def _verdict(
        self,
        best_sim: float | None,
        best_idx: int | None,
        agg: AggregatedMatch | None,
        items: list[dict[str, Any]],
        cluster: dict[str, Any],
        reviews_df: pd.DataFrame,
        window: ReviewWindow,
    ) -> tuple[str | None, dict[str, Any] | None]:
        if not self._accepts_match(agg):
            return "IGNORED", None

        matched = items[best_idx] if best_idx is not None and items else None
        if matched is None:
            return "IGNORED", None

        as_of = classify_item_vs_window(matched, window)
        if as_of == "future":
            return "IGNORED", None

        if as_of == "closed":
            if self._misunderstood(cluster, reviews_df, matched, window):
                return "MISUNDERSTOOD", matched
            return None, matched

        age = age_days_as_of(matched, window)
        stale = age is not None and float(age) > 365
        has_milestone = bool(matched.get("milestone_title"))
        if stale or not has_milestone:
            return "UNDER-PRIORITIZED", matched
        return None, matched

    def _misunderstood(
        self,
        cluster: dict[str, Any],
        reviews_df: pd.DataFrame,
        matched: dict[str, Any],
        window: ReviewWindow,
    ) -> bool:
        closed = _item_closed(matched)
        if closed is None:
            return True
        if closed > window.end:
            return False

        ids = set(cluster["review_ids"])
        sub = reviews_df[reviews_df["review_id"].isin(ids)]
        for _, row in sub.iterrows():
            dt = parse_dt(row.get("created_at"))
            if dt is None:
                return True
            if dt >= closed and dt <= window.end:
                return True
            if dt > closed:
                return True
        return False

    def _later_addressed(
        self,
        cluster: dict[str, Any],
        reviews_df: pd.DataFrame,
        review_embeddings: np.ndarray,
        emb: np.ndarray,
        items: list[dict[str, Any]],
        future_idx: list[int],
    ) -> dict[str, Any] | None:
        if not future_idx:
            return None
        agg = self._cluster_match(
            cluster, reviews_df, review_embeddings, emb, future_idx
        )
        if not self._accepts_match(agg) or agg is None:
            return None
        return self._later_payload(items[agg.item_index], agg.score, agg)

    def _later_addressed_post_close(
        self,
        cluster: dict[str, Any],
        reviews_df: pd.DataFrame,
        review_embeddings: np.ndarray,
        emb: np.ndarray,
        items: list[dict[str, Any]],
        window: ReviewWindow,
    ) -> dict[str, Any] | None:
        idxs = []
        for i, it in enumerate(items):
            created = _item_created(it)
            if created is not None and created > window.end:
                continue
            closed = _item_closed(it)
            updated = _item_updated(it)
            post = False
            if closed is not None and closed > window.end:
                post = True
            if updated is not None and updated > window.end and _is_closed_like(it):
                post = True
            if post:
                idxs.append(i)
        if not idxs:
            return None
        agg = self._cluster_match(cluster, reviews_df, review_embeddings, emb, idxs)
        if not self._accepts_match(agg) or agg is None:
            return None
        return self._later_payload(items[agg.item_index], agg.score, agg)

    def _later_payload(
        self, item: dict[str, Any], sim: float, agg: AggregatedMatch | None = None
    ) -> dict[str, Any]:
        closed = _item_closed(item)
        created = _item_created(item)
        updated = _item_updated(item)
        date = closed or created or updated
        payload = {
            "title": (item.get("title") or item.get("text") or "")[:200],
            "url": item.get("url"),
            "state": item.get("state"),
            "date": date.isoformat() if date else None,
            "similarity": float(sim),
        }
        if agg is not None:
            payload["agreement"] = agg.n_agree
            payload["agreement_rate"] = agg.agreement_rate
        return payload


def main() -> None:
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
