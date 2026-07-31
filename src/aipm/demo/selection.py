"""Choosing which apps the precomputed demo ships with.

The demo must show the product at its best without cherry-picking by hand, so
selection is a scored, explainable, configurable strategy rather than a literal
list of app ids.

Five signals, each normalised across the candidate pool:

* **volume**   - enough reviews for clusters to form at all (log-scaled)
* **quality**  - share of reviews long enough to say something specific
* **recency**  - enough reviews in the recent window to look current
* **coverage** - how many star levels the sample actually contains
* **engagement** - share of reviews that other users voted helpful

Category diversity is *not* a scoring term. Weighting it would let one crowded
category quietly dominate; instead it is a hard constraint applied during greedy
selection, which is both stronger and easier to explain.

A note on this corpus: it is a quota-capped scrape. Some apps hold an identical
number of reviews per star level; others hold only 1-star reviews. `coverage`
exists to keep the second kind out of the demo, and `quota_capped` is recorded on
every candidate so the UI can disclose it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aipm.config import Settings
from aipm.ingest.normalize import primary_category
from aipm.schemas import App
from aipm.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {
    "volume": 0.30,
    "quality": 0.30,
    "recency": 0.20,
    "coverage": 0.10,
    "engagement": 0.10,
}


@dataclass(frozen=True)
class DemoSelectionConfig:
    """The strategy. Overridable from settings, a JSON file, or CLI flags."""

    n_apps: int = 8
    min_apps: int = 5
    max_apps: int = 10

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    # -- hard filters ------------------------------------------------------
    min_reviews: int = 800
    #: Apps whose sample covers fewer star levels than this are excluded. An app
    #: with only 1-star reviews produces a confident, completely misleading demo.
    min_star_levels: int = 3
    min_share_substantive: float = 0.10
    min_months_covered: int = 6
    recency_window_days: int = 365
    min_share_recent: float = 0.0

    # -- diversity ---------------------------------------------------------
    #: Hard cap on apps per primary category.
    max_per_category: int = 2
    #: Try to fill this many distinct categories before doubling up on any.
    prefer_distinct_categories: bool = True

    # -- pinning -----------------------------------------------------------
    include_app_ids: tuple[str, ...] = ()
    exclude_app_ids: tuple[str, ...] = ()

    strategy_name: str = "default"

    def __post_init__(self) -> None:
        if not self.min_apps <= self.n_apps <= self.max_apps:
            raise ValueError(
                f"n_apps ({self.n_apps}) must be between min_apps ({self.min_apps}) "
                f"and max_apps ({self.max_apps})"
            )
        unknown = set(self.weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown selection weight(s): {sorted(unknown)}")
        if sum(self.weights.values()) <= 0:
            raise ValueError("selection weights must sum to a positive number")

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> DemoSelectionConfig:
        base: dict[str, Any] = {
            "n_apps": settings.demo_n_apps,
            "min_apps": settings.demo_min_apps,
            "max_apps": settings.demo_max_apps,
        }
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)

    @classmethod
    def from_json(cls, path: Path, **overrides: Any) -> DemoSelectionConfig:
        """Load a strategy from disk, so a demo can be tuned without a code change."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "weights" in payload:
            payload["weights"] = {str(k): float(v) for k, v in payload["weights"].items()}
        for key in ("include_app_ids", "exclude_app_ids"):
            if key in payload:
                payload[key] = tuple(str(v) for v in payload[key])
        payload.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**payload)

    def merged(self, **overrides: Any) -> DemoSelectionConfig:
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "n_apps": self.n_apps,
            "weights": dict(self.weights),
            "min_reviews": self.min_reviews,
            "min_star_levels": self.min_star_levels,
            "min_share_substantive": self.min_share_substantive,
            "min_months_covered": self.min_months_covered,
            "recency_window_days": self.recency_window_days,
            "min_share_recent": self.min_share_recent,
            "max_per_category": self.max_per_category,
            "include_app_ids": list(self.include_app_ids),
            "exclude_app_ids": list(self.exclude_app_ids),
        }


@dataclass
class AppCandidate:
    """One scored app, with the reasoning that produced the score."""

    app_id: str
    name: str
    category: str
    n_reviews: int
    avg_score: float
    share_substantive: float
    share_recent: float
    n_months: int
    n_star_levels: int
    share_helpful: float
    quota_capped: bool = False

    signals: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    rejected_because: str | None = None

    @property
    def eligible(self) -> bool:
        return self.rejected_because is None


@dataclass
class SelectionResult:
    selected: list[AppCandidate] = field(default_factory=list)
    rejected: list[AppCandidate] = field(default_factory=list)
    considered: int = 0
    config: DemoSelectionConfig = field(default_factory=DemoSelectionConfig)
    warnings: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        categories = {c.category for c in self.selected}
        return (
            f"selected {len(self.selected)}/{self.considered} apps "
            f"across {len(categories)} categories"
        )


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scale to 0..1. A constant column becomes 0.5 - it carries no information."""
    if len(values) == 0:
        return values
    finite = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    span = finite.max() - finite.min()
    if span <= 0:
        return np.full_like(finite, 0.5)
    return (finite - finite.min()) / span


class DemoAppSelector:
    """Scores every app with reviews, then picks a diverse, high-quality subset."""

    def __init__(self, config: DemoSelectionConfig | None = None) -> None:
        self.config = config or DemoSelectionConfig()

    # -- scoring -----------------------------------------------------------

    def build_candidates(
        self,
        stats: pd.DataFrame,
        apps: Mapping[str, App],
        *,
        recent_shares: Mapping[str, float] | None = None,
        raw_categories: Mapping[str, str] | None = None,
    ) -> list[AppCandidate]:
        """Assemble candidates from the per-app aggregate frame."""
        recent_shares = recent_shares or {}
        candidates: list[AppCandidate] = []

        for app_id, row in stats.iterrows():
            app_id = str(app_id)
            app = apps.get(app_id)
            if app is None:
                # Reviews exist for an app absent from apps_info; cannot label it.
                continue
            category = (
                app.categories[0]
                if app.categories
                else primary_category((raw_categories or {}).get(app_id, ""))
            )
            n_star_levels = int(row.get("n_star_levels", 0) or 0)
            candidates.append(
                AppCandidate(
                    app_id=app_id,
                    name=app.name,
                    category=category or "Uncategorised",
                    n_reviews=int(row["n_reviews"]),
                    avg_score=float(row.get("avg_score", 0.0) or 0.0),
                    share_substantive=float(row.get("share_substantive", 0.0) or 0.0),
                    share_recent=float(recent_shares.get(app_id, 0.0)),
                    n_months=int(row.get("n_months", 0) or 0),
                    n_star_levels=n_star_levels,
                    share_helpful=float(row.get("share_helpful", 0.0) or 0.0),
                    quota_capped=_looks_quota_capped(row),
                )
            )
        return candidates

    def score(self, candidates: Sequence[AppCandidate]) -> list[AppCandidate]:
        """Normalise each signal across the pool and combine with the weights."""
        if not candidates:
            return []

        signals = {
            # Log-scaled: 20k reviews is not twice as useful as 10k.
            "volume": _minmax(np.log1p([c.n_reviews for c in candidates])),
            "quality": _minmax(np.array([c.share_substantive for c in candidates])),
            "recency": _minmax(np.array([c.share_recent for c in candidates])),
            "coverage": np.array([c.n_star_levels / 5.0 for c in candidates]),
            "engagement": _minmax(np.array([c.share_helpful for c in candidates])),
        }
        weights = self.config.weights
        total_weight = sum(weights.get(k, 0.0) for k in signals) or 1.0

        for i, candidate in enumerate(candidates):
            candidate.signals = {k: round(float(v[i]), 4) for k, v in signals.items()}
            candidate.score = round(
                sum(candidate.signals[k] * weights.get(k, 0.0) for k in signals) / total_weight,
                4,
            )
            candidate.reasons = self._reasons(candidate)
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def _reasons(self, candidate: AppCandidate) -> list[str]:
        """Plain-language justification, shown in the manifest and the catalogue."""
        reasons = [f"{candidate.n_reviews:,} reviews"]
        if candidate.share_substantive >= 0.5:
            reasons.append(f"{candidate.share_substantive:.0%} substantive review text")
        if candidate.share_recent >= 0.25:
            reasons.append(f"{candidate.share_recent:.0%} of reviews are recent")
        if candidate.n_months >= 24:
            reasons.append(f"spans {candidate.n_months} months")
        if candidate.n_star_levels == 5:
            reasons.append("all five star levels present")
        if candidate.share_helpful >= 0.4:
            reasons.append(f"{candidate.share_helpful:.0%} received helpful votes")
        if candidate.quota_capped:
            reasons.append("note: star distribution looks quota-capped by the scraper")
        return reasons

    # -- filtering ---------------------------------------------------------

    def apply_filters(self, candidates: Sequence[AppCandidate]) -> list[AppCandidate]:
        """Mark ineligible candidates in place, returning the same list."""
        cfg = self.config
        for candidate in candidates:
            if candidate.app_id in cfg.exclude_app_ids:
                candidate.rejected_because = "explicitly excluded"
            elif candidate.app_id in cfg.include_app_ids:
                continue  # pinned apps bypass every quality filter
            elif candidate.n_reviews < cfg.min_reviews:
                candidate.rejected_because = (
                    f"only {candidate.n_reviews:,} reviews (min {cfg.min_reviews:,})"
                )
            elif candidate.n_star_levels < cfg.min_star_levels:
                candidate.rejected_because = (
                    f"sample covers only {candidate.n_star_levels} star level(s) "
                    f"(min {cfg.min_star_levels}) - not representative"
                )
            elif candidate.share_substantive < cfg.min_share_substantive:
                candidate.rejected_because = (
                    f"only {candidate.share_substantive:.0%} of reviews are substantive "
                    f"(min {cfg.min_share_substantive:.0%})"
                )
            elif candidate.n_months < cfg.min_months_covered:
                candidate.rejected_because = (
                    f"spans {candidate.n_months} months (min {cfg.min_months_covered})"
                )
            elif candidate.share_recent < cfg.min_share_recent:
                candidate.rejected_because = (
                    f"only {candidate.share_recent:.0%} recent reviews "
                    f"(min {cfg.min_share_recent:.0%})"
                )
        return list(candidates)

    # -- selection ---------------------------------------------------------

    def select(
        self,
        stats: pd.DataFrame,
        apps: Mapping[str, App],
        *,
        recent_shares: Mapping[str, float] | None = None,
    ) -> SelectionResult:
        """Score, filter, then greedily pick a category-diverse set."""
        cfg = self.config
        candidates = self.build_candidates(stats, apps, recent_shares=recent_shares)
        result = SelectionResult(considered=len(candidates), config=cfg)
        if not candidates:
            result.warnings.append("no candidate apps found in the review dataset")
            return result

        scored = self.score(candidates)
        self.apply_filters(scored)

        eligible = [c for c in scored if c.eligible]
        result.rejected = [c for c in scored if not c.eligible]

        if len(eligible) < cfg.min_apps:
            result.warnings.append(
                f"only {len(eligible)} app(s) passed the filters, below min_apps "
                f"({cfg.min_apps}); relaxing category diversity to fill the demo"
            )

        target = max(cfg.min_apps, min(cfg.n_apps, cfg.max_apps, len(eligible)))
        result.selected = self._greedy_diverse(eligible, target)

        if len(result.selected) < cfg.min_apps:
            result.warnings.append(
                f"selected {len(result.selected)} app(s), fewer than min_apps "
                f"({cfg.min_apps}); loosen the filters or check the dataset"
            )
        log.info("selection: %s", result.summary_line())
        return result

    def _greedy_diverse(
        self, eligible: Sequence[AppCandidate], target: int
    ) -> list[AppCandidate]:
        """Pinned apps first, then best-scoring app from each unused category.

        Two passes so that diversity is satisfied before quality doubles up:
        pass one takes the top app per category, pass two fills remaining slots
        from what is left, still respecting `max_per_category`.
        """
        cfg = self.config
        selected: list[AppCandidate] = []
        per_category: dict[str, int] = {}

        def take(candidate: AppCandidate) -> None:
            selected.append(candidate)
            per_category[candidate.category] = per_category.get(candidate.category, 0) + 1

        by_id = {c.app_id: c for c in eligible}
        for app_id in cfg.include_app_ids:
            pinned = by_id.get(app_id)
            if pinned and pinned not in selected:
                take(pinned)
                pinned.reasons.insert(0, "pinned via include_app_ids")

        remaining = [c for c in eligible if c not in selected]

        if cfg.prefer_distinct_categories:
            for candidate in remaining:
                if len(selected) >= target:
                    break
                if per_category.get(candidate.category, 0) == 0:
                    take(candidate)

        for candidate in remaining:
            if len(selected) >= target:
                break
            if candidate in selected:
                continue
            if per_category.get(candidate.category, 0) >= cfg.max_per_category:
                continue
            take(candidate)

        return sorted(selected, key=lambda c: c.score, reverse=True)


def _looks_quota_capped(row: pd.Series) -> bool:
    """Detect a manufactured star distribution from the aggregate row."""
    counts = [float(row.get(f"n_score_{s}", 0) or 0) for s in range(1, 6)]
    present = [c for c in counts if c > 0]
    if len(present) < 3 or sum(present) < 50:
        return len(present) <= 2 and sum(present) >= 50
    mean = float(np.mean(present))
    return mean > 0 and float(np.std(present)) / mean < 0.02
