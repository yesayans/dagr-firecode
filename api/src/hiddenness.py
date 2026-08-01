"""Hiddenness and insight scoring (ported from alternative `aipm.analysis.confidence`).

A need users already ask for directly is not hidden — the PM has seen it in the
store. A need that only shows up as scattered symptoms is. Ranking uses
`insight_score = hiddenness × (confidence / 100)` so loud explicit requests do
not dominate the demo board.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: Phrases that mark a review as an explicit feature request rather than a symptom.
EXPLICIT_REQUEST_MARKERS = (
    "please add",
    "please make",
    "would be nice if",
    "wish there was",
    "wish it had",
    "should have",
    "should add",
    "needs a",
    "need an option",
    "feature request",
    "add a feature",
    "hope you add",
    "please include",
    "why isn't there",
    "why is there no",
    "no option to",
    "there should be",
)


def count_explicit_requests(texts: Sequence[str]) -> int:
    """Substring match, deliberately. A classifier here would be unauditable."""
    lowered = [t.lower() for t in texts]
    return sum(
        1 for t in lowered if any(marker in t for marker in EXPLICIT_REQUEST_MARKERS)
    )


def hiddenness(
    n_explicit_requests: int,
    n_total_mentions: int,
    *,
    cross_cluster: bool = False,
) -> float:
    """1 - (explicit feature requests / total mentions), boosted for cross-cluster needs."""
    if n_total_mentions <= 0:
        return 0.0
    base = 1.0 - (n_explicit_requests / n_total_mentions)
    if cross_cluster:
        base = base + (1.0 - base) * 0.25
    return round(max(0.0, min(1.0, base)), 4)


def insight_score(hiddenness_value: float, confidence_0_100: float) -> float:
    """Product of hiddenness and normalised confidence (0–1 scale result)."""
    return round(float(hiddenness_value) * (float(confidence_0_100) / 100.0), 4)


def annotate_metrics_hiddenness(
    metrics: dict[str, Any],
    texts: Sequence[str],
    *,
    confidence_key: str = "deterministic_confidence",
    cross_cluster: bool = False,
) -> dict[str, Any]:
    """Mutate/return metrics with hiddenness, explicit_request_count, insight_score."""
    n_total = len(texts)
    n_explicit = count_explicit_requests(texts)
    h = hiddenness(n_explicit, n_total, cross_cluster=cross_cluster)
    conf = float(metrics.get(confidence_key) or metrics.get("deterministic_confidence") or 0.0)
    metrics["explicit_request_count"] = n_explicit
    metrics["mention_count"] = n_total
    metrics["hiddenness"] = h
    metrics["insight_score"] = insight_score(h, conf)
    return metrics


def refresh_insight_score(
    metrics: dict[str, Any], confidence_0_100: float
) -> dict[str, Any]:
    """Recompute insight_score after final (possibly blended) confidence is known."""
    h = float(metrics.get("hiddenness") or 0.0)
    metrics["insight_score"] = insight_score(h, confidence_0_100)
    return metrics
