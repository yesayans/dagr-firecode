"""The confidence model - the product's central credibility claim.

Six components, each computed from data, each 0..1, combined with configurable
weights. **The LLM contributes nothing here.** A model-invented confidence score
is exactly what makes a PM stop trusting an AI tool, so every term traces back to
something countable.

The plain-English `explanation` is assembled from the same numbers, so the
sentence under the meter can never disagree with the bar above it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from aipm.schemas import ConfidenceBreakdown


@dataclass(frozen=True)
class ConfidenceInputs:
    """Everything the model needs, all of it measured."""

    n_units: int
    n_units_max: int  # largest cluster in this run, for log-scaling support
    cohesion: float  # mean intra-cluster cosine similarity
    separation: float  # distance to the nearest other centroid
    temporal: float  # share of active months the theme appears in
    duplicate_share: float  # share of supporting reviews that are near-duplicates
    n_citations_offered: int
    n_citations_validated: int
    n_months_present: int = 0


def support_score(n_units: int, n_units_max: int) -> float:
    """Log-scaled volume.

    Linear scaling would let the biggest cluster flatten everything else; the
    difference between 20 and 200 supporting segments matters far more than the
    difference between 2000 and 2200.
    """
    if n_units <= 0 or n_units_max <= 0:
        return 0.0
    return min(1.0, math.log1p(n_units) / math.log1p(n_units_max))


def diversity_score(duplicate_share: float) -> float:
    """1 - near-duplicate share. Punishes evidence that is one text repeated."""
    return max(0.0, min(1.0, 1.0 - duplicate_share))


def grounding_score(n_offered: int, n_validated: int) -> float:
    """Share of the model's citations that survived validation.

    No citations offered scores 0, not 1: an unevidenced need is the case this
    component exists to catch.
    """
    if n_offered <= 0:
        return 0.0
    return max(0.0, min(1.0, n_validated / n_offered))


def compute_confidence(
    inputs: ConfidenceInputs, weights: dict[str, float]
) -> ConfidenceBreakdown:
    """Combine the six components into a weighted total plus its explanation."""
    components = {
        "support": support_score(inputs.n_units, inputs.n_units_max),
        "cohesion": max(0.0, min(1.0, inputs.cohesion)),
        "separation": max(0.0, min(1.0, inputs.separation)),
        "temporal": max(0.0, min(1.0, inputs.temporal)),
        "diversity": diversity_score(inputs.duplicate_share),
        "grounding": grounding_score(
            inputs.n_citations_offered, inputs.n_citations_validated
        ),
    }

    total_weight = sum(weights.get(name, 0.0) for name in components) or 1.0
    total = sum(components[name] * weights.get(name, 0.0) for name in components) / total_weight

    breakdown = ConfidenceBreakdown(
        **{name: round(value, 4) for name, value in components.items()},
        total=round(total, 4),
    )
    return breakdown.model_copy(
        update={"explanation": explain(breakdown, inputs)}
    )


def explain(breakdown: ConfidenceBreakdown, inputs: ConfidenceInputs) -> str:
    """One sentence a PM can read without a legend.

    Leads with the band, states the volume and time spread, then names whichever
    component is dragging the score down - the useful part when confidence is low.
    """
    band = breakdown.band
    parts = [f"{inputs.n_units} supporting review segment{'s' if inputs.n_units != 1 else ''}"]

    if inputs.n_months_present > 0:
        parts.append(
            f"across {inputs.n_months_present} month{'s' if inputs.n_months_present != 1 else ''}"
        )
    parts.append(_cohesion_phrase(breakdown.cohesion))
    if inputs.n_citations_offered > 0:
        parts.append(
            f"{inputs.n_citations_validated} of {inputs.n_citations_offered} "
            f"citations verified"
        )
    else:
        parts.append("no citations verified")

    sentence = f"{band.capitalize()} confidence: " + ", ".join(parts) + "."

    weakest = min(
        ("support", breakdown.support), ("cohesion", breakdown.cohesion),
        ("separation", breakdown.separation), ("temporal", breakdown.temporal),
        ("diversity", breakdown.diversity), ("grounding", breakdown.grounding),
        key=lambda pair: pair[1],
    )
    if weakest[1] < 0.35:
        sentence += f" Weakest signal: {weakest[0]} ({weakest[1]:.2f})."
    if inputs.duplicate_share > 0.25:
        sentence += (
            f" Note: {inputs.duplicate_share:.0%} of supporting reviews are near-duplicates."
        )
    return sentence


def _cohesion_phrase(cohesion: float) -> str:
    if cohesion >= 0.6:
        return "tightly clustered"
    if cohesion >= 0.35:
        return "moderately clustered"
    return "loosely clustered"


def hiddenness(
    n_explicit_requests: int, n_total_mentions: int, *, cross_cluster: bool = False
) -> float:
    """1 - (explicit feature requests / total mentions), boosted for cross-cluster needs.

    A need users are already asking for directly is not hidden - the PM has read
    it in the app store. A need that only shows up as scattered symptoms is.
    """
    if n_total_mentions <= 0:
        return 0.0
    base = 1.0 - (n_explicit_requests / n_total_mentions)
    if cross_cluster:
        # Spanning clusters means it is invisible to anyone reading linearly.
        base = base + (1.0 - base) * 0.25
    return round(max(0.0, min(1.0, base)), 4)


#: Phrases that mark a review as an explicit feature request rather than a symptom.
EXPLICIT_REQUEST_MARKERS = (
    "please add", "please make", "would be nice if", "wish there was", "wish it had",
    "should have", "should add", "needs a", "need an option", "feature request",
    "add a feature", "hope you add", "please include", "why isn't there",
    "why is there no", "no option to", "there should be",
)


def count_explicit_requests(texts: Sequence[str]) -> int:
    """Substring match, deliberately. A classifier here would be unauditable."""
    lowered = [t.lower() for t in texts]
    return sum(1 for t in lowered if any(marker in t for marker in EXPLICIT_REQUEST_MARKERS))
