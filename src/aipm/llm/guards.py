"""Citation validation - the guard between "the model said it" and "we display it".

Three checks, cheapest first:

1. **Existence.** The cited review id must be real. Models hallucinate ids.
2. **Membership.** It must belong to the cluster the need came from. A real id
   pulled from elsewhere in the prompt is still a fabricated citation.
3. **Relevance.** cosine(need statement, review text) must clear a threshold.
   This catches the subtle case: a real, in-cluster review that does not actually
   support the claim.

The drop rate is recorded rather than hidden. It feeds the grounding component of
confidence and it is worth showing a sceptical reader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from aipm.schemas import Evidence, Review
from aipm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class CitationAudit:
    """Outcome of validating one need's citations."""

    evidence: list[Evidence] = field(default_factory=list)
    n_offered: int = 0
    n_unknown_id: int = 0
    n_out_of_cluster: int = 0
    n_below_threshold: int = 0

    @property
    def n_validated(self) -> int:
        return sum(1 for e in self.evidence if e.validated)

    @property
    def n_dropped(self) -> int:
        return self.n_unknown_id + self.n_out_of_cluster + self.n_below_threshold

    def reason_summary(self) -> str:
        return (
            f"{self.n_validated}/{self.n_offered} kept "
            f"(unknown id {self.n_unknown_id}, out of cluster {self.n_out_of_cluster}, "
            f"irrelevant {self.n_below_threshold})"
        )


def quote_from(review: Review, *, max_chars: int = 300) -> str:
    text = review.text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def validate_citations(
    cited_ids: Sequence[str],
    *,
    need_statement: str,
    cluster_review_ids: set[str],
    reviews_by_id: Mapping[str, Review],
    need_vector: np.ndarray | None = None,
    review_vectors: Mapping[str, np.ndarray] | None = None,
    threshold: float = 0.30,
) -> CitationAudit:
    """Validate the model's citations for one need.

    When no vectors are supplied the relevance check is skipped (and every
    surviving citation is marked validated); existence and membership still apply.
    """
    audit = CitationAudit(n_offered=len(cited_ids))
    seen: set[str] = set()

    for review_id in cited_ids:
        review_id = str(review_id).strip()
        if not review_id or review_id in seen:
            continue
        seen.add(review_id)

        review = reviews_by_id.get(review_id)
        if review is None:
            audit.n_unknown_id += 1
            continue
        if cluster_review_ids and review_id not in cluster_review_ids:
            audit.n_out_of_cluster += 1
            continue

        relevance = 1.0
        if need_vector is not None and review_vectors is not None:
            vector = review_vectors.get(review_id)
            if vector is None:
                audit.n_below_threshold += 1
                continue
            relevance = float(np.dot(need_vector, vector))
            if relevance < threshold:
                audit.n_below_threshold += 1
                continue

        audit.evidence.append(
            Evidence(
                review_id=review_id,
                quote=quote_from(review),
                relevance=round(relevance, 4),
                review_score=review.score,
                review_date=review.review_date,
                helpful_count=review.helpful_count,
                validated=True,
            )
        )

    if audit.n_dropped:
        log.info("citation guard: %s", audit.reason_summary())
    return audit


def backfill_evidence(
    audit: CitationAudit,
    *,
    cluster_review_ids: Sequence[str],
    reviews_by_id: Mapping[str, Review],
    target: int = 3,
) -> CitationAudit:
    """Top up thin evidence with the most-helpful in-cluster reviews.

    Backfilled items are marked `validated=False`: they are real, in-cluster
    reviews, but the model did not cite them, so they must not inflate the
    grounding score. The UI can show them as context.
    """
    if len(audit.evidence) >= target:
        return audit

    already = {e.review_id for e in audit.evidence}
    candidates = [
        reviews_by_id[rid]
        for rid in cluster_review_ids
        if rid in reviews_by_id and rid not in already
    ]
    candidates.sort(key=lambda r: (r.helpful_count, len(r.text)), reverse=True)

    for review in candidates[: target - len(audit.evidence)]:
        audit.evidence.append(
            Evidence(
                review_id=review.review_id,
                quote=quote_from(review),
                relevance=0.0,
                review_score=review.score,
                review_date=review.review_date,
                helpful_count=review.helpful_count,
                validated=False,
            )
        )
    return audit
