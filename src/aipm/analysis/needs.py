"""Cluster -> Need. Where the LLM's words and Python's numbers are combined.

Division of labour, enforced by construction:

* The **extractor** (LLM or heuristic) produces language only: a title, a
  summary, the surface complaint, the workaround, the latent need.
* This module produces every number: confidence, hiddenness, reach, impact,
  value. The extractor's `evidence_strength` is retained only to detect
  disagreement with the computed score, never to display.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from aipm.analysis import confidence as conf
from aipm.analysis.stats import affected_rating_gap
from aipm.analysis.trends import months_covered, temporal_spread
from aipm.llm.client import LlmClient, LlmError, Usage
from aipm.llm.guards import backfill_evidence, validate_citations
from aipm.llm.prompts.cluster_need import ClusterInsight, build_messages
from aipm.llm.structured import StructuredOutputError, generate_structured, schema_hint
from aipm.schemas import (
    App,
    Cluster,
    Need,
    NeedCategory,
    PriorityScore,
    Review,
    ReviewUnit,
)
from aipm.utils.hashing import stable_hash
from aipm.utils.logging import get_logger

log = get_logger(__name__)

#: Divergence above this between the model's self-assessment and the computed
#: confidence is logged. It is a prompt-quality signal, not a user-facing number.
_DIVERGENCE_ALERT = 0.35


@dataclass
class ClusterContext:
    """Everything known about one cluster before language is attached to it."""

    cluster: Cluster
    units: list[ReviewUnit]
    reviews: list[Review]
    representatives: list[tuple[str, int | None, str]]  # (review_id, stars, text)
    n_units_max: int
    duplicate_share: float = 0.0

    @property
    def review_ids(self) -> set[str]:
        return {r.review_id for r in self.reviews}


class NeedExtractor(ABC):
    """Produces the *language* for one cluster. Never a displayed number."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def extract(self, context: ClusterContext, app: App) -> tuple[ClusterInsight, Usage]: ...


class LlmNeedExtractor(NeedExtractor):
    """Prompts the configured model, validating the response with pydantic."""

    def __init__(self, client: LlmClient, *, max_tokens: int = 1200) -> None:
        self.client = client
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"llm:{self.client.model}"

    def extract(self, context: ClusterContext, app: App) -> tuple[ClusterInsight, Usage]:
        messages = build_messages(
            app_name=app.name,
            app_category=app.categories[0] if app.categories else "Uncategorised",
            keywords=context.cluster.keywords,
            samples=context.representatives,
            schema=schema_hint(ClusterInsight),
        )
        return generate_structured(
            self.client,
            ClusterInsight,
            messages,
            max_tokens=self.max_tokens,
            context=f"cluster {context.cluster.cluster_id}",
        )


class HeuristicNeedExtractor(NeedExtractor):
    """Keyword-driven fallback used when no LLM endpoint is configured.

    Produces honest, obviously-templated language. It exists so the pipeline and
    the dashboard still run end-to-end offline; its output is labelled so nobody
    mistakes it for model-written analysis.
    """

    _THEMES: tuple[tuple[NeedCategory, tuple[str, ...], str], ...] = (
        (NeedCategory.RELIABILITY, ("crash", "bug", "freeze", "error", "broken", "glitch"),
         "the app to keep working through the task they started"),
        (NeedCategory.PERFORMANCE, ("slow", "lag", "load", "speed", "wait", "stuck"),
         "the app to respond fast enough to finish a task in one sitting"),
        (NeedCategory.USABILITY, ("confusing", "hard", "find", "navigate", "interface", "update"),
         "to complete a task without hunting for the control that does it"),
        (NeedCategory.TRUST_PRIVACY, ("account", "login", "password", "verify", "privacy",
                                      "security", "personal"),
         "to trust the app with their account and their data"),
        (NeedCategory.PRICING, ("price", "charge", "refund", "subscription", "fee", "money"),
         "to understand what they are being charged before they are charged"),
        (NeedCategory.SUPPORT, ("support", "help", "contact", "response", "service"),
         "a way to reach a resolution when something goes wrong"),
        (NeedCategory.FEATURE_GAP, ("option", "feature", "add", "missing", "cannot", "wish"),
         "a capability the product does not currently offer"),
    )

    @property
    def name(self) -> str:
        return "heuristic"

    def extract(self, context: ClusterContext, app: App) -> tuple[ClusterInsight, Usage]:
        keywords = context.cluster.keywords or []
        blob = " ".join(keywords + [t for _, _, t in context.representatives]).lower()

        category, goal = NeedCategory.OTHER, "to get their task done without friction"
        for candidate, markers, candidate_goal in self._THEMES:
            if any(marker in blob for marker in markers):
                category, goal = candidate, candidate_goal
                break

        headline = ", ".join(keywords[:4]) if keywords else "recurring issue"
        sample = context.representatives[0][2] if context.representatives else ""
        # Fold the cluster's own keywords into the statement. A purely templated
        # sentence shares no vocabulary with the reviews it claims to summarise,
        # so the relevance guard would (correctly) reject all of its citations.
        grounding = f" (recurring themes: {headline})" if keywords else ""
        return (
            ClusterInsight(
                title=f"[heuristic] {headline}"[:80],
                summary=(
                    f"Reviews in this cluster share the terms {headline}. "
                    "Generated without an LLM - treat as a starting point, not analysis."
                ),
                surface_complaint=sample[:200] or headline,
                workaround="",
                hidden_need=f"Users need {goal}{grounding}",
                underlying_goal=goal,
                category=category,
                evidence_strength=0.5,
                confidence_rationale="Heuristic extraction; no model assessment available.",
                cited_review_ids=[rid for rid, _, _ in context.representatives[:3]],
            ),
            Usage(),
        )


@dataclass
class NeedExtractionReport:
    needs: list[Need] = field(default_factory=list)
    #: The raw model output per cluster id. The pipeline uses the purpose-written
    #: `title` and `summary` to label clusters; without this they would be paid
    #: for and discarded.
    insights: dict[str, ClusterInsight] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    citations_dropped: int = 0
    n_clusters_failed: int = 0
    n_needs_discarded: int = 0
    extractor: str = ""
    warnings: list[str] = field(default_factory=list)


class NeedService:
    """Turns clusters into ranked, evidenced, scored needs."""

    def __init__(
        self,
        extractor: NeedExtractor,
        *,
        confidence_weights: Mapping[str, float],
        citation_threshold: float = 0.30,
        discard_ungrounded: bool = True,
        discard_on_relevance_only: bool = False,
    ) -> None:
        self.extractor = extractor
        self.confidence_weights = dict(confidence_weights)
        self.citation_threshold = citation_threshold
        #: Drop needs whose citations were fabricated (unknown id, or a real id
        #: pulled from outside the cluster). This is the anti-hallucination guard.
        self.discard_ungrounded = discard_ungrounded
        #: Whether to also drop needs whose citations were all real and
        #: in-cluster but scored below the relevance threshold. Off by default:
        #: that is a *weak* need, not an invented one, and the grounding
        #: component already drives its confidence down. Discarding it would
        #: throw away a real cluster because a paraphrase scored 0.28.
        self.discard_on_relevance_only = discard_on_relevance_only

    def build_needs(
        self,
        contexts: Sequence[ClusterContext],
        *,
        app: App,
        run_id: str,
        all_reviews: Sequence[Review],
        embed_text: "callable[[list[str]], np.ndarray] | None" = None,
        review_vectors: Mapping[str, np.ndarray] | None = None,
    ) -> NeedExtractionReport:
        report = NeedExtractionReport(extractor=self.extractor.name)
        baseline = app.score if app.score is not None else _mean_score(all_reviews)
        n_total_reviews = max(1, len(all_reviews))

        for context in contexts:
            try:
                insight, usage = self.extractor.extract(context, app)
                report.usage.add(usage)
            except (StructuredOutputError, LlmError) as exc:
                report.n_clusters_failed += 1
                report.usage.n_failures += 1
                message = (
                    f"cluster {context.cluster.cluster_id}: extraction failed "
                    f"({type(exc).__name__}: {exc})"
                )
                log.warning(message)
                report.warnings.append(message)
                continue

            need = self._assemble(
                insight,
                context,
                app=app,
                run_id=run_id,
                all_reviews=all_reviews,
                baseline=baseline,
                n_total_reviews=n_total_reviews,
                embed_text=embed_text,
                review_vectors=review_vectors,
                report=report,
            )
            if need is not None:
                report.needs.append(need)
                report.insights[context.cluster.cluster_id] = insight

        self._rank(report.needs)
        log.info(
            "needs: %d built, %d discarded, %d cluster(s) failed, %d citation(s) dropped",
            len(report.needs), report.n_needs_discarded,
            report.n_clusters_failed, report.citations_dropped,
        )
        return report

    def _assemble(
        self,
        insight: ClusterInsight,
        context: ClusterContext,
        *,
        app: App,
        run_id: str,
        all_reviews: Sequence[Review],
        baseline: float,
        n_total_reviews: int,
        embed_text,
        review_vectors,
        report: NeedExtractionReport,
    ) -> Need | None:
        reviews_by_id = {r.review_id: r for r in context.reviews}

        need_vector = None
        if embed_text is not None and review_vectors is not None:
            try:
                need_vector = embed_text([insight.hidden_need])[0]
            except Exception as exc:  # embedding the need is a nicety, not a gate
                log.warning("could not embed need statement (%s); skipping relevance check", exc)

        audit = validate_citations(
            insight.cited_review_ids,
            need_statement=insight.hidden_need,
            cluster_review_ids=context.review_ids,
            reviews_by_id=reviews_by_id,
            need_vector=need_vector,
            review_vectors=review_vectors,
            threshold=self.citation_threshold,
        )
        report.citations_dropped += audit.n_dropped

        if self.discard_ungrounded and audit.n_validated == 0 and insight.cited_review_ids:
            # Distinguish fabrication from weak paraphrase similarity. An unknown
            # or out-of-cluster id means the model invented the citation, which is
            # exactly what this guard exists to catch. Everything failing purely on
            # relevance means the need is weakly worded, not invented - keep it and
            # let the grounding component push its confidence down.
            fabricated = audit.n_unknown_id + audit.n_out_of_cluster > 0
            if fabricated or self.discard_on_relevance_only:
                report.n_needs_discarded += 1
                log.info(
                    "discarded need from cluster %s: %s (%s)",
                    context.cluster.cluster_id,
                    "fabricated citations" if fabricated else "no citation cleared relevance",
                    audit.reason_summary(),
                )
                return None
            log.info(
                "cluster %s: no citation cleared relevance (%s); keeping as a "
                "low-confidence hypothesis",
                context.cluster.cluster_id, audit.reason_summary(),
            )

        audit = backfill_evidence(
            audit,
            cluster_review_ids=[r.review_id for r in context.reviews],
            reviews_by_id=reviews_by_id,
            target=3,
        )

        n_months = months_covered(context.reviews)
        breakdown = conf.compute_confidence(
            conf.ConfidenceInputs(
                n_units=len(context.units),
                n_units_max=context.n_units_max,
                cohesion=context.cluster.cohesion,
                separation=context.cluster.separation,
                temporal=temporal_spread(context.reviews, all_reviews),
                duplicate_share=context.duplicate_share,
                n_citations_offered=audit.n_offered,
                n_citations_validated=audit.n_validated,
                n_months_present=n_months,
            ),
            self.confidence_weights,
        )
        # The model's qualitative reasoning rides alongside the computed
        # explanation; it never influences `total` or any component.
        breakdown = breakdown.model_copy(
            update={"llm_rationale": insight.confidence_rationale.strip()}
        )

        divergence = abs(insight.evidence_strength - breakdown.total)
        if divergence > _DIVERGENCE_ALERT:
            log.info(
                "cluster %s: model evidence_strength %.2f vs computed confidence %.2f "
                "(divergence %.2f)",
                context.cluster.cluster_id, insight.evidence_strength,
                breakdown.total, divergence,
            )

        texts = [r.text for r in context.reviews]
        hiddenness = conf.hiddenness(
            conf.count_explicit_requests(texts), len(texts), cross_cluster=False
        )

        reach = round(len(context.reviews) / n_total_reviews, 4)
        impact = affected_rating_gap(context.reviews, baseline=baseline)
        priority = PriorityScore(
            reach=reach,
            impact=impact,
            confidence=breakdown.total,
            value_score=round(reach * impact * breakdown.total, 6),
        )

        return Need(
            need_id=f"need_{stable_hash([run_id, context.cluster.cluster_id])}",
            run_id=run_id,
            statement=insight.hidden_need,
            underlying_goal=insight.underlying_goal,
            category=insight.category,
            surface_complaints=[insight.surface_complaint] if insight.surface_complaint else [],
            workarounds=[insight.workaround] if insight.workaround.strip() else [],
            cluster_ids=[context.cluster.cluster_id],
            evidence=audit.evidence,
            confidence=breakdown,
            priority=priority,
            hiddenness=hiddenness,
        )

    @staticmethod
    def _rank(needs: list[Need]) -> None:
        """Rank by value. `insight_score` drives the Needs page ordering separately."""
        needs.sort(key=lambda n: n.priority.value_score, reverse=True)
        for position, need in enumerate(needs, start=1):
            need.priority.rank = position


def _mean_score(reviews: Sequence[Review]) -> float:
    scored = [r.score for r in reviews if r.score is not None]
    return sum(scored) / len(scored) if scored else 0.0


def build_need_extractor(client: LlmClient, *, allow_heuristic: bool = True) -> NeedExtractor:
    """Pick an extractor based on what is actually reachable."""
    if client.available:
        return LlmNeedExtractor(client)
    if not allow_heuristic:
        raise LlmError(getattr(client, "reason", "LLM unavailable and heuristic disabled"))
    log.warning(
        "no LLM available (%s); using heuristic extraction. Needs will be templated.",
        getattr(client, "reason", "unknown"),
    )
    return HeuristicNeedExtractor()
