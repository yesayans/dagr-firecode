"""End-to-end: the analysis pipeline and the demo precompute service.

Runs entirely offline - `fixture` embeddings and a stub LLM - so the full
orchestration is exercised in milliseconds with no network and no API key.
"""

from __future__ import annotations

import pytest

from aipm.analysis.needs import (
    ClusterContext,
    HeuristicNeedExtractor,
    LlmNeedExtractor,
    NeedService,
    build_need_extractor,
)
from aipm.analysis.pipeline import AnalysisPipeline, PipelineConfig
from aipm.clustering.cluster import ClusteringConfig
from aipm.config import Settings
from aipm.demo.precompute import DemoPrecomputeService
from aipm.demo.selection import DemoSelectionConfig
from aipm.llm.client import NullLlmClient
from aipm.preprocess.pipeline import PreprocessConfig, ReviewPreprocessor
from aipm.schemas import AnalysisParams, Cluster, RunStatus
from aipm.storage.sqlite_repo import SqliteRepository
from tests.conftest import StubLlmClient, insight_json

WEIGHTS = Settings().confidence_weights()


def build_pipeline(embedding_service, extractor, **config_overrides) -> AnalysisPipeline:
    return AnalysisPipeline(
        preprocessor=ReviewPreprocessor(PreprocessConfig(min_segment_tokens=3)),
        embedding_service=embedding_service,
        need_service=NeedService(extractor, confidence_weights=WEIGHTS),
        params=AnalysisParams(umap_n_components=4, random_state=42),
        config=PipelineConfig(n_representatives=4, max_clusters_to_label=4,
                              **config_overrides),
        clustering_config=ClusteringConfig(min_cluster_size=3,
                                           min_clusters_before_fallback=2),
    )


class TestAnalysisPipeline:
    def test_produces_a_complete_run(self, app, reviews, embedding_service):
        pipeline = build_pipeline(embedding_service, HeuristicNeedExtractor())
        result = pipeline.run(app, reviews)

        assert result.run.status is RunStatus.COMPLETE
        assert result.run.n_reviews > 0
        assert result.run.n_units > 0
        assert result.stats.n_reviews == result.run.n_reviews
        assert result.trends

    def test_stats_populated_without_any_llm(self, app, reviews, embedding_service):
        """The dashboard must render even when the model endpoint is down."""
        pipeline = build_pipeline(embedding_service, HeuristicNeedExtractor())
        result = pipeline.run(app, reviews)
        assert result.stats.avg_score > 0
        assert sum(result.stats.score_distribution.values()) > 0
        assert result.stats.helpful_votes.total > 0
        assert result.stats.store_score == 4.2

    def test_progress_callback_reports_every_stage(self, app, reviews, embedding_service):
        stages: list[str] = []
        build_pipeline(embedding_service, HeuristicNeedExtractor()).run(
            app, reviews, progress=lambda stage, _detail: stages.append(stage)
        )
        for expected in ("preprocess", "embed", "reduce", "cluster", "stats", "needs"):
            assert expected in stages

    def test_projection_rows_shaped_for_the_scatter(self, app, reviews, embedding_service):
        result = build_pipeline(embedding_service, HeuristicNeedExtractor()).run(app, reviews)
        assert result.projection
        assert {"x", "y", "cluster", "review_id"} <= set(result.projection[0])

    def test_empty_reviews_still_returns_a_result(self, app, embedding_service):
        result = build_pipeline(embedding_service, HeuristicNeedExtractor()).run(app, [])
        assert result.run.status is RunStatus.COMPLETE
        assert result.needs == []

    def test_llm_extractor_is_called_per_cluster(self, app, reviews, embedding_service):
        client = StubLlmClient([insight_json() for _ in range(20)])
        result = build_pipeline(embedding_service, LlmNeedExtractor(client)).run(app, reviews)
        assert client.calls
        assert len(result.needs) == len(client.calls)

    def test_run_id_is_deterministic(self, app, reviews, embedding_service):
        a = build_pipeline(embedding_service, HeuristicNeedExtractor()).run(app, reviews)
        b = build_pipeline(embedding_service, HeuristicNeedExtractor()).run(app, reviews)
        assert a.run.run_id == b.run.run_id

    def test_relevance_guard_disabled_for_non_semantic_backend(
        self, app, reviews, embedding_service
    ):
        """Fixture vectors are noise; scoring paraphrase relevance on them is invalid."""
        assert embedding_service.supports_semantic_similarity is False
        result = build_pipeline(embedding_service, HeuristicNeedExtractor()).run(app, reviews)
        # Citations survive on existence + membership alone.
        assert any(n.evidence for n in result.needs)


class TestNeedService:
    def _context(self, reviews, cluster_id="c1") -> ClusterContext:
        return ClusterContext(
            cluster=Cluster(cluster_id=cluster_id, run_id="run1", size=len(reviews),
                            keywords=["crash", "order"], cohesion=0.7, separation=0.6),
            units=[],
            reviews=reviews,
            representatives=[(r.review_id, r.score, r.text) for r in reviews[:3]],
            n_units_max=10,
        )

    def test_numbers_come_from_python_not_the_model(self, app, reviews):
        """A model claiming 0.99 confidence must not move the displayed score."""
        cited = [r.review_id for r in reviews[:2]]
        confident = insight_json(evidence_strength=0.99, cited_review_ids=cited)
        modest = insight_json(evidence_strength=0.01, cited_review_ids=cited)

        totals = []
        for payload in (confident, modest):
            service = NeedService(
                LlmNeedExtractor(StubLlmClient([payload])), confidence_weights=WEIGHTS
            )
            report = service.build_needs(
                [self._context(reviews[:6])], app=app, run_id="run1", all_reviews=reviews
            )
            totals.append(report.needs[0].confidence.total)
        assert totals[0] == totals[1]

    def test_fabricated_citations_discard_the_need(self, app, reviews):
        service = NeedService(
            LlmNeedExtractor(StubLlmClient([insight_json(cited_review_ids=["ghost1"])])),
            confidence_weights=WEIGHTS,
        )
        report = service.build_needs(
            [self._context(reviews[:6])], app=app, run_id="run1", all_reviews=reviews
        )
        assert report.needs == []
        assert report.n_needs_discarded == 1

    def test_extraction_failure_does_not_abort_the_batch(self, app, reviews):
        client = StubLlmClient(["garbage", "still garbage", insight_json()])
        service = NeedService(LlmNeedExtractor(client), confidence_weights=WEIGHTS)
        report = service.build_needs(
            [self._context(reviews[:6], "c1"), self._context(reviews[6:], "c2")],
            app=app, run_id="run1", all_reviews=reviews,
        )
        assert report.n_clusters_failed == 1
        assert len(report.needs) == 1

    def test_needs_are_ranked_by_value(self, app, reviews):
        client = StubLlmClient([insight_json() for _ in range(3)])
        service = NeedService(LlmNeedExtractor(client), confidence_weights=WEIGHTS)
        report = service.build_needs(
            [self._context(reviews[:4], "c1"), self._context(reviews[4:8], "c2"),
             self._context(reviews[8:], "c3")],
            app=app, run_id="run1", all_reviews=reviews,
        )
        ranks = [n.priority.rank for n in report.needs]
        assert ranks == sorted(ranks)
        values = [n.priority.value_score for n in report.needs]
        assert values == sorted(values, reverse=True)

    def test_impact_measured_against_the_store_rating(self, app, reviews):
        """Using the quota-capped sample mean would make impact meaningless."""
        service = NeedService(
            LlmNeedExtractor(StubLlmClient([insight_json()])), confidence_weights=WEIGHTS
        )
        report = service.build_needs(
            [self._context([r for r in reviews if r.score == 1][:5])],
            app=app, run_id="run1", all_reviews=reviews,
        )
        assert report.needs[0].priority.impact == pytest.approx(3.2, abs=0.01)

    def test_usage_is_accumulated(self, app, reviews):
        client = StubLlmClient([insight_json(), insight_json()])
        service = NeedService(LlmNeedExtractor(client), confidence_weights=WEIGHTS)
        report = service.build_needs(
            [self._context(reviews[:4], "c1"), self._context(reviews[4:], "c2")],
            app=app, run_id="run1", all_reviews=reviews,
        )
        assert report.usage.n_calls == 2
        assert report.usage.total_tokens == 60


class TestExtractorSelection:
    def test_falls_back_to_heuristic_without_an_endpoint(self):
        assert isinstance(build_need_extractor(NullLlmClient("no key")), HeuristicNeedExtractor)

    def test_require_llm_raises_instead(self):
        from aipm.llm.client import LlmError

        with pytest.raises(LlmError):
            build_need_extractor(NullLlmClient("no key"), allow_heuristic=False)


class TestDemoPrecomputeService:
    def _service(self, tmp_path, embedding_service, extractor) -> tuple:
        repo = SqliteRepository(tmp_path / "demo.db")
        repo.init_schema()
        service = DemoPrecomputeService(
            repository=repo, pipeline=build_pipeline(embedding_service, extractor)
        )
        return service, repo

    def test_persists_a_readable_result(self, tmp_path, app, reviews, embedding_service):
        service, repo = self._service(tmp_path, embedding_service, HeuristicNeedExtractor())
        outcome = service.precompute_app(app, reviews)
        assert outcome.ok

        stored = repo.get_latest_result(app.app_id)
        assert stored is not None
        assert stored.run.run_id == outcome.entry.run_id
        repo.close()

    def test_full_review_text_is_stored_for_drilldown(
        self, tmp_path, app, reviews, embedding_service
    ):
        """The Evidence page promises whole reviews, not truncated quotes."""
        service, repo = self._service(tmp_path, embedding_service, HeuristicNeedExtractor())
        service.precompute_app(app, reviews)
        stored = {r.review_id: r for r in repo.get_reviews(app.app_id)}
        original = next(r for r in reviews if r.review_id == "r0_0")
        assert stored["r0_0"].text == original.text
        repo.close()

    def test_no_reviews_is_a_recorded_failure_not_a_crash(
        self, tmp_path, app, embedding_service
    ):
        service, repo = self._service(tmp_path, embedding_service, HeuristicNeedExtractor())
        outcome = service.precompute_app(app, [])
        assert not outcome.ok and "no reviews" in outcome.entry.error
        repo.close()

    def test_pipeline_exception_is_captured_per_app(
        self, tmp_path, app, reviews, embedding_service
    ):
        service, repo = self._service(tmp_path, embedding_service, HeuristicNeedExtractor())

        def explode(*_args, **_kwargs):
            raise RuntimeError("clustering exploded")

        # The service calls `run_with_reviews`, not `run`.
        service.pipeline.run_with_reviews = explode
        outcome = service.precompute_app(app, reviews)
        assert not outcome.ok and "clustering exploded" in outcome.entry.error
        repo.close()

    def test_second_run_reuses_the_cached_result(
        self, tmp_path, app, reviews, embedding_service
    ):
        service, repo = self._service(tmp_path, embedding_service, HeuristicNeedExtractor())
        first = service.precompute_app(app, reviews)
        second = service.precompute_app(app, reviews)
        assert not first.skipped and second.skipped
        assert first.entry.run_id == second.entry.run_id
        repo.close()

    def test_force_bypasses_the_cache(self, tmp_path, app, reviews, embedding_service):
        service, repo = self._service(tmp_path, embedding_service, HeuristicNeedExtractor())
        service.precompute_app(app, reviews)
        assert not service.precompute_app(app, reviews, force=True).skipped
        repo.close()

    def test_manifest_aggregates_the_batch(self, tmp_path, app, reviews, embedding_service):
        service, repo = self._service(tmp_path, embedding_service, HeuristicNeedExtractor())
        outcome = service.precompute_app(app, reviews)
        manifest = service.build_manifest(
            [outcome], selection_config=DemoSelectionConfig(),
            embed_backend="fixture", embed_model="fixture-32", embed_dim=32,
            llm_model="none", llm_enabled=False, duration_s=1.0,
        )
        service.save_manifest(manifest)

        stored = repo.get_demo_manifest()
        assert stored.n_apps == 1
        assert stored.total_reviews == outcome.entry.n_reviews
        assert stored.llm_enabled is False
        repo.close()

    def test_failed_apps_surface_in_manifest_warnings(self, tmp_path, app, embedding_service):
        service, repo = self._service(tmp_path, embedding_service, HeuristicNeedExtractor())
        outcome = service.precompute_app(app, [])
        manifest = service.build_manifest(
            [outcome], selection_config=DemoSelectionConfig(),
            embed_backend="fixture", embed_model="m", embed_dim=32,
            llm_model="none", llm_enabled=False, duration_s=1.0,
        )
        assert any(app.name in w for w in manifest.warnings)
        repo.close()


class TestLlmProducesNoNumbers:
    """The central credibility rule: Python computes every metric.

    The LLM contributes language and reasoning only. These tests fail loudly if
    a model-supplied value ever reaches a numeric field.
    """

    def _context(self, reviews, cluster_id="c1") -> ClusterContext:
        return ClusterContext(
            cluster=Cluster(cluster_id=cluster_id, run_id="run1", size=len(reviews),
                            keywords=["crash", "order"], cohesion=0.7, separation=0.6),
            units=[],
            reviews=reviews,
            representatives=[(r.review_id, r.score, r.text) for r in reviews[:3]],
            n_units_max=10,
        )

    def _need(self, app, reviews, payload):
        service = NeedService(
            LlmNeedExtractor(StubLlmClient([payload])), confidence_weights=WEIGHTS
        )
        report = service.build_needs(
            [self._context(reviews[:6])], app=app, run_id="run1", all_reviews=reviews
        )
        return report.needs[0]

    def test_model_confidence_claim_is_ignored_entirely(self, app, reviews):
        cited = [r.review_id for r in reviews[:2]]
        high = self._need(app, reviews, insight_json(evidence_strength=1.0,
                                                     cited_review_ids=cited))
        low = self._need(app, reviews, insight_json(evidence_strength=0.0,
                                                    cited_review_ids=cited))
        for field_name in ("support", "cohesion", "separation", "temporal",
                           "diversity", "grounding", "total"):
            assert getattr(high.confidence, field_name) == getattr(low.confidence, field_name)

    def test_every_numeric_field_is_reproducible_from_inputs(self, app, reviews):
        """Recomputing from measured inputs must reproduce the stored numbers."""
        from aipm.analysis.confidence import ConfidenceInputs, compute_confidence
        from aipm.analysis.trends import months_covered, temporal_spread

        context = self._context(reviews[:6])
        cited = [r.review_id for r in reviews[:2]]
        need = self._need(app, reviews, insight_json(cited_review_ids=cited))

        expected = compute_confidence(
            ConfidenceInputs(
                n_units=len(context.units), n_units_max=context.n_units_max,
                cohesion=context.cluster.cohesion, separation=context.cluster.separation,
                temporal=temporal_spread(context.reviews, reviews),
                duplicate_share=context.duplicate_share,
                n_citations_offered=len(cited),
                n_citations_validated=sum(1 for e in need.evidence if e.validated),
                n_months_present=months_covered(context.reviews),
            ),
            WEIGHTS,
        )
        assert need.confidence.total == expected.total
        for component in ("support", "cohesion", "separation", "temporal",
                          "diversity", "grounding"):
            assert getattr(need.confidence, component) == getattr(expected, component)

    def test_rationale_is_carried_but_kept_out_of_the_number(self, app, reviews):
        need = self._need(
            app, reviews,
            insight_json(confidence_rationale="Users describe the same workaround.",
                         cited_review_ids=[r.review_id for r in reviews[:2]]),
        )
        assert need.confidence.llm_rationale == "Users describe the same workaround."
        # The computed explanation is assembled separately, from the components.
        assert "supporting review segment" in need.confidence.explanation
        assert need.confidence.explanation != need.confidence.llm_rationale

    def test_computed_explanation_agrees_with_the_components(self, app, reviews):
        need = self._need(
            app, reviews,
            insight_json(cited_review_ids=[r.review_id for r in reviews[:2]]),
        )
        assert need.confidence.band in need.confidence.explanation.lower()

    def test_priority_metrics_are_computed_not_supplied(self, app, reviews):
        """PriorityScore has no LLM-writable field; the schema has no route in."""
        need = self._need(app, reviews, insight_json())
        assert need.priority.confidence == need.confidence.total
        assert need.priority.value_score == pytest.approx(
            round(need.priority.reach * need.priority.impact * need.confidence.total, 6)
        )

    def test_insight_schema_exposes_exactly_one_number(self, app):
        """A regression here means someone gave the model a new numeric field."""
        from aipm.llm.prompts.cluster_need import ClusterInsight

        numeric = {
            name for name, f in ClusterInsight.model_fields.items()
            if f.annotation in (float, int)
        }
        assert numeric == {"evidence_strength"}

    def test_cluster_labels_use_the_model_title_not_the_need_statement(
        self, app, reviews, embedding_service
    ):
        payload = insight_json(title="Orders vanish after submission",
                               summary="Users lose the order and the cart.")
        client = StubLlmClient([payload for _ in range(20)])
        result = build_pipeline(embedding_service, LlmNeedExtractor(client)).run(app, reviews)
        labelled = [c for c in result.clusters if c.label]
        assert labelled
        assert labelled[0].label == "Orders vanish after submission"
        assert labelled[0].summary == "Users lose the order and the cart."


class TestPreparedReviewPersistence:
    def test_run_with_reviews_returns_annotated_reviews(
        self, app, reviews, embedding_service
    ):
        """Raw reviews carry no quality weight; the stored ones must."""
        pipeline = build_pipeline(embedding_service, HeuristicNeedExtractor())
        result, prepared = pipeline.run_with_reviews(app, reviews)

        assert result.run.status is RunStatus.COMPLETE
        assert prepared
        assert all(r.lang is not None for r in prepared)
        assert any(r.quality_weight != 1.0 for r in prepared)
        assert any(r.is_duplicate for r in prepared)

    def test_run_delegates_and_keeps_its_contract(self, app, reviews, embedding_service):
        pipeline = build_pipeline(embedding_service, HeuristicNeedExtractor())
        assert pipeline.run(app, reviews).run.run_id == (
            pipeline.run_with_reviews(app, reviews)[0].run.run_id
        )

    def test_empty_input_returns_a_pair(self, app, embedding_service):
        result, prepared = build_pipeline(
            embedding_service, HeuristicNeedExtractor()
        ).run_with_reviews(app, [])
        assert result.run.status is RunStatus.COMPLETE and prepared == []

    def test_precompute_stores_the_annotated_reviews(
        self, tmp_path, app, reviews, embedding_service
    ):
        repo = SqliteRepository(tmp_path / "demo.db")
        repo.init_schema()
        service = DemoPrecomputeService(
            repository=repo,
            pipeline=build_pipeline(embedding_service, HeuristicNeedExtractor()),
        )
        service.precompute_app(app, reviews)

        stored = repo.get_reviews(app.app_id)
        assert stored
        assert any(r.quality_weight != 1.0 for r in stored)
        assert all(r.lang for r in stored)
        repo.close()
