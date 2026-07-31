"""The per-app analysis pipeline.

One injectable service that takes an app plus its raw reviews and returns the
`AnalysisResult` the Streamlit layer reads. Everything it needs is passed in at
construction, so a test can swap the embedding provider for `fixture` and the
extractor for the heuristic one and exercise the whole thing offline in
milliseconds.

The statistics branch never touches the LLM, deliberately: if the model endpoint
is unreachable, the run still produces a complete dashboard and simply reports
that the needs are missing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from aipm.analysis.needs import ClusterContext, NeedService
from aipm.analysis.stats import compute_overview_stats
from aipm.analysis.trends import compute_trends
from aipm.clustering.cluster import ClusteringConfig, cluster_embeddings
from aipm.clustering.keywords import extract_cluster_keywords
from aipm.clustering.metrics import centroid, cohesion, medoid_index, separation
from aipm.clustering.reduce import ReductionConfig, reduce_embeddings
from aipm.clustering.representatives import select_representatives
from aipm.embeddings.store import EmbeddingService
from aipm.preprocess.dedupe import duplicate_share
from aipm.preprocess.pipeline import ReviewPreprocessor
from aipm.schemas import (
    AnalysisParams,
    AnalysisResult,
    AnalysisRun,
    App,
    Cluster,
    Review,
    ReviewUnit,
    RunStatus,
)
from aipm.utils.hashing import stable_hash
from aipm.utils.logging import get_logger
from aipm.utils.timing import stage

log = get_logger(__name__)

ProgressFn = Callable[[str, str], None]


@dataclass(frozen=True)
class PipelineConfig:
    n_representatives: int = 10
    max_clusters_to_label: int = 12
    citation_threshold: float = 0.30
    max_trend_months: int | None = 60
    projection_sample_cap: int = 4000


class AnalysisPipeline:
    """preprocess -> embed -> reduce -> cluster -> characterise -> stats -> needs."""

    def __init__(
        self,
        *,
        preprocessor: ReviewPreprocessor,
        embedding_service: EmbeddingService,
        need_service: NeedService,
        params: AnalysisParams,
        config: PipelineConfig | None = None,
        clustering_config: ClusteringConfig | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.embeddings = embedding_service
        self.needs = need_service
        self.params = params
        self.config = config or PipelineConfig()
        self.clustering_config = clustering_config or ClusteringConfig(
            min_cluster_size=params.min_cluster_size,
            min_samples=params.min_samples,
            random_state=params.random_state,
        )

    def run(
        self,
        app: App,
        reviews: Sequence[Review],
        *,
        progress: ProgressFn | None = None,
    ) -> AnalysisResult:
        """Execute the full pipeline. Never raises for empty or degenerate input."""
        return self.run_with_reviews(app, reviews, progress=progress)[0]

    def run_with_reviews(
        self,
        app: App,
        reviews: Sequence[Review],
        *,
        progress: ProgressFn | None = None,
    ) -> tuple[AnalysisResult, list[Review]]:
        """Same run, but also returns the *preprocessed* reviews.

        Callers that persist reviews want these, not the raw input: only these
        carry `quality_weight`, `lang` and `is_duplicate`. Storing the raw list
        makes the evidence table advertise a quality column that is 1.00 for
        every row. Exposed as a second method so `run()` keeps its contract and
        nothing has to re-run the preprocessor to get the annotations.
        """
        emit = progress or (lambda _stage, _detail: None)
        started = datetime.now(timezone.utc)
        run_id = f"run_{stable_hash([app.app_id, self.params.params_hash(), len(reviews)])}"

        run = AnalysisRun(
            run_id=run_id,
            app_id=app.app_id,
            params_hash=self.params.params_hash(),
            params=self.params,
            status=RunStatus.RUNNING,
            started_at=started,
        )

        # 1. Preprocess -----------------------------------------------------
        emit("preprocess", f"{len(reviews):,} raw reviews")
        with stage(f"preprocess[{app.app_id}]"):
            prepared = self.preprocessor.run(reviews)

        if not prepared.clusterable_units:
            empty = self._empty_result(app, run, prepared.reviews, "no clusterable units")
            return empty, prepared.reviews

        # 2. Embed ----------------------------------------------------------
        emit("embed", f"{len(prepared.clusterable_units):,} units")
        with stage(f"embed[{app.app_id}]"):
            matrix = self.embeddings.embed_units(prepared.clusterable_units)

        # 3. Reduce ---------------------------------------------------------
        emit("reduce", f"{matrix.dim}D -> {self.params.umap_n_components}D")
        with stage(f"reduce[{app.app_id}]"):
            reduction = reduce_embeddings(
                matrix.vectors,
                ReductionConfig(
                    n_components=self.params.umap_n_components,
                    n_neighbors=self.params.umap_n_neighbors,
                    random_state=self.params.random_state,
                ),
            )

        # 4. Cluster --------------------------------------------------------
        emit("cluster", reduction.method)
        with stage(f"cluster[{app.app_id}]"):
            clustering = cluster_embeddings(reduction.embedding, self.clustering_config)

        # 5. Characterise ---------------------------------------------------
        emit("characterise", f"{clustering.n_clusters} clusters")
        clusters, contexts = self._characterise(
            run_id=run_id,
            units=prepared.clusterable_units,
            reviews_by_id=prepared.reviews_by_id(),
            vectors=matrix.vectors,
            clustering=clustering,
        )

        # 6. Stats and trends (no LLM) --------------------------------------
        emit("stats", f"{len(prepared.reviews):,} reviews")
        trends = compute_trends(prepared.reviews, max_months=self.config.max_trend_months)
        stats = compute_overview_stats(
            prepared.reviews,
            app=app,
            n_clusters=clustering.n_clusters,
            noise_ratio=clustering.noise_ratio,
            trends=trends,
        )

        # 7. Needs (LLM) ----------------------------------------------------
        emit("needs", f"{len(contexts)} clusters to label")
        # The relevance guard compares a paraphrased need statement against
        # review text, which only means something under a semantic embedding.
        # With a lexical backend the scores collapse toward zero and would
        # reject every valid citation, so the check is skipped rather than
        # trusted - existence and cluster-membership checks still apply.
        semantic = self.embeddings.supports_semantic_similarity
        if semantic:
            review_vectors = self._review_vectors(prepared.clusterable_units, matrix)
            embed_text = self.embeddings.embed_texts
        else:
            review_vectors, embed_text = None, None
            log.warning(
                "embedding backend '%s' is not semantic; citation relevance scoring is "
                "disabled for this run (existence and membership checks still apply)",
                self.embeddings.model,
            )

        with stage(f"needs[{app.app_id}]"):
            need_report = self.needs.build_needs(
                contexts,
                app=app,
                run_id=run_id,
                all_reviews=prepared.reviews,
                embed_text=embed_text,
                review_vectors=review_vectors,
            )

        # Attach the LLM-written label and summary back onto the clusters. These
        # are the model's purpose-written `title`/`summary` fields - the need
        # statement is a different artefact and reads badly as a cluster label.
        clusters = [
            c.model_copy(
                update={
                    "label": need_report.insights[c.cluster_id].title,
                    "summary": need_report.insights[c.cluster_id].summary,
                }
            )
            if c.cluster_id in need_report.insights
            else c
            for c in clusters
        ]

        run = run.model_copy(
            update={
                "status": RunStatus.COMPLETE,
                "n_reviews": len(prepared.reviews),
                "n_units": len(prepared.clusterable_units),
                "n_clusters": clustering.n_clusters,
                "noise_ratio": round(clustering.noise_ratio, 4),
                "clustering_fallback": clustering.fallback or reduction.fallback,
                "citations_dropped": need_report.citations_dropped,
                "cost_usd": round(need_report.usage.cost_usd, 6),
                "finished_at": datetime.now(timezone.utc),
            }
        )

        emit("done", f"{len(need_report.needs)} needs")
        result = AnalysisResult(
            run=run,
            app=app,
            stats=stats,
            trends=trends,
            clusters=clusters,
            needs=need_report.needs,
            projection=self._projection(
                prepared.clusterable_units, reduction.projection_2d, clustering.labels,
                prepared.reviews_by_id(),
            ),
        )
        return result, prepared.reviews

    # -- steps -------------------------------------------------------------

    def _characterise(
        self,
        *,
        run_id: str,
        units: Sequence[ReviewUnit],
        reviews_by_id: dict[str, Review],
        vectors: np.ndarray,
        clustering,
    ) -> tuple[list[Cluster], list[ClusterContext]]:
        """Keywords, geometry, medoid and representatives for every cluster."""
        indices_by_label = clustering.cluster_indices()
        if not indices_by_label:
            return [], []

        texts = [u.text for u in units]
        keywords_by_label = extract_cluster_keywords(
            texts, {label: idx.tolist() for label, idx in indices_by_label.items()}
        )
        centroids = {
            label: centroid(vectors[idx]) for label, idx in indices_by_label.items()
        }
        n_units_max = max(len(idx) for idx in indices_by_label.values())

        clusters: list[Cluster] = []
        contexts: list[ClusterContext] = []

        # Label the largest clusters first, and cap how many reach the LLM.
        ordered = sorted(indices_by_label.items(), key=lambda kv: len(kv[1]), reverse=True)

        for label, idx in ordered:
            member_units = [units[i] for i in idx]
            member_vectors = vectors[idx]
            cluster_id = f"c_{stable_hash([run_id, int(label)])}"

            others = [c for other_label, c in centroids.items() if other_label != label]
            medoid = medoid_index(member_vectors)

            member_reviews = _unique_reviews(member_units, reviews_by_id)
            weights = [
                reviews_by_id[u.review_id].helpful_count
                if u.review_id in reviews_by_id
                else 0.0
                for u in member_units
            ]
            rep_indices = select_representatives(
                member_vectors, n=self.config.n_representatives, weights=weights
            )

            cluster = Cluster(
                cluster_id=cluster_id,
                run_id=run_id,
                size=len(member_units),
                keywords=keywords_by_label.get(label, []),
                cohesion=round(cohesion(member_vectors), 4),
                separation=round(separation(centroids[label], others), 4),
                # sklearn's HDBSCAN does not expose cluster persistence; the field
                # stays 0.0 rather than being filled with a fabricated value.
                persistence=0.0,
                medoid_unit_id=member_units[medoid].unit_id if medoid >= 0 else None,
                member_unit_ids=[u.unit_id for u in member_units],
                representative_unit_ids=[member_units[i].unit_id for i in rep_indices],
            )
            clusters.append(cluster)

            if len(contexts) < self.config.max_clusters_to_label:
                contexts.append(
                    ClusterContext(
                        cluster=cluster,
                        units=member_units,
                        reviews=member_reviews,
                        representatives=[
                            (
                                member_units[i].review_id,
                                reviews_by_id[member_units[i].review_id].score
                                if member_units[i].review_id in reviews_by_id
                                else None,
                                member_units[i].text,
                            )
                            for i in rep_indices
                        ],
                        n_units_max=n_units_max,
                        duplicate_share=duplicate_share(member_reviews),
                    )
                )
        return clusters, contexts

    def _review_vectors(
        self, units: Sequence[ReviewUnit], matrix
    ) -> dict[str, np.ndarray]:
        """One vector per review: the mean of its units, renormalised.

        Citations point at reviews, but embeddings are per unit, so they have to
        be pooled before the relevance guard can compare them to a need.
        """
        grouped: dict[str, list[np.ndarray]] = {}
        for unit in units:
            grouped.setdefault(unit.review_id, []).append(matrix.row(unit.unit_id))
        out: dict[str, np.ndarray] = {}
        for review_id, rows in grouped.items():
            pooled = np.mean(rows, axis=0)
            norm = float(np.linalg.norm(pooled))
            out[review_id] = pooled / norm if norm > 1e-12 else pooled
        return out

    def _projection(
        self,
        units: Sequence[ReviewUnit],
        projection_2d: np.ndarray,
        labels: np.ndarray,
        reviews_by_id: dict[str, Review],
    ) -> list[dict]:
        """Rows for the UMAP scatter. Sampled - a 40k-point scatter helps nobody."""
        n = len(units)
        if n == 0 or len(projection_2d) != n:
            return []
        indices = range(n)
        cap = self.config.projection_sample_cap
        if n > cap:
            rng = np.random.default_rng(self.params.random_state)
            indices = sorted(rng.choice(n, cap, replace=False).tolist())
        rows = []
        for i in indices:
            unit = units[i]
            review = reviews_by_id.get(unit.review_id)
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "review_id": unit.review_id,
                    "x": float(projection_2d[i][0]),
                    "y": float(projection_2d[i][1]),
                    "cluster": int(labels[i]),
                    "score": review.score if review else None,
                    "helpful_count": review.helpful_count if review else 0,
                    "text": unit.text[:200],
                }
            )
        return rows

    def _empty_result(
        self, app: App, run: AnalysisRun, reviews: Sequence[Review], reason: str
    ) -> AnalysisResult:
        """A run with statistics but no clusters. Still renders a dashboard."""
        log.warning("app %s: %s", app.app_id, reason)
        trends = compute_trends(reviews, max_months=self.config.max_trend_months)
        return AnalysisResult(
            run=run.model_copy(
                update={
                    "status": RunStatus.COMPLETE,
                    "n_reviews": len(reviews),
                    "finished_at": datetime.now(timezone.utc),
                    "error": reason,
                }
            ),
            app=app,
            stats=compute_overview_stats(reviews, app=app, trends=trends),
            trends=trends,
        )


def _unique_reviews(
    units: Sequence[ReviewUnit], reviews_by_id: dict[str, Review]
) -> list[Review]:
    """Distinct reviews behind a set of units, order preserved."""
    seen: set[str] = set()
    out: list[Review] = []
    for unit in units:
        if unit.review_id in seen:
            continue
        seen.add(unit.review_id)
        review = reviews_by_id.get(unit.review_id)
        if review is not None:
            out.append(review)
    return out
