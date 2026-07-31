"""End-to-end analysis orchestration with stage callbacks."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from src.config import Settings, get_settings
from src.data_ingestion import ReviewScraper
from src.embedding_engine import EmbeddingEngine
from src.gap_analyzer import GapMatrix, compute_review_window
from src.llm_extractor import ExtractedGap, LatentNeedExtractor
from src.matching_space import build_matching_space
from src.need_filter import select_need_bearing
from src.resolver import RoadmapResolver
from src.store import Store

StageCallback = Callable[[str, int, dict[str, Any] | None], None]


def config_hash(app_id: str, max_reviews: int, settings: Settings) -> str:
    payload = {
        "app_id": app_id,
        "max_reviews": max_reviews,
        "embedding_backend": settings.embedding_backend,
        "match_threshold": settings.active_match_threshold(),
        "null_percentile": getattr(settings, "null_percentile", 95.0),
        "roadmap_matching_enabled": getattr(
            settings, "roadmap_matching_enabled", True
        ),
        "llm": settings.llm_enabled,
        "model": settings.openrouter_model,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class AnalysisPipeline:
    def __init__(
        self,
        store: Store,
        settings: Settings | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.resolver = RoadmapResolver(self.settings)
        self.reviews = ReviewScraper(self.settings)
        self.extractor = LatentNeedExtractor(self.settings)

    def run(self, job_id: str, app: dict[str, Any], max_reviews: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        degraded: list[str] = []
        llm_used = False
        backend_name = self.settings.embedding_backend
        total_reviews = 0
        n_clusters = 0
        n_roadmap = 0
        roadmap_source = "none"
        need_stats = {"reviews_total": 0, "reviews_need_bearing": 0}

        def cb(stage: str, progress: int, extra: dict[str, Any] | None = None) -> None:
            fields: dict[str, Any] = {
                "stage": stage,
                "progress": progress,
                "status": "failed" if stage == "failed" else "running",
            }
            if stage == "queued":
                fields["status"] = "queued"
            if extra:
                if "error" in extra:
                    fields["error"] = extra["error"]
                if "summary" in extra:
                    fields["summary"] = extra["summary"]
                if "stats" in extra:
                    fields["stats"] = extra["stats"]
                if "roadmap_snapshot" in extra:
                    fields["roadmap_snapshot"] = extra["roadmap_snapshot"]
                if "completed_at" in extra:
                    fields["completed_at"] = extra["completed_at"]
                if "status" in extra:
                    fields["status"] = extra["status"]
            self.store.update_job(job_id, fields)

        try:
            self.store.update_job(
                job_id,
                {
                    "status": "running",
                    "stage": "resolving_roadmap",
                    "progress": 5,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # --- resolve roadmap ---
            meta = app.get("metadata") or {}
            resolved = self.resolver.resolve(
                app_name=app["display_name"],
                package_name=app["package_name"],
                github_repo=app.get("github_repo"),
                refresh=bool(meta.get("force_roadmap_refresh")),
                external_roadmap_urls=list(meta.get("external_roadmap_urls") or []),
                external_roadmap_text=meta.get("external_roadmap_text") or "",
            )
            roadmap_source = resolved.roadmap_source
            degraded.extend(resolved.degraded)
            roadmap_items = resolved.roadmap_items
            n_roadmap = 0 if roadmap_items is None else len(roadmap_items)

            self.store.upsert_app(
                {
                    **app,
                    "github_repo": resolved.github_repo or app.get("github_repo"),
                    "roadmap_source": roadmap_source,
                    "roadmap_item_count": n_roadmap,
                    "metadata": {
                        **(app.get("metadata") or {}),
                        "roadmap_item_count": n_roadmap,
                    },
                }
            )

            cb(
                "fetching_reviews",
                20,
                {
                    "roadmap_snapshot": {
                        "roadmap_source": roadmap_source,
                        "github_repo": resolved.github_repo,
                        "web_urls": resolved.web_urls,
                        "item_count": n_roadmap,
                        "notes": resolved.notes,
                    }
                },
            )

            # --- reviews ---
            review_result = self.reviews.fetch_reviews(
                app["package_name"], max_reviews=max_reviews
            )
            reviews_df = review_result.df
            degraded.extend(review_result.degraded)
            review_provenance = review_result.provenance
            if review_provenance == "fixture":
                degraded.append(
                    "INTEGRITY: using fixture reviews — do not attribute quotes to real users"
                )
            if reviews_df is None or reviews_df.empty:
                raise RuntimeError(
                    f"No reviews available for {app['package_name']} "
                    "(missing parquet cache and HF fetch failed)"
                )
            total_reviews = len(reviews_df)
            review_window = compute_review_window(reviews_df)

            # Per-review need-bearing filter (keeps polite 4★ wants; drops empty praise)
            need_df, need_stats = select_need_bearing(reviews_df)
            if need_df.empty:
                raise RuntimeError(
                    f"No need-bearing reviews in {app['package_name']} "
                    f"({need_stats['reviews_total']} total)"
                )

            cb("embedding", 40)

            # --- embed + cluster need-bearing only (CPU-bound → thread) ---
            roadmap_texts = (
                []
                if roadmap_items is None or roadmap_items.empty
                else [str(t) for t in roadmap_items["text"].tolist()]
            )

            def _embed_work() -> tuple[EmbeddingEngine, dict[str, Any], Any]:
                engine = EmbeddingEngine(self.settings)
                clustered = engine.embed_and_cluster(
                    need_df, roadmap_texts=roadmap_texts
                )
                road_emb = engine.embed_roadmap_items(
                    roadmap_items if roadmap_items is not None else pd.DataFrame()
                )
                return engine, clustered, road_emb

            with ThreadPoolExecutor(max_workers=1) as pool:
                engine, clustered, road_emb = pool.submit(_embed_work).result()

            degraded.extend(engine.degraded)
            backend_name = engine.backend_name
            clusters = clustered["clusters"]
            n_clusters = len(clusters)
            reviews_with_clusters = clustered["reviews_df"]
            review_emb = clustered["embeddings"]

            cb("clustering", 55)
            cb("matching", 65)

            matching_space = None
            threshold = self.settings.active_match_threshold(backend_name)
            margin = self.settings.active_match_margin(backend_name)
            if (
                self.settings.roadmap_matching_enabled
                and roadmap_source != "none"
                and roadmap_texts
            ):
                matching_space = build_matching_space(
                    roadmap_texts, self.settings
                )
                threshold = float(matching_space.threshold)
                if matching_space.null.n_control_clusters == 0:
                    degraded.append(
                        "null-model: no control reviews; using absolute TF-IDF floor"
                    )

            analyzer = GapMatrix(
                self.settings,
                match_threshold=threshold,
                match_margin=margin,
                matching_space=matching_space,
            )
            candidates = analyzer.analyze(
                clusters=clusters,
                review_embeddings=review_emb,
                reviews_df=reviews_with_clusters,
                roadmap_items=roadmap_items
                if roadmap_items is not None
                else pd.DataFrame(),
                roadmap_embeddings=road_emb,
                roadmap_source=roadmap_source,
                total_reviews=total_reviews,
                review_window=review_window,
            )

            cb("extracting", 80)

            reviews_by_id = {
                str(r["review_id"]): r
                for r in reviews_with_clusters.to_dict(orient="records")
            }
            extracted = self.extractor.extract_all_sync(
                candidates, reviews_by_id, roadmap_source, top_n=5
            )
            degraded.extend(self.extractor.degraded)
            llm_used = any(e.llm_used for e in extracted)

            cb("persisting", 90)

            gap_rows = self._to_gap_rows(extracted, reviews_by_id, roadmap_source)
            # Hard rule 1: drop gaps with zero evidence, re-rank
            gap_rows = [g for g in gap_rows if g.get("evidence")]
            for i, g in enumerate(gap_rows, start=1):
                g["rank"] = i
            gap_rows = gap_rows[:5]

            written = self.store.write_gaps_with_evidence(job_id, gap_rows)

            elapsed = round(time.perf_counter() - t0, 3)
            window_meta = review_window.to_iso()
            stats = {
                "total_reviews": total_reviews,
                "reviews_total": need_stats["reviews_total"],
                "reviews_need_bearing": need_stats["reviews_need_bearing"],
                "clusters": n_clusters,
                "roadmap_items": n_roadmap,
                "llm_used": llm_used,
                "embedding_backend": backend_name,
                "elapsed_s": elapsed,
                "degraded": sorted(set(degraded)),
                "match_threshold": threshold,
                "match_threshold_source": (
                    "null_model" if matching_space is not None else "absolute"
                ),
                "null_percentile": getattr(self.settings, "null_percentile", 95.0),
                "null_control_clusters": (
                    matching_space.null.n_control_clusters
                    if matching_space is not None
                    else 0
                ),
                "review_provenance": review_provenance,
                **window_meta,
            }
            summary = _summary_text(roadmap_source, written)

            cb(
                "done",
                100,
                {
                    "status": "completed",
                    "summary": summary,
                    "stats": stats,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "roadmap_snapshot": {
                        "roadmap_source": roadmap_source,
                        "github_repo": resolved.github_repo,
                        "web_urls": resolved.web_urls,
                        "item_count": n_roadmap,
                        "notes": resolved.notes,
                        **window_meta,
                    },
                },
            )
            job = self.store.get_job(job_id)
            assert job is not None
            return job

        except Exception as e:
            elapsed = round(time.perf_counter() - t0, 3)
            stats = {
                "total_reviews": total_reviews,
                "clusters": n_clusters,
                "roadmap_items": n_roadmap,
                "llm_used": llm_used,
                "embedding_backend": backend_name,
                "elapsed_s": elapsed,
                "degraded": sorted(set(degraded + [str(e)])),
            }
            self.store.update_job(
                job_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "progress": 100,
                    "error": str(e),
                    "stats": stats,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            job = self.store.get_job(job_id)
            assert job is not None
            return job

    def _to_gap_rows(
        self,
        extracted: list[ExtractedGap],
        reviews_by_id: dict[str, dict[str, Any]],
        roadmap_source: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rank, ex in enumerate(extracted, start=1):
            evidence = []
            cite_ids = ex.cited_review_ids or ex.review_ids[:3]
            for rid in cite_ids:
                rev = reviews_by_id.get(rid)
                if not rev:
                    continue
                evidence.append(
                    {
                        "evidence_id": rid,
                        "source_type": "review",
                        "title": f"Review {rid}",
                        "snippet": str(rev.get("review_text") or "")[:280],
                        "url": None,
                        "payload": {
                            "rating": rev.get("rating"),
                            "created_at": rev.get("created_at"),
                            **{
                                k: ex.metrics[k]
                                for k in (
                                    "components",
                                    "weights",
                                    "deterministic_confidence",
                                    "cluster_size",
                                    "total_reviews",
                                    "best_similarity",
                                    "mean_rating",
                                    "rating_spread",
                                    "cohesion",
                                )
                                if k in ex.metrics
                            },
                        },
                    }
                )
            if ex.matched_item and ex.matched_item.get("url"):
                evidence.append(
                    {
                        "evidence_id": str(
                            ex.matched_item.get("item_id")
                            or ex.matched_item.get("issue_id")
                            or "matched"
                        ),
                        "source_type": (
                            "github_issue"
                            if (ex.matched_item.get("source") == "github"
                                or str(ex.matched_item.get("issue_id") or "").startswith(
                                    "issue"
                                )
                                or str(ex.matched_item.get("item_id") or "").startswith(
                                    "issue"
                                ))
                            else "web_page"
                        ),
                        "title": (ex.matched_item.get("text") or "")[:120],
                        "snippet": (ex.matched_item.get("text") or "")[:280],
                        "url": ex.matched_item.get("url"),
                        "payload": {
                            "state": ex.matched_item.get("state"),
                            "milestone_title": ex.matched_item.get("milestone_title"),
                        },
                    }
                )
            if not evidence:
                continue
            rows.append(
                {
                    "rank": rank,
                    "need": ex.latent_need,
                    "one_sentence_summary": ex.one_sentence_summary,
                    "verdict": ex.verdict,
                    "confidence": ex.confidence,
                    "confidence_rationale": ex.confidence_rationale,
                    "latent_reasoning": ex.latent_reasoning,
                    "need_source": ex.need_source,
                    "metrics": ex.metrics,
                    "evidence": evidence,
                }
            )
        return rows


def _summary_text(roadmap_source: str, gaps: list[dict[str, Any]]) -> str:
    if not gaps:
        return "No latent unmet needs surfaced from the available reviews."
    if roadmap_source == "none":
        return (
            f"Surfaced {len(gaps)} needs with no public roadmap to verify against "
            f"(top: {gaps[0].get('need', '')})."
        )
    return (
        f"Found {len(gaps)} roadmap gaps (mode={roadmap_source}); "
        f"top verdict {gaps[0].get('verdict')} — {gaps[0].get('need', '')}."
    )


def main() -> None:
    from src.store import LocalJsonStore, load_catalog

    settings = get_settings()
    store = LocalJsonStore(settings=settings)
    catalog = load_catalog(settings)
    store.seed_apps_from_catalog(catalog)
    app = store.get_app_by_package("de.danoeh.antennapod")
    if not app:
        print("AntennaPod not in catalog")
        return
    job = store.create_job(
        {
            "app_id": app["id"],
            "config_hash": config_hash(app["id"], 200, settings),
            "status": "queued",
            "stage": "queued",
        }
    )
    pipe = AnalysisPipeline(store, settings)
    result = pipe.run(job["id"], app, max_reviews=200)
    print(
        {
            "status": result["status"],
            "stage": result["stage"],
            "gaps": len(result["gaps"]),
            "roadmap_source": result["roadmap_source"],
            "stats": result["stats"],
        }
    )


if __name__ == "__main__":
    main()
